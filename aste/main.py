import json
import shutil
import time
from os import remove
from pathlib import Path
from typing import List, Tuple, Optional

import _jsonnet  # noqa
import pandas as pd
from fire import Fire
from pydantic import BaseModel

from data_utils import (
    LabelEnum,
    SplitEnum,
    Sentence,
    SentimentTriple,
    Data,
    ResultAnalyzer,
)
from evaluation import nereval, LinearInstance, FScore
from utils import Shell, hash_text, update_nested_dict


class SpanModelDocument(BaseModel):
    sentences: List[List[str]]
    ner: List[List[Tuple[int, int, str]]]
    relations: List[List[Tuple[int, int, int, int, str]]]
    doc_key: str

    @property
    def is_valid(self) -> bool:
        return len(set(map(len, [self.sentences, self.ner, self.relations]))) == 1

    @classmethod
    def from_sentence(cls, x: Sentence):
        ner: List[Tuple[int, int, str]] = []
        for t in x.triples:
            ner.append((t.o_start, t.o_end, LabelEnum.opinion))
            ner.append((t.t_start, t.t_end, LabelEnum.target))
        ner = sorted(set(ner), key=lambda n: n[0])
        relations = [
            (t.o_start, t.o_end, t.t_start, t.t_end, t.label) for t in x.triples
        ]
        return cls(
            sentences=[x.tokens],
            ner=[ner],
            relations=[relations],
            doc_key=str(x.id),
        )


class SpanModelPrediction(SpanModelDocument):
    predicted_ner: List[List[Tuple[int, int, LabelEnum, float, float]]] = [
        []
    ]  # If loss_weights["ner"] == 0.0
    predicted_relations: List[List[Tuple[int, int, int, int, LabelEnum, float, float]]]

    def to_sentence(self) -> Sentence:
        for lst in [self.sentences, self.predicted_ner, self.predicted_relations]:
            assert len(lst) == 1

        triples = [
            SentimentTriple(o_start=os, o_end=oe, t_start=ts, t_end=te, label=label)
            for os, oe, ts, te, label, value, prob in self.predicted_relations[0]
        ]
        return Sentence(
            id=int(self.doc_key),
            tokens=self.sentences[0],
            pos=[],
            weight=1,
            is_labeled=False,
            triples=triples,
            spans=[lst[:3] for lst in self.predicted_ner[0]],
        )

    def update_instance(self, x: LinearInstance) -> LinearInstance:
        x.set_prediction(self.to_sentence().to_instance().output)
        return x


class SpanModelData(BaseModel):
    root: Path
    data_split: SplitEnum
    documents: Optional[List[SpanModelDocument]]

    @classmethod
    def read(cls, path: Path) -> List[SpanModelDocument]:
        docs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                raw: dict = json.loads(line)
                docs.append(SpanModelDocument(**raw))
        return docs

    def load(self):
        if self.documents is None:
            path = self.root / f"{self.data_split}.json"
            self.documents = self.read(path)

    def dump(self, path: Path, sep="\n"):
        for d in self.documents:
            assert d.is_valid
        with open(path, "w") as f:
            f.write(sep.join([d.json() for d in self.documents]))
        assert all(
            [a.dict() == b.dict() for a, b in zip(self.documents, self.read(path))]
        )

    @classmethod
    def from_data(cls, x: Data):
        data = cls(root=x.root, data_split=x.data_split)
        data.documents = [SpanModelDocument.from_sentence(s) for s in x.sentences]
        return data


class SpanModelConfigMaker(BaseModel):
    root: Path = Path("/tmp/config_maker")

    def run(self, path_in: Path, **kwargs) -> Path:
        self.root.mkdir(exist_ok=True)
        path_out = self.root / path_in.name
        config = json.loads(_jsonnet.evaluate_file(str(path_in)))
        assert isinstance(config, dict)

        for key, value in kwargs.items():
            config = update_nested_dict(config, key, value)
        with open(path_out, "w") as f:
            f.write(json.dumps(config, indent=2))
        return path_out


class SpanModelTrainer(BaseModel):
    root: Path
    train_kwargs: dict
    path_config: Path = Path("training_config/aste.jsonnet").resolve()
    repo_span_model: Path = Path(".").resolve()
    output_dir: Optional[Path]
    model_path: Optional[Path]
    data_name: Optional[str]
    task_name: Optional[str]

    @property
    def name(self) -> str:
        hash_id = hash_text(str(self.train_kwargs))
        return "_".join([self.task_name, self.data_name, hash_id])

    def load(self, overwrite: bool):
        if self.data_name is None:
            self.data_name = self.root.stem
        if self.task_name is None:
            self.task_name = self.path_config.stem
        if self.model_path is None:
            self.model_path = Path(f"models/{self.name}/model.tar.gz")
        if self.output_dir is None:
            self.output_dir = Path(f"model_outputs/{self.name}")
        if self.model_path.parent.exists() and overwrite:
            print(dict(rmtree=self.model_path.parent))
            shutil.rmtree(self.model_path.parent)
        if self.output_dir.exists() and overwrite:
            print(dict(rmtree=self.output_dir))
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(self.json(indent=2))

    def get_processed_data_path(self, data_split: SplitEnum) -> Path:
        # Should match the path in .jsonnet config file
        return self.output_dir / f"{data_split}.json"

    def get_predict_path(self, data_split: SplitEnum) -> Path:
        return self.output_dir / f"predict_{data_split}.jsonl"

    def setup_data(self):
        for data_split in [SplitEnum.train, SplitEnum.dev, SplitEnum.test]:
            data = Data(root=self.root, data_split=data_split)
            data.load()
            new = SpanModelData.from_data(data)
            new.dump(self.get_processed_data_path(data_split))

    def train(self, overwrite=True):
        self.load(overwrite=overwrite)
        if overwrite and self.model_path.exists():
            return
        self.setup_data()
        kwargs = dict(self.train_kwargs)
        data_map = dict(
            train_data_path=SplitEnum.train,
            validation_data_path=SplitEnum.dev,
            test_data_path=SplitEnum.test,
        )
        for k, v in data_map.items():
            kwargs[k] = str(self.get_processed_data_path(v).resolve())

        kwargs.setdefault("seed", 0)  # A bit sneaky to put "seed" in **kwargs but this is surgical
        seed = kwargs.pop("seed")
        for key in ["random_seed", "numpy_seed", "pytorch_seed"]:
            kwargs[key] = seed

        config_maker = SpanModelConfigMaker(root=self.output_dir)
        path_config = config_maker.run(self.path_config, **kwargs).resolve()
        shell = Shell()
        shell.run(
            f"cd {self.repo_span_model} && allennlp train {path_config}",
            serialization_dir=self.model_path.parent,
            include_package="span_model",
        )
        assert self.model_path.exists()

    def predict(self, data_split: SplitEnum) -> Path:
        self.load(overwrite=False)
        path = self.get_predict_path(data_split)
        if path.exists():
            remove(path)
        shell = Shell()
        shell.run(
            f"cd {self.repo_span_model} && allennlp predict {self.model_path}",
            self.get_processed_data_path(data_split),
            predictor="span_model",
            include_package="span_model",
            use_dataset_reader="",
            output_file=path,
            cuda_device=self.train_kwargs["trainer__cuda_device"],
            silent="",
        )
        return path

    def eval(self, data_split: SplitEnum) -> FScore:
        data = Data(root=self.root, data_split=data_split)
        data.load()
        instances = [s.to_instance() for s in data.sentences]

        path = self.predict(data_split)
        with open(path) as f:
            preds = [SpanModelPrediction(**json.loads(line.strip())) for line in f]
        for i, p in zip(instances, preds):
            p.update_instance(i)

        pred_sents = [p.to_sentence() for p in preds]
        for name, sents in dict(pred=pred_sents, gold=data.sentences).items():
            path_out = self.output_dir / f"sentences_{data_split}_{name}.json"
            print(dict(path_out=path_out))
            with open(path_out, "w") as f:
                f.write("\n".join([s.json() for s in sents]))

        scorer = nereval()
        analyzer = ResultAnalyzer()
        analyzer.run(pred=pred_sents, gold=data.sentences)
        return scorer.eval(instances)  # noqa


def main_single(path: Path, overwrite=False, **kwargs):
    trainer = SpanModelTrainer(root=path.resolve(), train_kwargs=kwargs)
    trainer.train(overwrite=overwrite)
    scores = {}
    for data_split in [SplitEnum.dev, SplitEnum.test]:
        scores[data_split] = trainer.eval(data_split=data_split)
    return scores


def main(
    root="aste/data/triplet_data",
    names=("14lap",),
    seeds=(0,),
    sep=",",
    name_out="results",
    **kwargs,
):
    print(json.dumps(locals(), indent=2))
    records = {}
    names = names if type(names) in {tuple, list} else names.split(sep)
    paths = [Path(root) / n for n in names]
    assert all([p.exists() for p in paths])
    assert len(seeds) == len(paths)

    for i, p in enumerate(paths):
        start = time.time()
        scores = main_single(p, overwrite=True, seed=seeds[i], **kwargs)
        duration = time.time() - start
        for k, v in scores.items():
            row = dict(name=p.stem, k=k, score=str(v), duration=duration)
            records.setdefault(k, []).append(row)
            df = pd.DataFrame(records[k])
            print(df)
            path = Path(f"{name_out}_{k}.csv")
            path.parent.mkdir(exist_ok=True)
            df.to_csv(path, index=False)
            print(dict(path_results=path))


if __name__ == "__main__":
    Fire(main)
