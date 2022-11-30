import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import List, Tuple, Optional

from allennlp.commands.predict import _predict
from allennlp.commands.train import train_model
from allennlp.common import Params
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_utils import Data, SentimentTriple, SplitEnum, Sentence, LabelEnum
from utils import safe_divide


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


class SpanModel(BaseModel):
    save_dir: str
    random_seed: int
    path_config_base: str = "training_config/config.jsonnet"

    def save_temp_data(self, path_in: str, name: str, is_test: bool = False) -> Path:
        path_temp = Path(self.save_dir) / "temp_data" / f"{name}.json"
        path_temp = path_temp.resolve()
        path_temp.parent.mkdir(exist_ok=True, parents=True)
        data = Data.load_from_full_path(path_in)

        if is_test:
            # SpanModel error if s.triples is empty list
            assert data.sentences is not None
            for s in data.sentences:
                s.triples = [SentimentTriple.make_dummy()]

        span_data = SpanModelData.from_data(data)
        span_data.dump(path_temp)
        return path_temp

    def fit(self, path_train: str, path_dev: str):
        weights_dir = Path(self.save_dir) / "weights"
        weights_dir.mkdir(exist_ok=True, parents=True)
        print(dict(weights_dir=weights_dir))

        params = Params.from_file(
            self.path_config_base,
            params_overrides=dict(
                random_seed=self.random_seed,
                numpy_seed=self.random_seed,
                pytorch_seed=self.random_seed,
                train_data_path=str(self.save_temp_data(path_train, "train")),
                validation_data_path=str(self.save_temp_data(path_dev, "dev")),
                test_data_path=str(self.save_temp_data(path_dev, "dev")),
            ),
        )

        # Register custom modules
        sys.path.append(".")
        from span_model.data.dataset_readers.span_model import SpanModelReader

        assert SpanModelReader is not None
        train_model(params, serialization_dir=str(weights_dir))

    def predict(self, path_in: str, path_out: str):
        path_model = Path(self.save_dir) / "weights" / "model.tar.gz"
        path_temp_in = self.save_temp_data(path_in, "pred_in", is_test=True)
        path_temp_out = Path(self.save_dir) / "temp_data" / "pred_out.json"
        if path_temp_out.exists():
            os.remove(path_temp_out)

        args = Namespace(
            archive_file=str(path_model),
            input_file=str(path_temp_in),
            output_file=str(path_temp_out),
            weights_file="",
            batch_size=1,
            silent=True,
            cuda_device=0,
            use_dataset_reader=True,
            dataset_reader_choice="validation",
            overrides="",
            predictor="span_model",
            file_friendly_logging=False,
        )

        # Register custom modules
        sys.path.append(".")
        from span_model.data.dataset_readers.span_model import SpanModelReader
        from span_model.predictors.span_model import SpanModelPredictor

        assert SpanModelReader is not None
        assert SpanModelPredictor is not None
        _predict(args)

        with open(path_temp_out) as f:
            preds = [SpanModelPrediction(**json.loads(line.strip())) for line in f]
        data = Data(
            root=Path(),
            data_split=SplitEnum.test,
            sentences=[p.to_sentence() for p in preds],
        )
        data.save_to_path(path_out)

    @classmethod
    def score(cls, path_pred: str, path_gold: str) -> dict:
        pred = Data.load_from_full_path(path_pred)
        gold = Data.load_from_full_path(path_gold)
        assert pred.sentences is not None
        assert gold.sentences is not None
        assert len(pred.sentences) == len(gold.sentences)
        num_pred = 0
        num_gold = 0
        num_correct = 0

        for i in range(len(gold.sentences)):
            num_pred += len(pred.sentences[i].triples)
            num_gold += len(gold.sentences[i].triples)
            for p in pred.sentences[i].triples:
                for g in gold.sentences[i].triples:
                    if p.dict() == g.dict():
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        info = dict(
            path_pred=path_pred,
            path_gold=path_gold,
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
        )
        return info


def run_score(path_pred: str, path_gold: str) -> dict:
    return SpanModel.score(path_pred, path_gold)


def run_train(path_train: str, path_dev: str, save_dir: str, random_seed: int):
    print(dict(run_train=locals()))
    if Path(save_dir).exists():
        return

    model = SpanModel(save_dir=save_dir, random_seed=random_seed)
    model.fit(path_train, path_dev)


def run_train_many(save_dir_template: str, random_seeds: List[int], **kwargs):
    for seed in tqdm(random_seeds):
        save_dir = save_dir_template.format(seed)
        run_train(save_dir=save_dir, random_seed=seed, **kwargs)


def run_eval(path_test: str, save_dir: str):
    print(dict(run_eval=locals()))
    model = SpanModel(save_dir=save_dir, random_seed=0)
    path_pred = str(Path(save_dir) / "pred.txt")
    model.predict(path_test, path_pred)
    results = model.score(path_pred, path_test)
    print(results)
    return results


def run_eval_many(save_dir_template: str, random_seeds: List[int], **kwargs):
    results = []
    for seed in tqdm(random_seeds):
        save_dir = save_dir_template.format(seed)
        results.append(run_eval(save_dir=save_dir, **kwargs))

    precision = sum(r["precision"] for r in results) / len(random_seeds)
    recall = sum(r["recall"] for r in results) / len(random_seeds)
    score = safe_divide(2 * precision * recall, precision + recall)
    print(dict(precision=precision, recall=recall, score=score))


if __name__ == "__main__":
    Fire()
