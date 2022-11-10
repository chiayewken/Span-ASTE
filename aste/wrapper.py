import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import List

from allennlp.commands.predict import _predict
from allennlp.commands.train import train_model
from allennlp.common import Params
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_utils import Data, SentimentTriple, SplitEnum
from main import SpanModelData, SpanModelPrediction
from utils import safe_divide


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

    def score(self, path_pred: str, path_gold: str) -> dict:
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
