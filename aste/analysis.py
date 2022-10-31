import json
import random
import sys
from pathlib import Path
from typing import List

import _jsonnet
import numpy as np
import torch
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.data import DatasetReader, Vocabulary, DataLoader
from allennlp.models import Model
from allennlp.training import Trainer
from fire import Fire
from tqdm import tqdm

from data_utils import Data, Sentence
from wrapper import SpanModel, safe_divide


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_load(
    path: str = "training_config/config.jsonnet",
    path_train: str = "outputs/14lap/seed_0/temp_data/train.json",
    path_dev: str = "outputs/14lap/seed_0/temp_data/validation.json",
    save_dir="outputs/temp",
):
    # Register custom modules
    sys.path.append(".")
    from span_model.data.dataset_readers.span_model import SpanModelReader

    assert SpanModelReader is not None
    params = Params.from_file(
        path,
        params_overrides=dict(
            train_data_path=path_train,
            validation_data_path=path_dev,
            test_data_path=path_dev,
        ),
    )

    train_model(params, serialization_dir=save_dir, force=True)
    breakpoint()

    config = json.loads(_jsonnet.evaluate_file(path))
    set_seed(config["random_seed"])
    reader = DatasetReader.from_params(Params(config["dataset_reader"]))
    data_train = reader.read(path_train)
    data_dev = reader.read(path_dev)
    vocab = Vocabulary.from_instances(data_train + data_dev)
    model = Model.from_params(Params(config["model"]), vocab=vocab)

    data_train.index_with(vocab)
    data_dev.index_with(vocab)
    trainer = Trainer.from_params(
        Params(config["trainer"]),
        model=model,
        data_loader=DataLoader.from_params(
            Params(config["data_loader"]), dataset=data_train
        ),
        validation_data_loader=DataLoader.from_params(
            Params(config["data_loader"]), dataset=data_dev
        ),
        serialization_dir=save_dir,
    )
    breakpoint()
    trainer.train()
    breakpoint()


class Scorer:
    name: str = ""

    def run(self, path_pred: str, path_gold: str) -> dict:
        pred = Data.load_from_full_path(path_pred)
        gold = Data.load_from_full_path(path_gold)
        assert pred.sentences is not None
        assert gold.sentences is not None
        assert len(pred.sentences) == len(gold.sentences)
        num_pred = 0
        num_gold = 0
        num_correct = 0

        for i in range(len(gold.sentences)):
            tuples_pred = self.make_tuples(pred.sentences[i])
            tuples_gold = self.make_tuples(gold.sentences[i])
            num_pred += len(tuples_pred)
            num_gold += len(tuples_gold)
            for p in tuples_pred:
                for g in tuples_gold:
                    if p == g:
                        num_correct += 1

        precision = safe_divide(num_correct, num_pred)
        recall = safe_divide(num_correct, num_gold)

        info = dict(
            precision=precision,
            recall=recall,
            score=safe_divide(2 * precision * recall, precision + recall),
        )
        return info

    def make_tuples(self, sent: Sentence) -> List[tuple]:
        raise NotImplementedError


class SentimentTripletScorer(Scorer):
    name: str = "sentiment triplet"

    def make_tuples(self, sent: Sentence) -> List[tuple]:
        return [(t.o_start, t.o_end, t.t_start, t.t_end, t.label) for t in sent.triples]


class TripletScorer(Scorer):
    name: str = "triplet"

    def make_tuples(self, sent: Sentence) -> List[tuple]:
        return [(t.o_start, t.o_end, t.t_start, t.t_end) for t in sent.triples]


class OpinionScorer(Scorer):
    name: str = "opinion"

    def make_tuples(self, sent: Sentence) -> List[tuple]:
        return sorted(set((t.o_start, t.o_end) for t in sent.triples))


class TargetScorer(Scorer):
    name: str = "target"

    def make_tuples(self, sent: Sentence) -> List[tuple]:
        return sorted(set((t.t_start, t.t_end) for t in sent.triples))


class OrigScorer(Scorer):
    name: str = "orig"

    def make_tuples(self, sent: Sentence) -> List[tuple]:
        raise NotImplementedError

    def run(self, path_pred: str, path_gold: str) -> dict:
        model = SpanModel(save_dir="", random_seed=0)
        return model.score(path_pred, path_gold)


def run_eval_domains(
    save_dir_template: str,
    path_test_template: str,
    random_seeds: List[int] = (0, 1, 2, 3, 4),
    domain_names: List[str] = ("hotel", "restaurant", "laptop"),
):
    print(locals())
    all_results = {}

    for domain in domain_names:
        results = []
        for seed in tqdm(random_seeds):
            model = SpanModel(save_dir=save_dir_template.format(seed), random_seed=0)
            path_pred = str(Path(model.save_dir, f"pred_{domain}.txt"))
            path_test = path_test_template.format(domain)
            if not Path(path_pred).exists():
                model.predict(path_test, path_pred)
            results.append(model.score(path_pred, path_test))

        precision = sum(r["precision"] for r in results) / len(random_seeds)
        recall = sum(r["recall"] for r in results) / len(random_seeds)
        score = safe_divide(2 * precision * recall, precision + recall)
        all_results[domain] = dict(p=precision, r=recall, f=score)
        for k, v in all_results.items():
            print(k, v)


def test_scorer(path_pred: str, path_gold: str):
    for scorer in [
        OpinionScorer(),
        TargetScorer(),
        TripletScorer(),
        SentimentTripletScorer(),
        OrigScorer(),
    ]:
        print(scorer.name)
        print(scorer.run(path_pred, path_gold))


if __name__ == "__main__":
    Fire()
