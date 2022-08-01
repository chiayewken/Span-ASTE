from pathlib import Path
from typing import List

from fire import Fire
from tqdm import tqdm

from data_utils import Data, Sentence
from wrapper import SpanModel, safe_divide


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


"""

p aste/wrapper.py run_train_many \
--path_train ../cd-aste/outputs/data/span_aste/hotel/train.txt \
--path_dev ../cd-aste/outputs/data/span_aste/hotel/dev.txt \
--save_dir_template "outputs/hotel/seed_{}" \
--random_seeds [0,1,2,3,4]

p aste/analysis.py run_eval_domains \
--path_test_template "../cd-aste/outputs/data/span_aste/{}/test.txt" \
--save_dir_template "outputs/hotel/seed_{}"

hotel {'p': 0.6046356690646132, 'r': 0.558909090909091, 'f': 0.5808738670882764}
restaurant {'p': 0.5739233685721778, 'r': 0.5038748137108793, 'f': 0.5366227837034503}
laptop {'p': 0.41013929718152226, 'r': 0.2543720190779014, 'f': 0.3139990503533453}

################################################################################

p aste/wrapper.py run_train_many \
--path_train ../cd-aste/outputs/data/span_aste/restaurant/train.txt \
--path_dev ../cd-aste/outputs/data/span_aste/restaurant/dev.txt \
--save_dir_template "outputs/restaurant/seed_{}" \
--random_seeds [0,1,2,3,4]

p aste/analysis.py run_eval_domains \
--path_test_template "../cd-aste/outputs/data/span_aste/{}/test.txt" \
--save_dir_template "outputs/restaurant/seed_{}"

hotel {'p': 0.5420025318500602, 'r': 0.4458181818181818, 'f': 0.48922760972067975}
restaurant {'p': 0.6908746977548742, 'r': 0.6566318926974665, 'f': 0.6733182065570285}
laptop {'p': 0.4815914262002341, 'r': 0.3764705882352941, 'f': 0.4225918510795445}

################################################################################

p aste/wrapper.py run_train_many \
--path_train ../cd-aste/outputs/data/span_aste/laptop/train.txt \
--path_dev ../cd-aste/outputs/data/span_aste/laptop/dev.txt \
--save_dir_template "outputs/laptop/seed_{}" \
--random_seeds [0,1,2,3,4]

p aste/analysis.py run_eval_domains \
--path_test_template "../cd-aste/outputs/data/span_aste/{}/test.txt" \
--save_dir_template "outputs/laptop/seed_{}"

hotel {'p': 0.49103226740943046, 'r': 0.35454545454545455, 'f': 0.41177352223205876}
restaurant {'p': 0.6083371502063453, 'r': 0.5020864381520119, 'f': 0.5501285025729805}
laptop {'p': 0.6095129267678342, 'r': 0.5351351351351351, 'f': 0.5699075432675059}

"""

if __name__ == "__main__":
    Fire()
