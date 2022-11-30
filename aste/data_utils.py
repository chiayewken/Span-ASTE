import ast
import copy
import json
import os
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from fire import Fire
from pydantic import BaseModel
from sklearn.metrics import classification_report

from utils import count_joins, get_simple_stats

RawTriple = Tuple[List[int], int, int, int, int]
Span = Tuple[int, int]


class SplitEnum(str, Enum):
    train = "train"
    dev = "dev"
    test = "test"


class LabelEnum(str, Enum):
    positive = "POS"
    negative = "NEG"
    neutral = "NEU"
    opinion = "OPINION"
    target = "TARGET"

    @classmethod
    def as_list(cls):
        return [cls.neutral, cls.positive, cls.negative]

    @classmethod
    def i_to_label(cls, i: int):
        return cls.as_list()[i]

    @classmethod
    def label_to_i(cls, label) -> int:
        return cls.as_list().index(label)


class SentimentTriple(BaseModel):
    o_start: int
    o_end: int
    t_start: int
    t_end: int
    label: LabelEnum

    @classmethod
    def make_dummy(cls):
        return cls(o_start=0, o_end=0, t_start=0, t_end=0, label=LabelEnum.neutral)

    @property
    def opinion(self) -> Tuple[int, int]:
        return self.o_start, self.o_end

    @property
    def target(self) -> Tuple[int, int]:
        return self.t_start, self.t_end

    @classmethod
    def from_raw_triple(cls, x: RawTriple):
        (o_start, o_end), polarity, direction, gap_a, gap_b = x
        # Refer: TagReader
        if direction == 0:
            t_end = o_start - gap_a
            t_start = o_start - gap_b
        elif direction == 1:
            t_start = gap_a + o_start
            t_end = gap_b + o_start
        else:
            raise ValueError

        return cls(
            o_start=o_start,
            o_end=o_end,
            t_start=t_start,
            t_end=t_end,
            label=LabelEnum.i_to_label(polarity),
        )

    def to_raw_triple(self) -> RawTriple:
        polarity = LabelEnum.label_to_i(self.label)
        if self.t_start < self.o_start:
            direction = 0
            gap_a, gap_b = self.o_start - self.t_end, self.o_start - self.t_start
        else:
            direction = 1
            gap_a, gap_b = self.t_start - self.o_start, self.t_end - self.o_start
        return [self.o_start, self.o_end], polarity, direction, gap_a, gap_b

    def as_text(self, tokens: List[str]) -> str:
        opinion = " ".join(tokens[self.o_start : self.o_end + 1])
        target = " ".join(tokens[self.t_start : self.t_end + 1])
        return f"{opinion}-{target} ({self.label})"


class TripleHeuristic(BaseModel):
    @staticmethod
    def run(
        opinion_to_label: Dict[Span, LabelEnum],
        target_to_label: Dict[Span, LabelEnum],
    ) -> List[SentimentTriple]:
        # For each target, pair with the closest opinion (and vice versa)
        spans_o = list(opinion_to_label.keys())
        spans_t = list(target_to_label.keys())
        pos_o = np.expand_dims(np.array(spans_o).mean(axis=-1), axis=1)
        pos_t = np.expand_dims(np.array(spans_t).mean(axis=-1), axis=0)
        dists = np.absolute(pos_o - pos_t)
        raw_triples: Set[Tuple[int, int, LabelEnum]] = set()

        closest = np.argmin(dists, axis=1)
        for i, span in enumerate(spans_o):
            raw_triples.add((i, int(closest[i]), opinion_to_label[span]))
        closest = np.argmin(dists, axis=0)
        for i, span in enumerate(spans_t):
            raw_triples.add((int(closest[i]), i, target_to_label[span]))

        triples = []
        for i, j, label in raw_triples:
            os, oe = spans_o[i]
            ts, te = spans_t[j]
            triples.append(
                SentimentTriple(o_start=os, o_end=oe, t_start=ts, t_end=te, label=label)
            )
        return triples


class TagMaker(BaseModel):
    @staticmethod
    def run(spans: List[Span], labels: List[LabelEnum], num_tokens: int) -> List[str]:
        raise NotImplementedError


class BioesTagMaker(TagMaker):
    @staticmethod
    def run(spans: List[Span], labels: List[LabelEnum], num_tokens: int) -> List[str]:
        tags = ["O"] * num_tokens
        for (start, end), lab in zip(spans, labels):
            assert end >= start
            length = end - start + 1
            if length == 1:
                tags[start] = f"S-{lab}"
            else:
                tags[start] = f"B-{lab}"
                tags[end] = f"E-{lab}"
                for i in range(start + 1, end):
                    tags[i] = f"I-{lab}"
        return tags


class Sentence(BaseModel):
    tokens: List[str]
    pos: List[str]
    weight: int
    id: int
    is_labeled: bool
    triples: List[SentimentTriple]
    spans: List[Tuple[int, int, LabelEnum]] = []

    def extract_spans(self) -> List[Tuple[int, int, LabelEnum]]:
        spans = []
        for t in self.triples:
            spans.append((t.o_start, t.o_end, LabelEnum.opinion))
            spans.append((t.t_start, t.t_end, LabelEnum.target))
        spans = sorted(set(spans))
        return spans

    def as_text(self) -> str:
        tokens = list(self.tokens)
        for t in self.triples:
            tokens[t.o_start] = "(" + tokens[t.o_start]
            tokens[t.o_end] = tokens[t.o_end] + ")"
            tokens[t.t_start] = "[" + tokens[t.t_start]
            tokens[t.t_end] = tokens[t.t_end] + "]"
        return " ".join(tokens)

    @classmethod
    def from_line_format(cls, text: str):
        front, back = text.split("#### #### ####")
        tokens = front.split(" ")
        triples = []

        for a, b, label in ast.literal_eval(back):
            t = SentimentTriple(
                t_start=a[0],
                t_end=a[0] if len(a) == 1 else a[-1],
                o_start=b[0],
                o_end=b[0] if len(b) == 1 else b[-1],
                label=label,
            )
            triples.append(t)

        return cls(
            tokens=tokens, triples=triples, id=0, pos=[], weight=1, is_labeled=True
        )

    def to_line_format(self) -> str:
        # ([1], [4], 'POS')
        # ([1,2], [4], 'POS')
        triplets = []
        for t in self.triples:
            parts = []
            for start, end in [(t.t_start, t.t_end), (t.o_start, t.o_end)]:
                if start == end:
                    parts.append([start])
                else:
                    parts.append([start, end])
            parts.append(f"{t.label}")
            triplets.append(tuple(parts))

        line = " ".join(self.tokens) + "#### #### ####" + str(triplets) + "\n"
        assert self.from_line_format(line).tokens == self.tokens
        assert self.from_line_format(line).triples == self.triples
        return line


class Data(BaseModel):
    root: Path
    data_split: SplitEnum
    sentences: Optional[List[Sentence]]
    full_path: str = ""
    num_instances: int = -1
    opinion_offset: int = 3  # Refer: jet_o.py
    is_labeled: bool = False

    def load(self):
        if self.sentences is None:
            path = self.root / f"{self.data_split}.txt"
            if self.full_path:
                path = self.full_path

            with open(path) as f:
                self.sentences = [Sentence.from_line_format(line) for line in f]

    @classmethod
    def load_from_full_path(cls, path: str):
        data = cls(full_path=path, root=Path(path).parent, data_split=SplitEnum.train)
        data.load()
        return data

    def save_to_path(self, path: str):
        assert self.sentences is not None
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.sentences:
                f.write(s.to_line_format())

        data = Data.load_from_full_path(path)
        assert data.sentences is not None
        for i, s in enumerate(data.sentences):
            assert s.tokens == self.sentences[i].tokens
            assert s.triples == self.sentences[i].triples

    def analyze_spans(self):
        print("\nHow often is target closer to opinion than any invalid target?")
        records = []
        for s in self.sentences:
            valid_pairs = set([(a.opinion, a.target) for a in s.triples])
            for a in s.triples:
                closest = None
                for b in s.triples:
                    dist_a = abs(np.mean(a.opinion) - np.mean(a.target))
                    dist_b = abs(np.mean(a.opinion) - np.mean(b.target))
                    if dist_b <= dist_a and (a.opinion, b.target) not in valid_pairs:
                        closest = b.target

                spans = [a.opinion, a.target]
                if closest is not None:
                    spans.append(closest)

                tokens = list(s.tokens)
                for start, end in spans:
                    tokens[start] = "[" + tokens[start]
                    tokens[end] = tokens[end] + "]"

                start = min([s[0] for s in spans])
                end = max([s[1] for s in spans])
                tokens = tokens[start : end + 1]

                records.append(dict(is_closest=closest is None, text=" ".join(tokens)))
        df = pd.DataFrame(records)
        print(df["is_closest"].mean())
        print(df[~df["is_closest"]].head())

    def analyze_joined_spans(self):
        print("\nHow often are target/opinion spans joined?")
        join_targets = 0
        join_opinions = 0
        total_targets = 0
        total_opinions = 0

        for s in self.sentences:
            targets = set([t.target for t in s.triples])
            opinions = set([t.opinion for t in s.triples])
            total_targets += len(targets)
            total_opinions += len(opinions)
            join_targets += count_joins(targets)
            join_opinions += count_joins(opinions)

        print(
            dict(
                targets=join_targets / total_targets,
                opinions=join_opinions / total_opinions,
            )
        )

    def analyze_tag_counts(self):
        print("\nHow many tokens are target/opinion/none?")
        record = []
        for s in self.sentences:
            tags = [str(None) for _ in s.tokens]
            for t in s.triples:
                for i in range(t.o_start, t.o_end + 1):
                    tags[i] = "Opinion"
                for i in range(t.t_start, t.t_end + 1):
                    tags[i] = "Target"
            record.extend(tags)
        print({k: v / len(record) for k, v in Counter(record).items()})

    def analyze_span_distance(self):
        print("\nHow far is the target/opinion from each other on average?")
        distances = []
        for s in self.sentences:
            for t in s.triples:
                x_opinion = (t.o_start + t.o_end) / 2
                x_target = (t.t_start + t.t_end) / 2
                distances.append(abs(x_opinion - x_target))
        print(get_simple_stats(distances))

    def analyze_opinion_labels(self):
        print("\nFor opinion/target how often is it associated with only 1 polarity?")
        for key in ["opinion", "target"]:
            records = []
            for s in self.sentences:
                term_to_labels: Dict[Tuple[int, int], List[LabelEnum]] = {}
                for t in s.triples:
                    term_to_labels.setdefault(getattr(t, key), []).append(t.label)
                records.extend([len(set(labels)) for labels in term_to_labels.values()])
            is_single_label = [n == 1 for n in records]
            print(
                dict(
                    key=key,
                    is_single_label=sum(is_single_label) / len(is_single_label),
                    stats=get_simple_stats(records),
                )
            )

    def analyze_tag_score(self):
        print("\nIf have all target and opinion terms (unpaired), what is max f_score?")
        pred = copy.deepcopy(self.sentences)
        for s in pred:
            target_to_label = {t.target: t.label for t in s.triples}
            opinion_to_label = {t.opinion: t.label for t in s.triples}
            s.triples = TripleHeuristic().run(opinion_to_label, target_to_label)

        analyzer = ResultAnalyzer()
        analyzer.run(pred, gold=self.sentences, print_limit=0)

    def analyze_ner(self):
        print("\n How many opinion/target per sentence?")
        num_o, num_t = [], []
        for s in self.sentences:
            opinions, targets = set(), set()
            for t in s.triples:
                opinions.add((t.o_start, t.o_end))
                targets.add((t.t_start, t.t_end))
            num_o.append(len(opinions))
            num_t.append(len(targets))
        print(
            dict(
                num_o=get_simple_stats(num_o),
                num_t=get_simple_stats(num_t),
                sentences=len(self.sentences),
            )
        )

    def analyze_direction(self):
        print("\n For targets, is opinion offset always positive/negative/both?")
        records = []
        for s in self.sentences:
            span_to_offsets = {}
            for t in s.triples:
                off = np.mean(t.target) - np.mean(t.opinion)
                span_to_offsets.setdefault(t.opinion, []).append(off)
            for span, offsets in span_to_offsets.items():
                labels = [
                    LabelEnum.positive if off > 0 else LabelEnum.negative
                    for off in offsets
                ]
                lab = labels[0] if len(set(labels)) == 1 else LabelEnum.neutral
                records.append(
                    dict(
                        span=" ".join(s.tokens[span[0] : span[1] + 1]),
                        text=s.as_text(),
                        offsets=lab,
                    )
                )
        df = pd.DataFrame(records)
        print(df["offsets"].value_counts(normalize=True))
        df = df[df["offsets"] == LabelEnum.neutral].drop(columns=["offsets"])
        with pd.option_context("display.max_colwidth", 999):
            print(df.head())

    def analyze(self):
        triples = [t for s in self.sentences for t in s.triples]
        info = dict(
            root=self.root,
            sentences=len(self.sentences),
            sentiments=Counter([t.label for t in triples]),
            target_lengths=get_simple_stats(
                [abs(t.t_start - t.t_end) + 1 for t in triples]
            ),
            opinion_lengths=get_simple_stats(
                [abs(t.o_start - t.o_end) + 1 for t in triples]
            ),
            sentence_lengths=get_simple_stats([len(s.tokens) for s in self.sentences]),
        )
        for k, v in info.items():
            print(k, v)

        self.analyze_direction()
        self.analyze_ner()
        self.analyze_spans()
        self.analyze_joined_spans()
        self.analyze_tag_counts()
        self.analyze_span_distance()
        self.analyze_opinion_labels()
        self.analyze_tag_score()
        print("#" * 80)


def test_save_to_path(path: str = "aste/data/triplet_data/14lap/train.txt"):
    print("\nEnsure that Data.save_to_path works properly")
    path_temp = "temp.txt"
    data = Data.load_from_full_path(path)
    data.save_to_path(path_temp)
    print("\nSamples")
    with open(path_temp) as f:
        for line in f.readlines()[:5]:
            print(line)
    os.remove(path_temp)


def merge_data(items: List[Data]) -> Data:
    merged = Data(root=Path(), data_split=items[0].data_split, sentences=[])
    for data in items:
        data.load()
        merged.sentences.extend(data.sentences)
    return merged


class Result(BaseModel):
    num_sentences: int
    num_pred: int = 0
    num_gold: int = 0
    num_correct: int = 0
    num_start_correct: int = 0
    num_start_end_correct: int = 0
    num_opinion_correct: int = 0
    num_target_correct: int = 0
    num_span_overlap: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f_score: float = 0.0


class ResultAnalyzer(BaseModel):
    @staticmethod
    def check_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return (b_start <= a_start <= b_end) or (b_start <= a_end <= b_end)

    @staticmethod
    def run_sentence(pred: Sentence, gold: Sentence):
        assert pred.tokens == gold.tokens
        triples_gold = set([t.as_text(gold.tokens) for t in gold.triples])
        triples_pred = set([t.as_text(pred.tokens) for t in pred.triples])
        tp = triples_pred.intersection(triples_gold)
        fp = triples_pred.difference(triples_gold)
        fn = triples_gold.difference(triples_pred)
        if fp or fn:
            print(dict(gold=gold.as_text()))
            print(dict(pred=pred.as_text()))
            print(dict(tp=tp))
            print(dict(fp=fp))
            print(dict(fn=fn))
            print("#" * 80)

    @staticmethod
    def analyze_labels(pred: List[Sentence], gold: List[Sentence]):
        y_pred = []
        y_gold = []
        for i in range(len(pred)):
            for p in pred[i].triples:
                for g in gold[i].triples:
                    if (p.opinion, p.target) == (g.opinion, g.target):
                        y_pred.append(str(p.label))
                        y_gold.append(str(g.label))

        print(dict(num_span_correct=len(y_pred)))
        if y_pred:
            print(classification_report(y_gold, y_pred))

    @staticmethod
    def analyze_spans(pred: List[Sentence], gold: List[Sentence]):
        num_triples_gold, triples_found_o, triples_found_t = 0, set(), set()
        for label in [LabelEnum.opinion, LabelEnum.target]:
            num_correct, num_pred, num_gold = 0, 0, 0
            is_target = {LabelEnum.opinion: False, LabelEnum.target: True}[label]
            for i, (p, g) in enumerate(zip(pred, gold)):
                spans_gold = set(g.spans if g.spans else g.extract_spans())
                spans_pred = set(p.spans if p.spans else p.extract_spans())
                spans_gold = set([s for s in spans_gold if s[-1] == label])
                spans_pred = set([s for s in spans_pred if s[-1] == label])

                num_gold += len(spans_gold)
                num_pred += len(spans_pred)
                num_correct += len(spans_gold.intersection(spans_pred))

                for t in g.triples:
                    num_triples_gold += 1
                    span = (t.target if is_target else t.opinion) + (label,)
                    if span in spans_pred:
                        t_unique = (i,) + tuple(t.dict().items())
                        if is_target:
                            triples_found_t.add(t_unique)
                        else:
                            triples_found_o.add(t_unique)

            if num_correct and num_pred and num_gold:
                p = round(num_correct / num_pred, ndigits=4)
                r = round(num_correct / num_gold, ndigits=4)
                f = round(2 * p * r / (p + r), ndigits=4)
                info = dict(label=label, p=p, r=r, f=f)
                print(json.dumps(info, indent=2))

        assert num_triples_gold % 2 == 0  # Was double-counted above
        num_triples_gold = num_triples_gold // 2
        num_triples_pred_ceiling = len(triples_found_o.intersection(triples_found_t))
        triples_pred_recall_ceiling = num_triples_pred_ceiling / num_triples_gold
        print("\n What is the upper bound for RE from predicted O & T?")
        print(dict(recall=round(triples_pred_recall_ceiling, ndigits=4)))

    @classmethod
    def run(cls, pred: List[Sentence], gold: List[Sentence], print_limit=16):
        assert len(pred) == len(gold)
        cls.analyze_labels(pred, gold)

        r = Result(num_sentences=len(pred))
        for i in range(len(pred)):
            if i < print_limit:
                cls.run_sentence(pred[i], gold[i])
            r.num_pred += len(pred[i].triples)
            r.num_gold += len(gold[i].triples)
            for p in pred[i].triples:
                for g in gold[i].triples:
                    if p.dict() == g.dict():
                        r.num_correct += 1
                    if (p.o_start, p.t_start) == (g.o_start, g.t_start):
                        r.num_start_correct += 1
                    if (p.opinion, p.target) == (g.opinion, g.target):
                        r.num_start_end_correct += 1
                    if p.opinion == g.opinion:
                        r.num_opinion_correct += 1
                    if p.target == g.target:
                        r.num_target_correct += 1
                    if cls.check_overlap(*p.opinion, *g.opinion) and cls.check_overlap(
                        *p.target, *g.target
                    ):
                        r.num_span_overlap += 1

        e = 1e-9
        r.precision = round(r.num_correct / (r.num_pred + e), 4)
        r.recall = round(r.num_correct / (r.num_gold + e), 4)
        r.f_score = round(2 * r.precision * r.recall / (r.precision + r.recall + e), 3)
        print(r.json(indent=2))
        cls.analyze_spans(pred, gold)


def test_merge(root="aste/data/triplet_data"):
    unmerged = [Data(root=p, data_split=SplitEnum.train) for p in Path(root).iterdir()]
    data = merge_data(unmerged)
    data.analyze()


if __name__ == "__main__":
    Fire()
