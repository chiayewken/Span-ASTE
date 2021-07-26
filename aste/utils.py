import copy
import hashlib
import pickle
import subprocess
import time
from pathlib import Path
from typing import Set, List, Tuple, Union

import pandas as pd
import spacy
from pydantic import BaseModel
from spacy.lang.en import English
from spacy.tokens import Doc, Token


class Shell(BaseModel):
    verbose: bool = True

    @classmethod
    def format_kwargs(cls, **kwargs) -> str:
        outputs = []
        for k, v in kwargs.items():
            k = k.replace("_", "-")
            k = f"--{k}"
            outputs.extend([k, str(v)])
        return " ".join(outputs)

    def run_command(self, command: str) -> str:
        # Continuously print outputs for long-running commands
        # Refer: https://fabianlee.org/2019/09/15/python-getting-live-output-from-subprocess-using-poll/
        print(dict(command=command))
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        outputs = []

        while True:
            if process.poll() is not None:
                break
            o = process.stdout.readline().decode()
            if o:
                outputs.append(o)
                if self.verbose:
                    print(o.strip())

        return "".join(outputs)

    def run(self, command: str, *args, **kwargs) -> str:
        args = [str(a) for a in args]
        command = " ".join([command] + args + [self.format_kwargs(**kwargs)])
        return self.run_command(command)


def hash_text(x: str) -> str:
    return hashlib.md5(x.encode()).hexdigest()


class Timer(BaseModel):
    name: str = ""
    start: float = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = round(time.time() - self.start, 3)
        print(f"Timer {self.name}: {duration}s")


class PickleSaver(BaseModel):
    path: Path

    def dump(self, obj):
        if not self.path.parent.exists():
            self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(obj, f)

    def load(self):
        with Timer(name=str(self.path)):
            with open(self.path, "rb") as f:
                return pickle.load(f)


class FlexiModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


def get_simple_stats(numbers: List[Union[int, float]]):
    return dict(min=min(numbers), max=max(numbers), avg=sum(numbers) / len(numbers),)


def count_joins(spans: Set[Tuple[int, int]]) -> int:
    count = 0
    for a_start, a_end in spans:
        for b_start, b_end in spans:
            if (a_start, a_end) == (b_start, b_end):
                continue

            if b_start <= a_start <= b_end + 1 or b_start - 1 <= a_end <= b_end:
                count += 1
    return count // 2


def test_spacy():
    texts = [
        "Autonomous cars are bad because they shift liability to manufacturers.",
        "I enjoyed this book very much.",
        "The design is nice and convenient.",
        "this speaker sucks",
        "I was disappointed in this.",
    ]
    nlp: English = spacy.load("en_core_web_sm")
    token: Token
    doc: Doc
    for doc in nlp.pipe(texts):
        records = []
        for token in doc:
            records.append(
                dict(
                    i=token.i,
                    text=token.text,
                    pos=token.pos_,
                    dep=token.dep_,
                    head=token.head,
                    head_pos=token.head.pos_,
                    children=list(token.children),
                )
            )
        print(pd.DataFrame(records))
        print(dict(chunks=list(doc.noun_chunks)))
        print("#" * 80)


def update_nested_dict(d: dict, k: str, v, i=0, sep="__"):
    d = copy.deepcopy(d)
    keys = k.split(sep)
    assert keys[i] in d.keys(), str(dict(keys=keys, d=d, i=i))
    if i == len(keys) - 1:
        orig = d[keys[i]]
        if v != orig:
            print(dict(updated_key=k, new_value=v, orig=orig))
            d[keys[i]] = v
    else:
        d[keys[i]] = update_nested_dict(d=d[keys[i]], k=k, v=v, i=i + 1)
    return d


def test_update_nested_dict():
    d = dict(top=dict(middle_a=dict(last=1), middle_b=0))
    print(update_nested_dict(d, k="top__middle_b", v=-1))
    print(update_nested_dict(d, k="top__middle_a__last", v=-1))
    print(update_nested_dict(d, k="top__middle_a__last", v=1))


if __name__ == "__main__":
    test_shell()
    test_spacy()
    test_update_nested_dict()
