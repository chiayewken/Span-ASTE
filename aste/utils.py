import copy
import hashlib
import pickle
import subprocess
import time
from pathlib import Path
from typing import List, Set, Tuple, Union

from fire import Fire
from pydantic import BaseModel


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
    return dict(
        min=min(numbers),
        max=max(numbers),
        avg=sum(numbers) / len(numbers),
    )


def count_joins(spans: Set[Tuple[int, int]]) -> int:
    count = 0
    for a_start, a_end in spans:
        for b_start, b_end in spans:
            if (a_start, a_end) == (b_start, b_end):
                continue

            if b_start <= a_start <= b_end + 1 or b_start - 1 <= a_end <= b_end:
                count += 1
    return count // 2


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


def clean_up_triplet_data(path: str):
    outputs = []
    with open(path) as f:
        for line in f:
            sep = "####"
            text, tags_t, tags_o, triplets = line.split(sep)
            outputs.append(sep.join([text, " ", " ", triplets]))

    with open(path, "w") as f:
        f.write("".join(outputs))


def clean_up_many(pattern: str = "data/triplet_data/*/*.txt"):
    for path in sorted(Path().glob(pattern)):
        print(path)
        clean_up_triplet_data(str(path))


def merge_data(
    folders_in: List[str] = [
        "aste/data/triplet_data/14res/",
        "aste/data/triplet_data/15res/",
        "aste/data/triplet_data/16res/",
    ],
    folder_out: str = "aste/data/triplet_data/res_all/",
):
    for name in ["train.txt", "dev.txt", "test.txt"]:
        outputs = []
        for folder in folders_in:
            path = Path(folder) / name
            with open(path) as f:
                for line in f:
                    assert line.endswith("\n")
                    outputs.append(line)

        path_out = Path(folder_out) / name
        path_out.parent.mkdir(exist_ok=True, parents=True)
        with open(path_out, "w") as f:
            f.write("".join(outputs))


def safe_divide(a: float, b: float) -> float:
    if a == 0 or b == 0:
        return 0
    return a / b


if __name__ == "__main__":
    Fire()
