from typing import List, Tuple, Optional

import spacy
import torch
from spacy.lang.en import English
from spacy.tokens import Token, Doc
from torch import Tensor
from tqdm import tqdm

from utils import FlexiModel


class SpacyParser(FlexiModel):
    model_name = "en_core_web_sm"
    nlp: Optional[English]

    def load(self):
        if self.nlp is None:
            self.nlp = spacy.load(self.model_name)
            self.nlp.tokenizer = self.nlp.tokenizer.tokens_from_list


class PosTagger(SpacyParser):
    def run(self, sentences: List[List[str]]) -> List[List[str]]:
        self.load()
        token: Token
        return [
            [token.pos_ for token in doc]
            for doc in tqdm(self.nlp.pipe(sentences, disable=["ner"]))
        ]


class DependencyGraph(FlexiModel):
    indices: List[Tuple[int, int]]
    labels: List[str]
    tokens: List[str]
    heads: List[str]

    @property
    def num_nodes(self) -> int:
        return len(self.tokens)

    @property
    def matrix(self) -> Tensor:
        assert len(self.indices) == len(self.labels)
        x = torch.zeros(self.num_nodes, self.num_nodes)
        for i, j in self.indices:
            x[i, j] = 1
        return x


class DependencyParser(SpacyParser):
    @staticmethod
    def run_doc(d: Doc) -> DependencyGraph:
        token: Token
        graph = DependencyGraph(
            indices=[],
            labels=[],
            tokens=[token.text for token in d],
            heads=[token.head.text for token in d],
        )
        for i, token in enumerate(d):
            j = token.head.i
            # Symmetric edges
            if i != j:
                graph.indices.extend([(i, j), (j, i)])
                graph.labels.extend([token.dep_, token.dep_])
        return graph

    def run(self, sentences: List[List[str]]) -> List[DependencyGraph]:
        self.load()
        return [self.run_doc(d) for d in self.nlp.pipe(sentences, disable=["ner"])]
