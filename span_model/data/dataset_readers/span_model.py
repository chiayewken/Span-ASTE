import json
import logging
import pickle as pkl
import warnings
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import (AdjacencyField, LabelField, ListField,
                                  MetadataField, SequenceLabelField, SpanField,
                                  TextField)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from span_model.data.dataset_readers.document import Document, Sentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# New
import sys

sys.path.append("aste")
from data_utils import BioesTagMaker
from pydantic import BaseModel


class Stats(BaseModel):
    entity_total: int = 0
    entity_drop: int = 0
    relation_total: int = 0
    relation_drop: int = 0
    graph_total: int = 0
    graph_edges: int = 0
    grid_total: int = 0
    grid_paired: int = 0


class SpanModelDataException(Exception):
    pass


@DatasetReader.register("span_model")
class SpanModelReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """

    def __init__(
        self,
        max_span_width: int,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # New
        self.stats = Stats()
        self.is_train = False
        self.tag_maker = BioesTagMaker()

        print("#" * 80)

        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        self.is_train = "train" in file_path  # New
        for line in lines:
            # Loop over the documents.
            doc_text = json.loads(line)
            instance = self.text_to_instance(doc_text)
            yield instance

        # New
        print(dict(file_path=file_path, stats=self.stats))
        self.stats = Stats()

    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width

    def _process_ner(self, span_tuples, sent):
        ner_labels = [""] * len(span_tuples)

        for span, label in sent.ner_dict.items():
            if self._too_long(span):
                continue
            # New
            self.stats.entity_total += 1
            if span not in span_tuples:
                self.stats.entity_drop += 1
                continue
            ix = span_tuples.index(span)
            ner_labels[ix] = label

        return ner_labels

    def _process_tags(self, sent) -> List[str]:
        if not sent.ner_dict:
            return []
        spans, labels = zip(*sent.ner_dict.items())
        return self.tag_maker.run(spans, labels, num_tokens=len(sent.text))

    def _process_relations(self, span_tuples, sent):
        relations = []
        relation_indices = []

        # Loop over the gold spans. Look up their indices in the list of span tuples and store
        # values.
        for (span1, span2), label in sent.relation_dict.items():
            # If either span is beyond the max span width, skip it.
            if self._too_long(span1) or self._too_long(span2):
                continue
            # New
            self.stats.relation_total += 1
            if (span1 not in span_tuples) or (span2 not in span_tuples):
                self.stats.relation_drop += 1
                continue
            ix1 = span_tuples.index(span1)
            ix2 = span_tuples.index(span2)
            relation_indices.append((ix1, ix2))
            relations.append(label)

        return relations, relation_indices

    def _process_grid(self, sent):
        indices = []
        for ((a_start, a_end), (b_start, b_end)), label in sent.relation_dict.items():
            for i in [a_start, a_end]:
                for j in [b_start, b_end]:
                    indices.append((i, j))
        indices = sorted(set(indices))
        assert indices
        self.stats.grid_paired += len(indices)
        self.stats.grid_total += len(sent.text) ** 2
        return indices

    def _process_sentence(self, sent: Sentence, dataset: str):
        # Get the sentence text and define the `text_field`.
        sentence_text = [self._normalize_word(word) for word in sent.text]
        text_field = TextField(
            [Token(word) for word in sentence_text], self._token_indexers
        )

        # Enumerate spans.
        spans = []
        for start, end in enumerate_spans(
            sentence_text, max_span_width=self._max_span_width
        ):
            spans.append(SpanField(start, end, text_field))

        # New
        # spans = spans[:len(spans)//2]  # bug: deliberately truncate
        # labeled:Set[Tuple[int, int]] = set([span for span,label in sent.ner_dict.items()])
        # for span_pair, label in sent.relation_dict.items():
        #     labeled.update(span_pair)
        # existing:Set[Tuple[int, int]] = set([(s.span_start, s.span_end) for s in spans])
        # for start, end in labeled:
        #     if (start, end) not in existing:
        #         spans.append(SpanField(start, end, text_field))

        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        # Convert data to fields.
        # NOTE: The `ner_labels` and `coref_labels` would ideally have type
        # `ListField[SequenceLabelField]`, where the sequence labels are over the `SpanField` of
        # `spans`. But calling `as_tensor_dict()` fails on this specific data type. Matt G
        # recognized that this is an AllenNLP API issue and suggested that represent these as
        # `ListField[ListField[LabelField]]` instead.
        fields = {}
        fields["text"] = text_field
        fields["spans"] = span_field

        if sent.ner is not None:
            ner_labels = self._process_ner(span_tuples, sent)
            fields["ner_labels"] = ListField(
                [
                    LabelField(entry, label_namespace=f"{dataset}__ner_labels")
                    for entry in ner_labels
                ]
            )
            fields["tag_labels"] = SequenceLabelField(
                self._process_tags(sent),
                text_field,
                label_namespace=f"{dataset}__tag_labels",
            )
        if sent.relations is not None:
            relation_labels, relation_indices = self._process_relations(
                span_tuples, sent
            )
            fields["relation_labels"] = AdjacencyField(
                indices=relation_indices,
                sequence_field=span_field,
                labels=relation_labels,
                label_namespace=f"{dataset}__relation_labels",
            )
            fields["grid_labels"] = AdjacencyField(
                indices=self._process_grid(sent),
                sequence_field=text_field,
                labels=None,
                label_namespace=f"{dataset}__grid_labels",
            )

        return fields

    def _process_sentence_fields(self, doc: Document):
        # Process each sentence.
        sentence_fields = [
            self._process_sentence(sent, doc.dataset) for sent in doc.sentences
        ]

        # Make sure that all sentences have the same set of keys.
        first_keys = set(sentence_fields[0].keys())
        for entry in sentence_fields:
            if set(entry.keys()) != first_keys:
                raise SpanModelDataException(
                    f"Keys do not match across sentences for document {doc.doc_key}."
                )

        # For each field, store the data from all sentences together in a ListField.
        fields = {}
        keys = sentence_fields[0].keys()
        for key in keys:
            this_field = ListField([sent[key] for sent in sentence_fields])
            fields[key] = this_field

        return fields

    @overrides
    def text_to_instance(self, doc_text: Dict[str, Any]):
        """
        Convert a Document object into an instance.
        """
        doc = Document.from_json(doc_text)

        # Make sure there are no single-token sentences; these break things.
        sent_lengths = [len(x) for x in doc.sentences]
        if min(sent_lengths) < 2:
            msg = (
                f"Document {doc.doc_key} has a sentence with a single token or no tokens. "
                "This may break the modeling code."
            )
            warnings.warn(msg)

        fields = self._process_sentence_fields(doc)
        fields["metadata"] = MetadataField(doc)

        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
