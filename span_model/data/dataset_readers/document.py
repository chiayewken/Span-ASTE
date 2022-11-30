from span_model.models.shared import fields_to_batches, batches_to_fields
import copy
import numpy as np
import re
import json


def format_float(x):
    return round(x, 4)


class SpanCrossesSentencesError(ValueError):
    pass


def get_sentence_of_span(span, sentence_starts, doc_tokens):
    """
    Return the index of the sentence that the span is part of.
    """
    # Inclusive sentence ends
    sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_tokens - 1]
    in_between = [
        span[0] >= start and span[1] <= end
        for start, end in zip(sentence_starts, sentence_ends)
    ]
    if sum(in_between) != 1:
        raise SpanCrossesSentencesError
    the_sentence = in_between.index(True)
    return the_sentence


class Dataset:
    def __init__(self, documents):
        self.documents = documents

    def __getitem__(self, i):
        return self.documents[i]

    def __len__(self):
        return len(self.documents)

    def __repr__(self):
        return f"Dataset with {self.__len__()} documents."

    @classmethod
    def from_jsonl(cls, fname):
        documents = []
        with open(fname, "r") as f:
            for line in f:
                doc = Document.from_json(json.loads(line))
                documents.append(doc)

        return cls(documents)

    def to_jsonl(self, fname):
        to_write = [doc.to_json() for doc in self]
        with open(fname, "w") as f:
            for entry in to_write:
                print(json.dumps(entry), file=f)


class Document:
    def __init__(
        self,
        doc_key,
        dataset,
        sentences,
        weight=None,
    ):
        self.doc_key = doc_key
        self.dataset = dataset
        self.sentences = sentences
        self.weight = weight

    @classmethod
    def from_json(cls, js):
        "Read in from json-loaded dict."
        cls._check_fields(js)
        doc_key = js["doc_key"]
        dataset = js.get("dataset")
        entries = fields_to_batches(
            js,
            [
                "doc_key",
                "dataset",
                "weight",
            ],
        )
        sentence_lengths = [len(entry["sentences"]) for entry in entries]
        sentence_starts = np.cumsum(sentence_lengths)
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        sentence_starts = sentence_starts.tolist()
        sentences = [
            Sentence(entry, sentence_start, sentence_ix)
            for sentence_ix, (entry, sentence_start) in enumerate(
                zip(entries, sentence_starts)
            )
        ]

        # Get the loss weight for this document.
        weight = js.get("weight", None)

        return cls(
            doc_key,
            dataset,
            sentences,
            weight,
        )

    @staticmethod
    def _check_fields(js):
        "Make sure we only have allowed fields."
        allowed_field_regex = (
            "doc_key|dataset|sentences|weight|.*ner$|"
            ".*relations$|.*clusters$|.*events$|^_.*"
        )
        allowed_field_regex = re.compile(allowed_field_regex)
        unexpected = []
        for field in js.keys():
            if not allowed_field_regex.match(field):
                unexpected.append(field)

        if unexpected:
            msg = f"The following unexpected fields should be prefixed with an underscore: {', '.join(unexpected)}."
            raise ValueError(msg)

    def to_json(self):
        "Write to json dict."
        res = {"doc_key": self.doc_key, "dataset": self.dataset}
        sents_json = [sent.to_json() for sent in self]
        fields_json = batches_to_fields(sents_json)
        res.update(fields_json)

        if self.weight is not None:
            res["weight"] = self.weight

        return res

    # TODO: Write a unit test to make sure this does the correct thing.
    def split(self, max_tokens_per_doc):
        """
        Greedily split a long document into smaller documents, each shorter than
        `max_tokens_per_doc`. Each split document will get the same weight as its parent.
        """
        # If the document is already short enough, return it as a list with a single item.
        if self.n_tokens <= max_tokens_per_doc:
            return [self]

        sentences = copy.deepcopy(self.sentences)

        sentence_groups = []
        current_group = []
        group_length = 0
        sentence_tok_offset = 0
        sentence_ix_offset = 0
        for sentence in sentences:
            # Can't deal with single sentences longer than the limit.
            if len(sentence) > max_tokens_per_doc:
                msg = f"Sentence \"{''.join(sentence.text)}\" has more than {max_tokens_per_doc} tokens. Please split this sentence."
                raise ValueError(msg)

            if group_length + len(sentence) <= max_tokens_per_doc:
                # If we're not at the limit, add it to the current sentence group.
                sentence.sentence_start -= sentence_tok_offset
                sentence.sentence_ix -= sentence_ix_offset
                current_group.append(sentence)
                group_length += len(sentence)
            else:
                # Otherwise, start a new sentence group and adjust sentence offsets.
                sentence_groups.append(current_group)
                sentence_tok_offset = sentence.sentence_start
                sentence_ix_offset = sentence.sentence_ix
                sentence.sentence_start -= sentence_tok_offset
                sentence.sentence_ix -= sentence_ix_offset
                current_group = [sentence]
                group_length = len(sentence)

        # Add the final sentence group.
        sentence_groups.append(current_group)

        # Create a separate document for each sentence group.
        doc_keys = [f"{self.doc_key}_SPLIT_{i}" for i in range(len(sentence_groups))]
        res = [
            self.__class__(
                doc_key,
                self.dataset,
                sentence_group,
                self.weight,
            )
            for doc_key, sentence_group in zip(doc_keys, sentence_groups)
        ]

        return res

    def __repr__(self):
        return "\n".join(
            [
                str(i) + ": " + " ".join(sent.text)
                for i, sent in enumerate(self.sentences)
            ]
        )

    def __getitem__(self, ix):
        return self.sentences[ix]

    def __len__(self):
        return len(self.sentences)

    def print_plaintext(self):
        for sent in self:
            print(" ".join(sent.text))

    @property
    def n_tokens(self):
        return sum([len(sent) for sent in self.sentences])


class Sentence:
    def __init__(self, entry, sentence_start, sentence_ix):
        self.sentence_start = sentence_start
        self.sentence_ix = sentence_ix
        self.text = entry["sentences"]

        # Metadata fields are prefixed with a `_`.
        self.metadata = {k: v for k, v in entry.items() if re.match("^_", k)}

        if "ner" in entry:
            self.ner = [NER(this_ner, self) for this_ner in entry["ner"]]
            self.ner_dict = {entry.span.span_sent: entry.label for entry in self.ner}
        else:
            self.ner = None
            self.ner_dict = None

        # Predicted ner.
        if "predicted_ner" in entry:
            self.predicted_ner = [
                PredictedNER(this_ner, self) for this_ner in entry["predicted_ner"]
            ]
        else:
            self.predicted_ner = None

        # Store relations.
        if "relations" in entry:
            self.relations = [
                Relation(this_relation, self) for this_relation in entry["relations"]
            ]
            relation_dict = {}
            for rel in self.relations:
                key = (rel.pair[0].span_sent, rel.pair[1].span_sent)
                relation_dict[key] = rel.label
            self.relation_dict = relation_dict
        else:
            self.relations = None
            self.relation_dict = None

        # Predicted relations.
        if "predicted_relations" in entry:
            self.predicted_relations = [
                PredictedRelation(this_relation, self)
                for this_relation in entry["predicted_relations"]
            ]
        else:
            self.predicted_relations = None

    def to_json(self):
        res = {"sentences": self.text}
        if self.ner is not None:
            res["ner"] = [entry.to_json() for entry in self.ner]
        if self.predicted_ner is not None:
            res["predicted_ner"] = [entry.to_json() for entry in self.predicted_ner]
        if self.relations is not None:
            res["relations"] = [entry.to_json() for entry in self.relations]
        if self.predicted_relations is not None:
            res["predicted_relations"] = [
                entry.to_json() for entry in self.predicted_relations
            ]

        for k, v in self.metadata.items():
            res[k] = v

        return res

    def __repr__(self):
        the_text = " ".join(self.text)
        the_lengths = [len(x) for x in self.text]
        tok_ixs = ""
        for i, offset in enumerate(the_lengths):
            true_offset = offset if i < 10 else offset - 1
            tok_ixs += str(i)
            tok_ixs += " " * true_offset

        return the_text + "\n" + tok_ixs

    def __len__(self):
        return len(self.text)


class Span:
    def __init__(self, start, end, sentence, sentence_offsets=False):
        # The `start` and `end` are relative to the document. We convert them to be relative to the
        # sentence.
        self.sentence = sentence
        # Need to store the sentence text to make span objects hashable.
        self.sentence_text = " ".join(sentence.text)
        self.start_sent = start if sentence_offsets else start - sentence.sentence_start
        self.end_sent = end if sentence_offsets else end - sentence.sentence_start

    @property
    def start_doc(self):
        return self.start_sent + self.sentence.sentence_start

    @property
    def end_doc(self):
        return self.end_sent + self.sentence.sentence_start

    @property
    def span_doc(self):
        return (self.start_doc, self.end_doc)

    @property
    def span_sent(self):
        return (self.start_sent, self.end_sent)

    @property
    def text(self):
        return self.sentence.text[self.start_sent : self.end_sent + 1]

    def __repr__(self):
        return str((self.start_sent, self.end_sent, self.text))

    def __eq__(self, other):
        return (
            self.span_doc == other.span_doc
            and self.span_sent == other.span_sent
            and self.sentence == other.sentence
        )

    def __hash__(self):
        tup = self.span_sent + (self.sentence_text,)
        return hash(tup)


class Token:
    def __init__(self, ix, sentence, sentence_offsets=False):
        self.sentence = sentence
        self.ix_sent = ix if sentence_offsets else ix - sentence.sentence_start

    @property
    def ix_doc(self):
        return self.ix_sent + self.sentence.sentence_start

    @property
    def text(self):
        return self.sentence.text[self.ix_sent]

    def __repr__(self):
        return str((self.ix_sent, self.text))


class NER:
    def __init__(self, ner, sentence, sentence_offsets=False):
        self.span = Span(ner[0], ner[1], sentence, sentence_offsets)
        self.label = ner[2]

    def __repr__(self):
        return f"{self.span.__repr__()}: {self.label}"

    def __eq__(self, other):
        return self.span == other.span and self.label == other.label

    def to_json(self):
        return list(self.span.span_doc) + [self.label]


class PredictedNER(NER):
    def __init__(self, ner, sentence, sentence_offsets=False):
        "The input should be a list: [span_start, span_end, label, raw_score, softmax_score]."
        super().__init__(ner, sentence, sentence_offsets)
        self.raw_score = ner[3]
        self.softmax_score = ner[4]

    def __repr__(self):
        return super().__repr__() + f" with confidence {self.softmax_score:0.4f}"

    def to_json(self):
        return super().to_json() + [
            format_float(self.raw_score),
            format_float(self.softmax_score),
        ]


class Relation:
    def __init__(self, relation, sentence, sentence_offsets=False):
        start1, end1 = relation[0], relation[1]
        start2, end2 = relation[2], relation[3]
        label = relation[4]
        span1 = Span(start1, end1, sentence, sentence_offsets)
        span2 = Span(start2, end2, sentence, sentence_offsets)
        self.pair = (span1, span2)
        self.label = label

    def __repr__(self):
        return f"{self.pair[0].__repr__()}, {self.pair[1].__repr__()}: {self.label}"

    def __eq__(self, other):
        return (self.pair == other.pair) and (self.label == other.label)

    def to_json(self):
        return list(self.pair[0].span_doc) + list(self.pair[1].span_doc) + [self.label]


class PredictedRelation(Relation):
    def __init__(self, relation, sentence, sentence_offsets=False):
        "Input format: [start_1, end_1, start_2, end_2, label, raw_score, softmax_score]."
        super().__init__(relation, sentence, sentence_offsets)
        self.raw_score = relation[5]
        self.softmax_score = relation[6]

    def __repr__(self):
        return super().__repr__() + f" with confidence {self.softmax_score:0.4f}"

    def to_json(self):
        return super().to_json() + [
            format_float(self.raw_score),
            format_float(self.softmax_score),
        ]
