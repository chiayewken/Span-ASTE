import logging
from typing import Any, Dict, List, Optional, Callable

import torch
from torch.nn import functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from span_model.models.shared import FocalLoss, BiAffineSingleInput
from span_model.training.ner_metrics import NERMetrics
from span_model.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NERTagger(Model):
    """
    Named entity recognition module

    Parameters
    ----------
    mention_feedforward : ``FeedForward``
        This feedforward network is applied to the span representations which is then scored
        by a linear layer.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        make_feedforward: Callable,
        span_emb_dim: int,
        regularizer: Optional[RegularizerApplicator] = None,
        use_bi_affine: bool = False,
        neg_class_weight: float = -1,
        use_focal_loss: bool = False,
        focal_loss_gamma: int = 2,
        use_double_scorer: bool = False,
        use_gold_for_train_prune_scores: bool = False,
        use_single_pool: bool = False,
        name: str = "ner_labels"
    ) -> None:
        super(NERTagger, self).__init__(vocab, regularizer)

        self.use_single_pool = use_single_pool
        self.use_gold_for_train_prune_scores = use_gold_for_train_prune_scores
        self.use_double_scorer = use_double_scorer
        self.use_bi_affine = use_bi_affine
        self._name = name
        self._namespaces = [
            entry for entry in vocab.get_namespaces() if self._name in entry
        ]

        # Number of classes determine the output dimension of the final layer
        self._n_labels = {name: vocab.get_vocab_size(name) for name in self._namespaces}
        if self.use_single_pool:
            for n in self._namespaces:
                self._n_labels[n] -= 1

        # Null label is needed to keep track of when calculating the metrics
        for namespace in self._namespaces:
            null_label = vocab.get_token_index("", namespace)
            assert (
                null_label == 0
            )  # If not, the dummy class won't correspond to the null label.

        # The output dim is 1 less than the number of labels because we don't score the null label;
        # we just give it a score of 0 by default.

        # Create a separate scorer and metric for each dataset we're dealing with.
        self._ner_scorers = torch.nn.ModuleDict()
        self._ner_metrics = {}

        for namespace in self._namespaces:
            self._ner_scorers[namespace] = self.make_scorer(
                make_feedforward, span_emb_dim, self._n_labels[namespace])

            if self.use_double_scorer:
                self._ner_scorers[namespace] = None  # noqa
                self._ner_scorers["opinion"] = self.make_scorer(make_feedforward, span_emb_dim, 2)
                self._ner_scorers["target"] = self.make_scorer(make_feedforward, span_emb_dim, 2)

            self._ner_metrics[namespace] = NERMetrics(
                self._n_labels[namespace], null_label
            )

            self.i_opinion = vocab.get_token_index("OPINION", namespace)
            self.i_target = vocab.get_token_index("TARGET", namespace)
            if self.use_single_pool:
                self.i_opinion = self.i_target = 1

        self._active_namespace = None

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")
        if neg_class_weight != -1:
            assert len(self._namespaces) == 1
            num_pos_classes = self._n_labels[self._namespaces[0]] - 1
            pos_weight = (1 - neg_class_weight) / num_pos_classes
            weight = [neg_class_weight] + [pos_weight] * num_pos_classes
            print(dict(ner_class_weight=weight))
            self._loss = torch.nn.CrossEntropyLoss(reduction="sum", weight=torch.tensor(weight))

        if use_focal_loss:
            assert neg_class_weight != -1
            self._loss = FocalLoss(
                reduction="sum", weight=torch.tensor(weight), gamma=focal_loss_gamma)
        print(dict(ner_loss_fn=self._loss))

    def make_scorer(self, make_feedforward, span_emb_dim, n_labels):
        mention_feedforward = make_feedforward(input_dim=span_emb_dim)
        scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(
                torch.nn.Linear(
                    mention_feedforward.get_output_dim(),
                    n_labels
                )
            ),
        )
        if self.use_bi_affine:
            scorer = BiAffineSingleInput(
                input_size=span_emb_dim // 2,
                project_size=200,
                output_size=n_labels,
            )
        return scorer

    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,
        span_mask: torch.IntTensor,
        span_embeddings: torch.IntTensor,
        sentence_lengths: torch.Tensor,
        ner_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        TODO: Write documentation.
        """

        # Shape: (Batch size, Number of Spans, Span Embedding Size)
        # span_embeddings

        self._active_namespace = f"{metadata.dataset}__{self._name}"
        if self.use_double_scorer:
            opinion_scores = self._ner_scorers["opinion"](span_embeddings)
            target_scores = self._ner_scorers["target"](span_embeddings)
            null_scores = torch.stack([opinion_scores[..., 0], target_scores[..., 0]], dim=-1).mean(dim=-1, keepdim=True)
            pool = [null_scores, None, None]
            pool[self.i_opinion] = opinion_scores[..., [1]]
            pool[self.i_target] = target_scores[..., [1]]
            ner_scores = torch.cat(pool, dim=-1)
        else:
            scorer = self._ner_scorers[self._active_namespace]
            ner_scores = scorer(span_embeddings)

        # Give large positive scores to "null" class in masked-out elements
        ner_scores[..., 0] = util.replace_masked_values(ner_scores[..., 0], span_mask.bool(), 1e20)
        _, predicted_ner = ner_scores.max(2)

        predictions = self.predict(
            ner_scores.detach().cpu(),
            spans.detach().cpu(),
            span_mask.detach().cpu(),
            metadata,
        )
        output_dict = {"predictions": predictions}
        # New
        output_dict.update(ner_scores=ner_scores)
        output_dict.update(opinion_scores=ner_scores.softmax(dim=-1)[..., [self.i_opinion]])
        output_dict.update(target_scores=ner_scores.softmax(dim=-1)[..., [self.i_target]])

        if ner_labels is not None:
            if self.use_single_pool:
                ner_labels = torch.ne(ner_labels, 0.0).long()

            if self.use_gold_for_train_prune_scores:
                for name, i in dict(opinion_scores=self.i_opinion, target_scores=self.i_target).items():
                    mask = ner_labels.eq(i).unsqueeze(dim=-1)
                    assert mask.shape == output_dict[name].shape
                    output_dict[name] = output_dict[name].masked_fill(mask, 1e20)

            metrics = self._ner_metrics[self._active_namespace]
            metrics(predicted_ner, ner_labels, span_mask)
            ner_scores_flat = ner_scores.view(
                -1, self._n_labels[self._active_namespace]
            )
            ner_labels_flat = ner_labels.view(-1)
            mask_flat = span_mask.view(-1).bool()

            loss = self._loss(ner_scores_flat[mask_flat], ner_labels_flat[mask_flat])

            output_dict["loss"] = loss

        return output_dict

    def predict(self, ner_scores, spans, span_mask, metadata):
        # TODO: Make sure the iteration works in documents with a single sentence.
        # Zipping up and iterating iterates over the zeroth dimension of each tensor; this
        # corresponds to iterating over sentences.
        predictions = []
        zipped = zip(ner_scores, spans, span_mask, metadata)
        for ner_scores_sent, spans_sent, span_mask_sent, sentence in zipped:
            predicted_scores_raw, predicted_labels = ner_scores_sent.max(dim=1)
            softmax_scores = F.softmax(ner_scores_sent, dim=1)
            predicted_scores_softmax, _ = softmax_scores.max(dim=1)
            ix = (predicted_labels != 0) & span_mask_sent.bool()

            predictions_sent = []
            zip_pred = zip(
                predicted_labels[ix],
                predicted_scores_raw[ix],
                predicted_scores_softmax[ix],
                spans_sent[ix],
            )
            for label, label_score_raw, label_score_softmax, label_span in zip_pred:
                label_str = self.vocab.get_token_from_index(
                    label.item(), self._active_namespace
                )
                span_start, span_end = label_span.tolist()
                ner = [
                    span_start,
                    span_end,
                    label_str,
                    label_score_raw.item(),
                    label_score_softmax.item(),
                ]
                prediction = document.PredictedNER(ner, sentence, sentence_offsets=True)
                predictions_sent.append(prediction)

            predictions.append(predictions_sent)

        return predictions

    # TODO: This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._ner_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            to_update = {
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1": f1,
            }
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__{self._name.replace('_labels', '')}_{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res
