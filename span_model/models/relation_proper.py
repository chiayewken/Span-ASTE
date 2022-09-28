import logging
from typing import Any, Dict, List, Optional, Callable

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util, RegularizerApplicator
from allennlp.modules import TimeDistributed

from span_model.training.relation_metrics import RelationMetrics
from span_model.models.entity_beam_pruner import Pruner
from span_model.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

import json
from pydantic import BaseModel


class PruneOutput(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    span_embeddings: torch.Tensor
    span_mention_scores: torch.Tensor
    num_spans_to_keep: torch.Tensor
    span_mask: torch.Tensor
    span_indices: torch.Tensor
    spans: torch.Tensor


def analyze_info(info: dict):
    for k, v in info.items():
        if isinstance(v, torch.Size):
            v = tuple(v)
        info[k] = str(v)
    logging.info(json.dumps(info, indent=2))


class DistanceEmbedder(torch.nn.Module):
    def __init__(self, dim=128, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedder = torch.nn.Embedding(self.vocab_size, self.dim)

    def to_distance_buckets(
        self, spans_a: torch.Tensor, spans_b: torch.Tensor
    ) -> torch.Tensor:
        bs, num_a, dim = spans_a.shape
        bs, num_b, dim = spans_b.shape
        assert dim == 2

        spans_a = spans_a.view(bs, num_a, 1, dim)
        spans_b = spans_b.view(bs, 1, num_b, dim)
        d_ab = torch.abs(spans_b[..., 0] - spans_a[..., 1])
        d_ba = torch.abs(spans_a[..., 0] - spans_b[..., 1])
        distances = torch.minimum(d_ab, d_ba)

        # pos_a = spans_a.float().mean(dim=-1).unsqueeze(dim=-1)  # bs, num_spans, 1
        # pos_b = spans_b.float().mean(dim=-1).unsqueeze(dim=-2)  # bs, 1, num_spans
        # distances = torch.abs(pos_a - pos_b)

        x = util.bucket_values(distances, num_total_buckets=self.vocab_size)
        # [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+]
        x = x.long()
        assert x.shape == (bs, num_a, num_b)
        return x

    def forward(self, spans_a: torch.Tensor, spans_b: torch.Tensor) -> torch.Tensor:
        buckets = self.to_distance_buckets(spans_a, spans_b)
        x = self.embedder(buckets)  # bs, num_spans, num_spans, dim
        return x


def global_max_pool1d(x: torch.Tensor) -> torch.Tensor:
    bs, seq_len, features = x.shape
    x = x.transpose(-1, -2)
    x = torch.nn.functional.adaptive_max_pool1d(x, output_size=1, return_indices=False)
    x = x.transpose(-1, -2)
    x = x.squeeze(dim=1)
    assert tuple(x.shape) == (bs, features)
    return x


def test_pool():
    x = torch.zeros(3, 100, 32)
    y = global_max_pool1d(x)
    print(dict(x=x.shape, y=y.shape))


class ProperRelationExtractor(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        make_feedforward: Callable,
        span_emb_dim: int,
        feature_size: int,
        spans_per_word: float,
        positive_label_weight: float = 1.0,
        regularizer: Optional[RegularizerApplicator] = None,
        use_distance_embeds: bool = False,
        use_pruning: bool = True,
        **kwargs,  # noqa
    ) -> None:
        super().__init__(vocab, regularizer)

        print(dict(unused_keys=kwargs.keys()))
        print(dict(locals=locals()))
        self.use_pruning = use_pruning
        self.use_distance_embeds = use_distance_embeds
        self._text_embeds: Optional[torch.Tensor] = None
        self._text_mask: Optional[torch.Tensor] = None
        self._spans_a: Optional[torch.Tensor] = None
        self._spans_b: Optional[torch.Tensor] = None

        token_emb_dim = 768
        relation_scorer_dim = 2 * span_emb_dim
        if self.use_distance_embeds:
            self.d_embedder = DistanceEmbedder()
            relation_scorer_dim += self.d_embedder.dim
        print(
            dict(
                token_emb_dim=token_emb_dim,
                span_emb_dim=span_emb_dim,
                relation_scorer_dim=relation_scorer_dim,
            )
        )

        self._namespaces = [
            entry for entry in vocab.get_namespaces() if "relation_labels" in entry
        ]
        self._n_labels = {name: vocab.get_vocab_size(name) for name in self._namespaces}
        assert len(self._n_labels) == 1
        n_labels = list(self._n_labels.values())[0] + 1

        self._mention_pruners = torch.nn.ModuleDict()
        self._relation_feedforwards = torch.nn.ModuleDict()
        self._relation_scorers = torch.nn.ModuleDict()
        self._relation_metrics = {}

        self._pruner_o = self._make_pruner(span_emb_dim, make_feedforward)
        self._pruner_t = self._make_pruner(span_emb_dim, make_feedforward)
        if not self.use_pruning:
            self._pruner_o, self._pruner_t = None, None

        for namespace in self._namespaces:
            relation_feedforward = make_feedforward(input_dim=relation_scorer_dim)
            self._relation_feedforwards[namespace] = relation_feedforward
            relation_scorer = torch.nn.Linear(
                relation_feedforward.get_output_dim(), self._n_labels[namespace] + 1
            )
            self._relation_scorers[namespace] = relation_scorer

            self._relation_metrics[namespace] = RelationMetrics()

        self._spans_per_word = spans_per_word
        self._active_namespace = None
        self._loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
        print(dict(relation_loss_fn=self._loss))

    def _make_pruner(self, span_emb_dim: int, make_feedforward: Callable):
        mention_feedforward = make_feedforward(input_dim=span_emb_dim)

        feedforward_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)),
        )
        return Pruner(feedforward_scorer, use_external_score=True)

    @overrides
    def forward(
        self,  # type: ignore
        spans: torch.IntTensor,
        span_mask,
        span_embeddings,  # TODO: add type.
        sentence_lengths,
        relation_labels: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        self._active_namespace = f"{metadata.dataset}__relation_labels"

        pruned_o: PruneOutput = self._prune_spans(
            spans, span_mask, span_embeddings, sentence_lengths, "opinion"
        )
        pruned_t: PruneOutput = self._prune_spans(
            spans, span_mask, span_embeddings, sentence_lengths, "target"
        )
        relation_scores = self._compute_relation_scores(pruned_o, pruned_t)

        prediction_dict, predictions = self.predict(
            spans_a=pruned_o.spans.detach().cpu(),
            spans_b=pruned_t.spans.detach().cpu(),
            relation_scores=relation_scores.detach().cpu(),
            num_keep_a=pruned_o.num_spans_to_keep.detach().cpu(),
            num_keep_b=pruned_t.num_spans_to_keep.detach().cpu(),
            metadata=metadata,
        )

        output_dict = {"predictions": predictions}

        # Evaluate loss and F1 if labels were provided.
        if relation_labels is not None:
            # Compute cross-entropy loss.
            gold_relations = self._get_pruned_gold_relations(
                relation_labels, pruned_o, pruned_t
            )

            self._relation_scores, self._gold_relations = (
                relation_scores,
                gold_relations,
            )
            cross_entropy = self._get_cross_entropy_loss(
                relation_scores, gold_relations
            )

            # Compute F1.
            assert len(prediction_dict) == len(
                metadata
            )  # Make sure length of predictions is right.
            relation_metrics = self._relation_metrics[self._active_namespace]
            relation_metrics(prediction_dict, metadata)

            output_dict["loss"] = cross_entropy
        return output_dict

    def _prune_spans(
        self, spans, span_mask, span_embeddings, sentence_lengths, name: str
    ) -> PruneOutput:
        if not self.use_pruning:
            bs, num_spans, dim = span_embeddings.shape
            device = span_embeddings.device
            return PruneOutput(
                spans=spans,
                span_mask=span_mask.unsqueeze(dim=-1),
                span_embeddings=span_embeddings,
                num_spans_to_keep=torch.full(
                    (bs,), fill_value=num_spans, device=device, dtype=torch.long
                ),
                span_indices=torch.arange(num_spans, device=device, dtype=torch.long)
                .view(1, num_spans)
                .expand(bs, -1),
                span_mention_scores=torch.zeros(bs, num_spans, 1, device=device),
            )

        pruner = dict(opinion=self._pruner_o, target=self._pruner_t)[name]
        mention_scores = dict(opinion=self._opinion_scores, target=self._target_scores)[
            name
        ]
        pruner.set_external_score(mention_scores.detach())

        # Prune
        num_spans = spans.size(1)  # Max number of spans for the minibatch.

        # Keep different number of spans for each minibatch entry.
        num_spans_to_keep = torch.ceil(
            sentence_lengths.float() * self._spans_per_word
        ).long()

        outputs = pruner(span_embeddings, span_mask, num_spans_to_keep)
        (
            top_span_embeddings,
            top_span_mask,
            top_span_indices,
            top_span_mention_scores,
            num_spans_kept,
        ) = outputs

        top_span_mask = top_span_mask.unsqueeze(-1)

        flat_top_span_indices = util.flatten_and_batch_shift_indices(
            top_span_indices, num_spans
        )
        top_spans = util.batched_index_select(
            spans, top_span_indices, flat_top_span_indices
        )

        return PruneOutput(
            span_embeddings=top_span_embeddings,
            span_mention_scores=top_span_mention_scores,
            num_spans_to_keep=num_spans_to_keep,
            span_mask=top_span_mask,
            span_indices=top_span_indices,
            spans=top_spans,
        )

    def predict(
        self, spans_a, spans_b, relation_scores, num_keep_a, num_keep_b, metadata
    ):
        preds_dict = []
        predictions = []
        for i in range(relation_scores.shape[0]):
            # Each entry/sentence in batch
            pred_dict_sent, predictions_sent = self._predict_sentence(
                spans_a[i],
                spans_b[i],
                relation_scores[i],
                num_keep_a[i],
                num_keep_b[i],
                metadata[i],
            )
            preds_dict.append(pred_dict_sent)
            predictions.append(predictions_sent)

        return preds_dict, predictions

    def _predict_sentence(
        self,
        top_spans_a,
        top_spans_b,
        relation_scores,
        num_keep_a,
        num_keep_b,
        sentence,
    ):
        num_a = num_keep_a.item()  # noqa
        num_b = num_keep_b.item()  # noqa
        spans_a = [tuple(x) for x in top_spans_a.tolist()]
        spans_b = [tuple(x) for x in top_spans_b.tolist()]

        # Iterate over all span pairs and labels. Record the span if the label isn't null.
        predicted_scores_raw, predicted_labels = relation_scores.max(dim=-1)
        softmax_scores = F.softmax(relation_scores, dim=-1)
        predicted_scores_softmax, _ = softmax_scores.max(dim=-1)
        predicted_labels -= 1  # Subtract 1 so that null labels get -1.

        ix = predicted_labels >= 0  # TODO: Figure out their keep_mask (relation.py:202)

        res_dict = {}
        predictions = []

        for i, j in ix.nonzero(as_tuple=False):
            span_1 = spans_a[i]
            span_2 = spans_b[j]
            label = predicted_labels[i, j].item()
            raw_score = predicted_scores_raw[i, j].item()
            softmax_score = predicted_scores_softmax[i, j].item()

            label_name = self.vocab.get_token_from_index(
                label, namespace=self._active_namespace
            )
            res_dict[(span_1, span_2)] = label_name
            list_entry = (
                span_1[0],
                span_1[1],
                span_2[0],
                span_2[1],
                label_name,
                raw_score,
                softmax_score,
            )
            predictions.append(
                document.PredictedRelation(list_entry, sentence, sentence_offsets=True)
            )

        return res_dict, predictions

    # TODO: This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._relation_metrics.items():
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
            res_avg[f"MEAN__relation_{name}"] = (
                sum(values) / len(values) if values else 0
            )
            res.update(res_avg)

        return res

    def _make_pair_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape
        features = [a, b]
        if self.use_distance_embeds:
            features.append(self.d_embedder(self._spans_a, self._spans_b))

        x = torch.cat(features, dim=-1)
        return x

    def _compute_span_pair_embeddings(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        c = self._make_pair_features(a, b)
        return c

    def _compute_relation_scores(self, pruned_a: PruneOutput, pruned_b: PruneOutput):
        a_orig, b_orig = pruned_a.span_embeddings, pruned_b.span_embeddings
        bs, num_a, size = a_orig.shape
        bs, num_b, size = b_orig.shape
        chunk_size = max(1000 // num_a, 1)
        # logging.info(dict(a=num_a, b=num_b, chunk_size=chunk_size))
        pool = []

        for i in range(0, num_a, chunk_size):
            a = a_orig[:, i : i + chunk_size, :]
            num_chunk = a.shape[1]
            a = a.view(bs, num_chunk, 1, size).expand(-1, -1, num_b, -1)
            b = b_orig.view(bs, 1, num_b, size).expand(-1, num_chunk, -1, -1)
            assert a.shape == b.shape
            self._spans_a = pruned_a.spans[:, i : i + chunk_size, :]
            self._spans_b = pruned_b.spans

            embeds = self._compute_span_pair_embeddings(a, b)
            self._relation_embeds = embeds

            relation_feedforward = self._relation_feedforwards[self._active_namespace]
            relation_scorer = self._relation_scorers[self._active_namespace]
            embeds = torch.flatten(embeds, end_dim=-2)
            projected = relation_feedforward(embeds)
            scores = relation_scorer(projected)

            scores = scores.view(bs, num_chunk, num_b, -1)
            pool.append(scores)
        scores = torch.cat(pool, dim=1)
        return scores

    @staticmethod
    def _get_pruned_gold_relations(
        relation_labels: torch.Tensor, pruned_a: PruneOutput, pruned_b: PruneOutput
    ) -> torch.Tensor:
        """
        Loop over each slice and get the labels for the spans from that slice.
        All labels are offset by 1 so that the "null" label gets class zero. This is the desired
        behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
        the softmax loss ignores.
        """
        # TODO: Test and possibly optimize.
        relations = []

        indices_a, masks_a = pruned_a.span_indices, pruned_a.span_mask.bool()
        indices_b, masks_b = pruned_b.span_indices, pruned_b.span_mask.bool()

        for i in range(relation_labels.shape[0]):
            # Each entry in batch
            entry = relation_labels[i]
            entry = entry[indices_a[i], :][:, indices_b[i]]
            mask_entry = masks_a[i] & masks_b[i].transpose(0, 1)
            assert entry.shape == mask_entry.shape
            entry[mask_entry] += 1
            entry[~mask_entry] = -1
            relations.append(entry)

        # return torch.cat(relations, dim=0)
        # This should be a mistake, don't want to concat items within a batch together
        # Likely undiscovered because current bs=1 and _get_loss flattens everything
        return torch.stack(relations, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        # Need to add one for the null class.
        n_labels = self._n_labels[self._active_namespace] + 1
        scores_flat = relation_scores.view(-1, n_labels)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = relation_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._loss(scores_flat, labels_flat)
        return loss
