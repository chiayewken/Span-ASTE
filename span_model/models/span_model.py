import logging
from typing import Dict, List, Optional, Union
import copy

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.common.params import Params
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, TimeDistributed
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor, SpanExtractor
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator

from span_model.models.ner import NERTagger
from span_model.models.relation_proper import ProperRelationExtractor
from span_model.models.shared import BiAffineSpanExtractor

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# New
from torch import Tensor


class MaxPoolSpanExtractor(SpanExtractor):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return self._input_dim

    @staticmethod
    def extract_pooled(x, mask) -> Tensor:
        return util.masked_max(x, mask, dim=-2)

    @overrides
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.BoolTensor = None,
    ) -> Tensor:
        span_embeddings, span_mask = util.batched_span_select(sequence_tensor, span_indices)
        bs, num_spans, span_width, size = span_embeddings.shape
        span_mask = span_mask.view(bs, num_spans, span_width, 1)
        x = self.extract_pooled(span_embeddings, span_mask)

        if span_indices_mask is not None:
            # Above we were masking the widths of spans with respect to the max
            # span width in the batch. Here we are masking the spans which were
            # originally passed in as padding.
            x *= span_indices_mask.view(bs, num_spans, 1)

        assert tuple(x.shape) == (bs, num_spans, size)
        return x


class MeanPoolSpanExtractor(MaxPoolSpanExtractor):
    @staticmethod
    def extract_pooled(x, mask) -> Tensor:
        return util.masked_mean(x, mask, dim=-2)


class TextEmbedderWithBiLSTM(TextFieldEmbedder):
    def __init__(self, embedder: TextFieldEmbedder, hidden_size: int):
        super().__init__()
        self.embedder = embedder
        self.lstm = torch.nn.LSTM(
            input_size=self.embedder.get_output_dim(),
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=1,  # Increasing num_layers can help but we want fair comparison
        )
        self.dropout = torch.nn.Dropout(p=0.5)
        self.output_size = hidden_size * 2

    def get_output_dim(self) -> int:
        return self.output_size

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = self.embedder(*args, **kwargs)
        x = x.squeeze(dim=0)  # For some reason x.shape is (1, 1, seq_len, size)
        x = self.dropout(x)  # Seems to work best if dropout both before and after lstm
        x, state = self.lstm(x)
        x = self.dropout(x)
        x = x.unsqueeze(dim=0)
        return x


@Model.register("span_model")
class SpanModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: TextFieldEmbedder,
        modules,  # TODO: Add type.
        feature_size: int,
        max_span_width: int,
        target_task: str,
        feedforward_params: Dict[str, Union[int, float]],
        loss_weights: Dict[str, float],
        initializer: InitializerApplicator = InitializerApplicator(),
        module_initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
        # New
        use_ner_embeds: bool = None,
        span_extractor_type: str = None,
        use_double_mix_embedder: bool = None,
        relation_head_type: str = "base",
        use_span_width_embeds: bool = None,
        use_bilstm_after_embedder: bool = False,
    ) -> None:
        super(SpanModel, self).__init__(vocab, regularizer)

        # New
        info = dict(
            use_ner_embeds=use_ner_embeds,
            span_extractor_type=span_extractor_type,
            use_double_mix_embedder=use_double_mix_embedder,
            relation_head_type=relation_head_type,
            use_span_width_embeds=use_span_width_embeds,
        )
        for k, v in info.items():
            print(dict(locals=(k, v)))
            assert v is not None, k
        self.use_double_mix_embedder = use_double_mix_embedder
        self.relation_head_type = relation_head_type
        if use_bilstm_after_embedder:
            embedder = TextEmbedderWithBiLSTM(embedder, hidden_size=300)

        ####################

        assert span_extractor_type in {"endpoint", "attn", "max_pool", "mean_pool", "bi_affine"}
        # Create span extractor.
        if use_span_width_embeds:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                embedder.get_output_dim(),
                combination="x,y",
                num_width_embeddings=max_span_width,
                span_width_embedding_dim=feature_size,
                bucket_widths=False,
            )
        # New
        else:
            self._endpoint_span_extractor = EndpointSpanExtractor(
                embedder.get_output_dim(),
                combination="x,y",
            )
        if span_extractor_type == "attn":
            self._endpoint_span_extractor = SelfAttentiveSpanExtractor(
                embedder.get_output_dim()
            )
        if span_extractor_type == "max_pool":
            self._endpoint_span_extractor = MaxPoolSpanExtractor(
                embedder.get_output_dim()
            )
        if span_extractor_type == "mean_pool":
            self._endpoint_span_extractor = MeanPoolSpanExtractor(
                embedder.get_output_dim()
            )
        if span_extractor_type == "bi_affine":
            token_emb_dim = embedder.get_output_dim()
            assert self._endpoint_span_extractor.get_output_dim() == token_emb_dim * 2
            self._endpoint_span_extractor = BiAffineSpanExtractor(
                endpoint_extractor=self._endpoint_span_extractor,
                input_size=token_emb_dim,
                project_size=200,
                output_size=200,
            )
        self._visualize_outputs = []

        ####################

        # Set parameters.
        self._embedder = embedder
        self._loss_weights = loss_weights
        self._max_span_width = max_span_width
        self._display_metrics = self._get_display_metrics(target_task)
        token_emb_dim = self._embedder.get_output_dim()
        span_emb_dim = self._endpoint_span_extractor.get_output_dim()

        # New
        self._feature_size = feature_size
        ####################

        # Create submodules.

        modules = Params(modules)

        # Helper function to create feedforward networks.
        def make_feedforward(input_dim):
            return FeedForward(
                input_dim=input_dim,
                num_layers=feedforward_params["num_layers"],
                hidden_dims=feedforward_params["hidden_dims"],
                activations=torch.nn.ReLU(),
                dropout=feedforward_params["dropout"],
            )

        # Submodules

        self._ner = NERTagger.from_params(
            vocab=vocab,
            make_feedforward=make_feedforward,
            span_emb_dim=span_emb_dim,
            feature_size=feature_size,
            params=modules.pop("ner"),
        )

        # New
        self.use_ner_embeds = use_ner_embeds
        if self.use_ner_embeds:
            num_ner_labels = sorted(self._ner._n_labels.values())[0]
            self._ner_embedder = torch.nn.Linear(num_ner_labels, feature_size)
            span_emb_dim += feature_size

        params = dict(
            vocab=vocab,
            make_feedforward=make_feedforward,
            span_emb_dim=span_emb_dim,
            feature_size=feature_size,
            params=modules.pop("relation"),
        )
        if self.relation_head_type == "proper":
            self._relation = ProperRelationExtractor.from_params(**params)
        else:
            raise ValueError(f"Unknown: {dict(relation_head_type=relation_head_type)}")

        ####################

        # Initialize text embedder and all submodules
        for module in [self._ner, self._relation]:
            module_initializer(module)

        initializer(self)

    @staticmethod
    def _get_display_metrics(target_task):
        """
        The `target` is the name of the task used to make early stopping decisions. Show metrics
        related to this task.
        """
        lookup = {
            "ner": [
                f"MEAN__{name}" for name in ["ner_precision", "ner_recall", "ner_f1"]
            ],
            "relation": [
                f"MEAN__{name}"
                for name in ["relation_precision", "relation_recall", "relation_f1"]
            ],
        }
        if target_task not in lookup:
            raise ValueError(
                f"Invalied value {target_task} has been given as the target task."
            )
        return lookup[target_task]

    @staticmethod
    def _debatch(x):
        # TODO: Get rid of this when I find a better way to do it.
        return x if x is None else x.squeeze(0)

    def text_to_span_embeds(self, text_embeddings: torch.Tensor, spans):
        # Shape: (batch_size, num_spans, 2 * encoding_dim + feature_size)
        span_embeddings = self._endpoint_span_extractor(text_embeddings, spans)
        return span_embeddings

    @overrides
    def forward(
        self,
        text,
        spans,
        metadata,
        ner_labels=None,
        relation_labels=None,
        dep_graph_labels=None,  # New
        tag_labels=None,  # New
        grid_labels=None,  # New
    ):
        """
        TODO: change this.
        """
        # In AllenNLP, AdjacencyFields are passed in as floats. This fixes it.
        if relation_labels is not None:
            relation_labels = relation_labels.long()

        # TODO: Multi-document minibatching isn't supported yet. For now, get rid of the
        # extra dimension in the input tensors. Will return to this once the model runs.
        if len(metadata) > 1:
            raise NotImplementedError("Multi-document minibatching not yet supported.")

        metadata = metadata[0]
        spans = self._debatch(spans)  # (n_sents, max_n_spans, 2)
        ner_labels = self._debatch(ner_labels)  # (n_sents, max_n_spans)
        relation_labels = self._debatch(
            relation_labels
        )  # (n_sents, max_n_spans, max_n_spans)

        # Encode using BERT, then debatch.
        # Since the data are batched, we use `num_wrapping_dims=1` to unwrap the document dimension.
        # (1, n_sents, max_sententence_length, embedding_dim)

        # TODO: Deal with the case where the input is longer than 512.
        text_embeddings = self._embedder(text, num_wrapping_dims=1)
        # (n_sents, max_n_wordpieces, embedding_dim)
        text_embeddings = self._debatch(text_embeddings)

        # (n_sents, max_sentence_length)
        text_mask = self._debatch(
            util.get_text_field_mask(text, num_wrapping_dims=1).float()
        )
        sentence_lengths = text_mask.sum(dim=1).long()  # (n_sents)

        span_mask = (spans[:, :, 0] >= 0).float()  # (n_sents, max_n_spans)
        # SpanFields return -1 when they are used as padding. As we do some comparisons based on
        # span widths when we attend over the span representations that we generate from these
        # indices, we need them to be <= 0. This is only relevant in edge cases where the number of
        # spans we consider after the pruning stage is >= the total number of spans, because in this
        # case, it is possible we might consider a masked span.
        spans = F.relu(spans.float()).long()  # (n_sents, max_n_spans, 2)

        # New
        text_embeds_b = text_embeddings
        if self.use_double_mix_embedder:
            # DoubleMixPTMEmbedder has to output single concatenated tensor so we need to split
            embed_dim = self._embedder.get_output_dim()
            assert text_embeddings.shape[-1] == embed_dim * 2
            text_embeddings, text_embeds_b = text_embeddings[..., :embed_dim], text_embeddings[..., embed_dim:]

        kwargs = dict(spans=spans)
        span_embeddings = self.text_to_span_embeds(text_embeddings, **kwargs)
        span_embeds_b = self.text_to_span_embeds(text_embeds_b, **kwargs)

        # Make calls out to the modules to get results.
        output_ner = {"loss": 0}
        output_relation = {"loss": 0}

        # Make predictions and compute losses for each module
        if self._loss_weights["ner"] > 0:
            output_ner = self._ner(
                spans,
                span_mask,
                span_embeddings,
                sentence_lengths,
                ner_labels,
                metadata,
            )
            ner_scores = output_ner.pop("ner_scores")
        # New
        if self._loss_weights["relation"] > 0:
            if getattr(self._relation, "use_ner_scores_for_prune", False):
                self._relation._ner_scores = ner_scores
            self._relation._opinion_scores = output_ner["opinion_scores"]
            self._relation._target_scores = output_ner["target_scores"]
            self._relation._text_mask = text_mask
            self._relation._text_embeds = text_embeddings
            if getattr(self._relation, "use_span_loss_for_pruners", False):
                self._relation._ner_labels = ner_labels
            output_relation = self._relation(
                spans,
                span_mask,
                # span_embeddings,
                span_embeds_b,
                sentence_lengths,
                relation_labels,
                metadata,
            )

        # Use `get` since there are some cases where the output dict won't have a loss - for
        # instance, when doing prediction.
        loss = (
            + self._loss_weights["ner"] * output_ner.get("loss", 0)
            + self._loss_weights["relation"] * output_relation.get("loss", 0)
        )

        # Multiply the loss by the weight multiplier for this document.
        weight = metadata.weight if metadata.weight is not None else 1.0
        loss *= torch.tensor(weight)

        output_dict = dict(
            relation=output_relation,
            ner=output_ner,
        )
        output_dict["loss"] = loss

        output_dict["metadata"] = metadata

        return output_dict

    def update_span_embeddings(
        self,
        span_embeddings,
        span_mask,
        top_span_embeddings,
        top_span_mask,
        top_span_indices,
    ):
        # TODO(Ulme) Speed this up by tensorizing

        new_span_embeddings = span_embeddings.clone()
        for sample_nr in range(len(top_span_mask)):
            for top_span_nr, span_nr in enumerate(top_span_indices[sample_nr]):
                if (
                    top_span_mask[sample_nr, top_span_nr] == 0
                    or span_mask[sample_nr, span_nr] == 0
                ):
                    break
                new_span_embeddings[sample_nr, span_nr] = top_span_embeddings[
                    sample_nr, top_span_nr
                ]
        return new_span_embeddings

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """
        Converts the list of spans and predicted antecedent indices into clusters
        of spans for each element in the batch.

        Parameters
        ----------
        output_dict : ``Dict[str, torch.Tensor]``, required.
            The result of calling :func:`forward` on an instance or batch of instances.
        """

        doc = copy.deepcopy(output_dict["metadata"])

        if self._loss_weights["ner"] > 0:
            for predictions, sentence in zip(output_dict["ner"]["predictions"], doc):
                sentence.predicted_ner = predictions

        if self._loss_weights["relation"] > 0:
            for predictions, sentence in zip(
                output_dict["relation"]["predictions"], doc
            ):
                sentence.predicted_relations = predictions

        return doc

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner = self._ner.get_metrics(reset=reset)
        metrics_relation = self._relation.get_metrics(reset=reset)

        # Make sure that there aren't any conflicting names.
        metric_names = (
            list(metrics_ner.keys())
            + list(metrics_relation.keys())
        )
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(
            list(metrics_ner.items())
            + list(metrics_relation.items())
        )

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
