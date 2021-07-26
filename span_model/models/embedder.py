from typing import Optional, Tuple

from overrides import overrides
import torch

from allennlp.modules.token_embedders import PretrainedTransformerEmbedder, TokenEmbedder
from allennlp.nn import util
from allennlp.modules.scalar_mix import ScalarMix


@TokenEmbedder.register("double_mix_ptm")
class DoubleMixPTMEmbedder(TokenEmbedder):
    # Refer: PretrainedTransformerMismatchedEmbedder

    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to pool the resulting vectors to get word-level representations.

    Registered as a `TokenEmbedder` with name "pretrained_transformer_mismatchd".

    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    """

    def __init__(
            self,
            model_name: str,
            max_length: int = None,
            train_parameters: bool = True,
            last_layer_only: bool = True,
            gradient_checkpointing: Optional[bool] = None,
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedTransformerEmbedder(
            model_name,
            max_length=max_length,
            train_parameters=train_parameters,
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._matched_embedder.config.output_hidden_states = True
        num_layers = self._matched_embedder.config.num_hidden_layers
        mix_init = [float(i) for i in range(num_layers)]  # Try to give useful prior, after softmax will be [..., 0.08, 0.23, 0.63]
        self._mixer_a = ScalarMix(num_layers, initial_scalar_parameters=mix_init)
        self._mixer_b = ScalarMix(num_layers, initial_scalar_parameters=mix_init)
        self._matched_embedder.transformer_model.forward = self.make_fn_transformer(
            self._matched_embedder.transformer_model.forward
        )
        # This method doesn't work, gradient doesn't propagate properly
        # self.embeds_b = None  # Bonus output because TokenEmbedder should produce single Tensor output

    @classmethod
    def make_fn_transformer(cls, fn):
        def new_fn(*args, **kwargs):
            transformer_output: tuple = fn(*args, **kwargs)
            # As far as I can tell, the hidden states will always be the last element
            # in the output tuple as long as the model is not also configured to return
            # attention scores.
            # See, for example, the return value description for BERT:
            # https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel.forward
            # These hidden states will also include the embedding layer, which we don't
            # include in the scalar mix. Hence the `[1:]` slicing.
            hidden_states = transformer_output[-1][1:]
            # By default, PTM will return transformer_output[0] so we force the one we want in front
            return (hidden_states,) + transformer_output
        return new_fn

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()

    @staticmethod
    def run_match(embeddings, offsets):
        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings

    @overrides
    def forward(
            self,
            token_ids: torch.LongTensor,
            mask: torch.BoolTensor,
            offsets: torch.LongTensor,
            wordpiece_mask: torch.BoolTensor,
            type_ids: Optional[torch.LongTensor] = None,
            segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.

        # Returns

        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        hidden_states = self._matched_embedder(  # noqa
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )
        assert type(hidden_states) in {tuple, list}
        embeds_a = self.run_match(self._mixer_a(hidden_states), offsets)
        embeds_b = self.run_match(self._mixer_b(hidden_states), offsets)
        x = torch.cat([embeds_a, embeds_b], dim=-1)
        return x
        # self.embeds_b = embeds_b
        # return embeds_a

    def split_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Output has to be single tensor to suit forward signature but we need to split
        output_dim = self.get_output_dim()
        bs, seq_len, hidden_size = x.shape
        assert hidden_size == output_dim * 2
        return x[:, :, :output_dim], x[:, :, output_dim:]
