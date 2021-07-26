"""
Short utility functions.
"""

from typing import Optional, Callable

import torch
import torch.nn.functional as F
from allennlp.modules import FeedForward
from allennlp.modules.span_extractors import EndpointSpanExtractor, SpanExtractor
from allennlp.nn.util import batched_span_select
from overrides import overrides
from torch import Tensor


def cumsum_shifted(xs):
    """
    Assumes `xs` is a 1-d array.
    The usual cumsum has elements [x[1], x[1] + x[2], ...]. This one has elements
    [0, x[1], x[1] + x[2], ...]. Useful for calculating sentence offsets.
    """
    cs = xs.cumsum(dim=0)
    shift = torch.zeros(1, dtype=torch.long, device=cs.device)  # Put on correct device.
    return torch.cat([shift, cs[:-1]], dim=0)


def batch_identity(batch_size, matrix_size, *args, **kwargs):
    """
    Tile the identity matrix along axis 0, `batch_size` times.
    """
    ident = torch.eye(matrix_size, *args, **kwargs).unsqueeze(0)
    res = ident.repeat(batch_size, 1, 1)
    return res


def fields_to_batches(d, keys_to_ignore=[]):
    """
    The input is a dict whose items are batched tensors. The output is a list of dictionaries - one
    per entry in the batch - with the slices of the tensors for that entry. Here's an example.
    Input:
    d = {"a": [[1, 2], [3,4]], "b": [1, 2]}
    Output:
    res = [{"a": [1, 2], "b": 1}, {"a": [3, 4], "b": 2}].
    """
    keys = [key for key in d.keys() if key not in keys_to_ignore]

    # Make sure all input dicts have same length. If they don't, there's a problem.
    lengths = {k: len(d[k]) for k in keys}
    if len(set(lengths.values())) != 1:
        msg = f"fields have different lengths: {lengths}."
        # If there's a doc key, add it to specify where the error is.
        if "doc_key" in d:
            msg = f"For document {d['doc_key']}, " + msg
        raise ValueError(msg)

    length = list(lengths.values())[0]
    res = [{k: d[k][i] for k in keys} for i in range(length)]
    return res


def batches_to_fields(batches):
    """
    The inverse of `fields_to_batches`.
    """
    # Make sure all the keys match.
    first_keys = batches[0].keys()
    for entry in batches[1:]:
        if set(entry.keys()) != set(first_keys):
            raise ValueError("Keys to not match on all entries.")

    res = {k: [] for k in first_keys}
    for batch in batches:
        for k, v in batch.items():
            res[k].append(v)

    return res



class FocalLoss(torch.nn.Module):
    # Reference: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        gamma: float = 0.,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.gamma = gamma
        self.reduction = reduction
        self.nll_loss = torch.nn.NLLLoss(
            weight=weight, reduction="none", ignore_index=ignore_index)

    def forward(self, x, y):
        assert x.ndim == 2
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BiAffine(torch.nn.Module):
    def __init__(self, input_size: int, project_size: int, output_size: int):
        super().__init__()
        self.project_a = torch.nn.Linear(input_size, project_size)
        self.project_b = torch.nn.Linear(input_size, project_size)
        self.bi_affine = torch.nn.Bilinear(project_size, project_size, output_size)
        self.linear = torch.nn.Linear(project_size * 2, output_size)
        self.act = torch.nn.Tanh()
        self.input_size, self.output_size = input_size, output_size

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        a = self.act(self.project_a(a))
        b = self.act(self.project_b(b))
        c = self.bi_affine(a, b)
        d = self.linear(torch.cat([a, b], dim=-1))
        return c + d


class BiAffineSingleInput(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.net = BiAffine(**kwargs)

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-1]
        a, b = torch.split(x, split_size_or_sections=size // 2, dim=-1)
        return self.net(a, b)


class BiAffineV2(BiAffine):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        a = self.act(self.project_a(a))
        b = self.act(self.project_b(b))
        c = self.bi_affine(a, b)
        return c


class BiAffineSpanExtractor(SpanExtractor):
    def __init__(self, endpoint_extractor: EndpointSpanExtractor, **kwargs):
        super().__init__()
        self.endpoint_extractor = endpoint_extractor
        self.net = BiAffineSingleInput(**kwargs)

    def get_input_dim(self) -> int:
        return self.endpoint_extractor.get_input_dim()

    def get_output_dim(self) -> int:
        return self.net.net.output_size

    @overrides
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.BoolTensor = None,
    ) -> Tensor:
        x = self.endpoint_extractor(sequence_tensor, span_indices, span_indices_mask)
        x = self.net(x)
        return x


class LSTMWithMarkers(SpanExtractor):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.start = torch.nn.Parameter(torch.randn(input_size))
        self.end = torch.nn.Parameter(torch.randn(input_size))
        self.input_size = input_size
        self.hidden_size = hidden_size

    def get_input_dim(self) -> int:
        return self.input_size

    def get_output_dim(self) -> int:
        return self.hidden_size * 2

    @overrides
    def forward(
        self,
        sequence_tensor: torch.FloatTensor,
        span_indices: torch.LongTensor,
        span_indices_mask: torch.BoolTensor = None,
    ) -> Tensor:
        x, mask = batched_span_select(sequence_tensor, span_indices)
        assert mask[:, :, 0].float().sum().item() == torch.numel(mask[:, :, 0])
        bs, num_spans, max_width, size = x.shape
        _mask = mask.view(bs, num_spans, max_width, 1).expand_as(x)
        start = self.start.view(1, 1, 1, size).expand(bs, num_spans, 1, size)
        end = self.end.view(1, 1, 1, size).expand(bs, num_spans, 1, size)
        x = torch.where(_mask, x, end.expand_as(x))
        x = torch.cat([start, x, end], dim=-2)

        num_special = 2  # Start & end markers
        # num_special = 0
        x = x.view(bs * num_spans, max_width + num_special, size)
        # lengths = mask.view(bs * num_spans, max_width).sum(dim=-1) + num_special
        # x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (last_hidden, last_cell) = self.lstm(x)
        x = last_hidden.view(bs, num_spans, self.get_output_dim())
        return x


class LearntWeightCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(num_classes))
        self.kwargs = kwargs

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            input, target, weight=self.w, **self.kwargs)


class SpanLengthCrossEntropy(torch.nn.Module):
    def __init__(self, gamma: float, reduction: str, ignore_index: int):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.loss_fn = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=ignore_index)
        self.lengths: Optional[Tensor] = None

    def make_instance_weights(self) -> Tensor:
        assert self.lengths is not None
        w = self.lengths ** self.gamma
        self.lengths = None
        return w

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, c = input.shape
        w = self.make_instance_weights()
        assert tuple(w.shape) == (n,)
        x = self.loss_fn(input, target)
        x *= w

        if self.reduction == "sum":
            x = x.sum()
        elif self.reduction == "mean":
            x = x.mean()
        else:
            assert self.reduction == "none", f"Unknown {dict(reduction=self.reduction)}"

        return x


class BagPairScorer(torch.nn.Module):
    def __init__(self, make_feedforward: Callable[[int], FeedForward], span_emb_dim: int):
        super().__init__()
        self.feed = make_feedforward(span_emb_dim)
        self.input_dim = span_emb_dim * 2

    def get_output_dim(self) -> int:
        return self.feed.get_output_dim()

    def forward(self, x: Tensor) -> Tensor:
        *_, size = x.shape
        a, b, c, d = torch.split(x, split_size_or_sections=size // 4, dim=-1)
        bags = []
        for pair in [(a, c), (a, d), (b, c), (b, d)]:
            bags.append(self.feed(torch.cat(pair, dim=-1)))
        x = torch.stack(bags, dim=0).mean(dim=0)
        return x


class DualScorer(torch.nn.Module):
    def __init__(self, make_feedforward: Callable[[int], FeedForward], input_size: int, num_classes: int):
        super().__init__()
        self.make_feedforward = make_feedforward
        self.input_size = input_size
        self.detector = self.make_scorer(2)
        self.classifier = self.make_scorer(num_classes)

    def make_scorer(self, num_classes: int):
        feedforward = self.make_feedforward(self.input_size)
        scorer = torch.nn.Linear(feedforward.get_output_dim(), num_classes)
        return torch.nn.Sequential(feedforward, scorer)

    def forward(self, x: Tensor, mention_scores: Tensor) -> Tensor:
        x_detect = self.detector(x)
        x_detect[..., :1] += mention_scores
        scores_detect = x_detect.softmax(dim=-1)
        scores_class = self.classifier(x).softmax(dim=-1)
        scores = torch.cat([scores_detect[..., [0]], scores_class * scores_detect[..., [1]]], dim=-1)
        return scores