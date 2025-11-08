"""ByteNet architecture for Random Oracle.

Extracted from CloneBO repository.
Original: https://github.com/microsoft/protein-sequence-models
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PositionFeedForward(nn.Module):
    """Position-wise feed-forward layer using 1D convolution."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

    Shape:
        Input: (N, L, in_channels)
        input_mask: (N, L, 1), optional
        Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask[..., None]
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).

    Shape:
        Input: (N, L, d_in)
        input_mask: (N, L, 1), optional
        Output: (N, L, d_out)
    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, activation="relu"):
        super().__init__()

        self.conv = MaskedConv1d(
            d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups
        )

        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU

        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h),
            nn.LayerNorm(d_h),
            act(),
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out),
        ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        return x + self.sequence2(self.conv(self.sequence1(x), input_mask=input_mask))


class ByteNet(nn.Module):
    """Stacked residual blocks from ByteNet paper.

    Shape:
        Input: (N, L,)
        input_mask: (N, L, 1), optional
        Output: (N, L, d)
    """

    def __init__(
        self,
        n_tokens,
        d_embedding,
        d_model,
        n_layers,
        kernel_size,
        r,
        padding_idx=None,
        dropout=0.0,
        slim=True,
        activation="relu",
        down_embed=True,
    ):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :padding_idx: location of padding token in ordered alphabet
        :param slim: if True, use half as many dimensions in the NLP as in the CNN
        :param activation: 'relu' or 'gelu'
        :param down_embed: if True, have lower dimension for initial embedding than in CNN layers
        """
        super().__init__()
        if n_tokens is not None:
            self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(d_model, d_h, d_model, kernel_size, dilation=d, activation=activation)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x):
        e = self.embedder(x)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e
