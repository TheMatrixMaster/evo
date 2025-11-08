"""COVID-19 neutralization prediction models.

Extracted from CloneBO repository.
Models for predicting antibody neutralization of SARS-CoV-1 and SARS-CoV-2 variants.
"""

from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

# Amino acid vocabulary
AA_VOCAB = {
    "#": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "-": 22,
    "O": 23,
    "*": 24,
}


class RNNEncoder(nn.Module):
    """Implements a multi-layer RNN.

    This module can be used to create multi-layer RNN models using
    LSTM, GRU, or SRU cell types. Handles padding with PackedSequence.

    Attributes
    ----------
    rnn: nn.Module
        The rnn submodule
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0,
        attn_dropout: float = 0,
        attn_heads: int = 1,
        bidirectional: bool = False,
        layer_norm: bool = False,
        highway_bias: float = -2,
        rescale: bool = True,
        enforce_sorted: bool = False,
        **kwargs,
    ) -> None:
        """Initializes the RNNEncoder object.

        Parameters
        ----------
        input_size : int
            The dimension the input data
        hidden_size : int
            The hidden dimension to encode the data in
        n_layers : int, optional
            The number of rnn layers, defaults to 1
        rnn_type : str, optional
           The type of rnn cell, one of: `lstm`, `gru`, `sru`, `srupp`
           defaults to `lstm`
        dropout : float, optional
            Amount of dropout to use between RNN layers, defaults to 0
        bidirectional : bool, optional
            Set to use a bidrectional encoder, defaults to False
        layer_norm : bool, optional
            [SRU only] whether to use layer norm
        highway_bias : float, optional
            [SRU only] value to use for the highway bias
        rescale : bool, optional
            [SRU only] whether to use rescaling
        enforce_sorted: bool
            Whether rnn should enforce that sequences are ordered by
            length. Requires True for ONNX support. Defaults to False.

        Raises
        ------
        ValueError
            The rnn type should be one of: `lstm`, `gru`, `sru`, `srupp`
        """
        super().__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.enforce_sorted = enforce_sorted
        self.output_size = 2 * hidden_size if bidirectional else hidden_size

        if rnn_type in ["lstm", "gru"]:
            rnn_fn = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_fn(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif rnn_type == "sru":
            from sru import SRU

            try:
                self.rnn = SRU(
                    input_size,
                    hidden_size,
                    num_layers=n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    layer_norm=layer_norm,
                    rescale=rescale,
                    highway_bias=highway_bias,
                    **kwargs,
                )
            except TypeError:
                raise ValueError(f"Unkown kwargs passed to SRU: {kwargs}")
        elif rnn_type == "srupp":
            from sru import SRUpp

            try:
                self.rnn = SRUpp(
                    input_size,
                    hidden_size,
                    hidden_size // 2,
                    num_layers=n_layers,
                    highway_bias=highway_bias,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    num_heads=attn_heads,
                    layer_norm=layer_norm,
                    attn_layer_norm=True,
                    bidirectional=bidirectional,
                    **kwargs,
                )
            except TypeError:
                raise ValueError(f"Unkown kwargs passed to SRU: {kwargs}")
        else:
            raise ValueError(f"Unkown rnn type: {rnn_type}, use of of: gru, sru, lstm, srupp")

    def forward(
        self,
        data: Tensor,
        state: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Performs a forward pass through the network.

        Parameters
        ----------
        data : Tensor
            The input data, as a float tensor of shape [B x S x E]
        state: Tensor
            An optional previous state of shape [L x B x H]
        padding_mask: Tensor, optional
            The padding mask of shape [B x S], dtype should be bool

        Returns
        -------
        Tensor
            The encoded output, as a float tensor of shape [B x S x H]
        Tensor
            The encoded state, as a float tensor of shape [L x B x H]
        """
        data = data.transpose(0, 1)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1)

        if padding_mask is None:
            # Default RNN behavior
            output, state = self.rnn(data, state)
        elif self.rnn_type == "sru":
            # SRU takes a mask instead of PackedSequence objects
            output, state = self.rnn(data, state, mask_pad=(~padding_mask))
        elif self.rnn_type == "srupp":
            # SRUpp takes a mask instead of PackedSequence objects
            output, state, _ = self.rnn(data, state, mask_pad=(~padding_mask))
        else:
            # Deal with variable length sequences
            lengths = padding_mask.long().sum(dim=0)
            # Pass through the RNN
            packed = nn.utils.rnn.pack_padded_sequence(
                data, lengths.cpu(), enforce_sorted=self.enforce_sorted
            )
            output, state = self.rnn(packed, state)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=data.size(0))

        return output.transpose(0, 1).contiguous(), state


class SRUppModel(nn.Module):
    """SRU-based encoder for antibody sequences."""

    def __init__(
        self,
        num_aa: int,
        num_tokens: int,
        n_layers: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0,
        ab_pad_id: int = 0,
        virus_pad_id: int = 0,
        use_srupp: bool = False,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.seq_embedding = nn.Embedding(num_aa, hidden_dim // 4)

        # Antibody encoder
        rnn_type = "srupp" if use_srupp else "sru"
        self.dropout = nn.Dropout(dropout)
        self.rnn_ab = RNNEncoder(
            hidden_dim // 4,
            hidden_dim // 2,
            n_layers=n_layers,
            rnn_type=rnn_type,
            bidirectional=True,
            dropout=dropout,
        )
        self.ab_pad_id = ab_pad_id
        self.virus_pad_id = virus_pad_id
        self.num_tokens = num_tokens

    @property
    def output_dim(self):
        return self.hidden_dim

    def forward(self, ab_seq):
        # Compute padding mask
        padding_ab = ab_seq != self.ab_pad_id

        # Compute token embeddings
        ab_emb = self.dropout(self.seq_embedding(ab_seq))

        # Pass through SRU layers
        ab_encodings, _ = self.rnn_ab(ab_emb, padding_mask=padding_ab)
        return ab_encodings


class MultiABOnlyCoronavirusModel(pl.LightningModule):
    """Multi-target antibody neutralization prediction model.

    Predicts neutralization for SARS-CoV-1 and SARS-CoV-2 simultaneously.
    """

    def __init__(
        self,
        num_aa: int,
        num_tokens: int,
        n_layers: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0,
        lr: float = 1e-3,
        ab_pad_id: int = 0,
        virus_pad_id: int = 0,
        neut_lambda: float = 0.5,
        use_srupp: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.ab_pad_id = ab_pad_id
        self.virus_pad_id = virus_pad_id
        self.neut_lambda = neut_lambda
        self.dropout = nn.Dropout(dropout)
        self.encoder = SRUppModel(
            num_aa=num_aa,
            num_tokens=num_tokens,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            ab_pad_id=ab_pad_id,
            virus_pad_id=virus_pad_id,
            use_srupp=use_srupp,
        )
        encoder_dim = self.encoder.output_dim
        self.fc_neut = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim // 2, 2),
        )

    @classmethod
    def add_extra_args(cls):
        """Return default arguments for model initialization."""
        extra_args = {
            "num_aa": len(AA_VOCAB),
            "num_tokens": 1024,
            "ab_pad_id": AA_VOCAB["#"],
            "virus_pad_id": AA_VOCAB["#"],
        }
        return extra_args

    def average(self, data, padding):
        """Average pool over sequence dimension with padding mask."""
        data = (data * padding.unsqueeze(2)).sum(dim=1)
        padding_sum = padding.sum(dim=1)
        padding_sum[padding_sum == 0] = 1.0
        avg = data / padding_sum.unsqueeze(1)
        return avg

    def forward(self, ab_seq):
        """Forward pass.

        Parameters
        ----------
        ab_seq : Tensor
            Antibody sequence as token IDs, shape [B x L]

        Returns
        -------
        Tensor
            Neutralization logits for [CoV-1, CoV-2], shape [B x 2]
        """
        padding_mask_ab = (ab_seq != self.ab_pad_id).float()
        ab_encodings = self.encoder(ab_seq)
        output_encoding = self.average(ab_encodings, padding_mask_ab)
        output_encoding = self.dropout(output_encoding)
        neut_logits = self.fc_neut(output_encoding).squeeze(1)
        return neut_logits

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
        """Load model from checkpoint with weight transformations."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
            weights_only=False,  # Required for PyTorch 2.6+
        )
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if "transform_module.weight" in k:
                new_k = k.replace("transform_module.weight", "weight")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v

        # Transpose specific tensors for SRU compatibility
        new_state_dict["encoder.rnn_ab.rnn.rnn_lst.0.weight"] = new_state_dict[
            "encoder.rnn_ab.rnn.rnn_lst.0.weight"
        ].t()
        new_state_dict["encoder.rnn_ab.rnn.rnn_lst.1.weight"] = new_state_dict[
            "encoder.rnn_ab.rnn.rnn_lst.1.weight"
        ].t()

        model = cls(*args, **kwargs)
        model.load_state_dict(new_state_dict, strict=False)
        return model
