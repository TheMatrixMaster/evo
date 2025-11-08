"""Random neural net oracle.

Hybrid oracle that combines LLM oracle with a randomly initialized ByteNet
for robust fitness prediction with controllable noise.
"""

import os
import re
from typing import List

import numpy as np
import torch
from abnumber import Chain

from .base import BaseOracle
from .bytenet import ByteNet
from .llm_oracle import LLMOracle

# Constants from clonebo
AHO_HEAVY_LENGTH = 149
AHO_LIGHT_LENGTH = 127


def align_sequence(seq: str) -> str:
    """Align sequence using AHO numbering scheme.

    Parameters
    ----------
    seq : str
        Unaligned antibody sequence

    Returns
    -------
    str
        Aligned sequence with gaps
    """
    chain = Chain(seq.replace("-", ""), scheme="aho", cdr_definition="chothia")
    chain_dict = {int(re.sub("[^0-9]", "", str(pos)[1:])): aa for pos, aa in chain}
    expect_length = AHO_HEAVY_LENGTH if chain.is_heavy_chain() else AHO_LIGHT_LENGTH
    aligned = [chain_dict.get(i, "-") for i in range(1, expect_length + 1)]
    return "".join(aligned)


class RandOracle(BaseOracle):
    """Random neural net oracle with hybrid LLM/ByteNet scoring.

    Combines a fine-tuned LLM oracle with a randomly initialized ByteNet
    using weighted normalization. Alpha controls the mixing ratio.

    Output:
        Normalized weighted score (lower is better)

    Chain type:
        Heavy chain only

    Seed sequences:
        Same as LLM oracle (from clone_train_data.csv, indices 0-1)
    """

    # Same seed sequences as LLM oracle (from clone_train_data.csv)
    # Dictionary mapping seed IDs to sequence and precomputed fitness
    # NOTE: Fitness values computed using pre-aligned sequences (anarci not required)
    SEED_DATA = {
        "seed_0": {
            "sequence": "QITLKESGPTLVKPTQTLTLTCTFSGFSLNTRGVSVAWIRQPPGKALDWLALIYWNDDQHYSPSLNNRVTITKDTSKNQVVLKMTNMDYVDTGTDYCVYKKSECGTTGCCYPFYYYYYMDVWGKGTTVNVSS",
            "fitness": -0.713293,  # Computed via workaround (alpha=0.5, bypassing anarci)
        },
        "seed_1": {
            "sequence": "QITLKESGPTLVKPTQTLTLTCIFSGFSPNTRGVSVAWIRQPPGKALVWLALLYWNDDQHYIPSLNYKVTITKDTSNNHVVLTMTNMDYVDTGTYYCDYSKNECGTTGCCYPFYYYYYMDVWGKGTTVTVSS",
            "fitness": -0.291572,  # Computed via workaround (alpha=0.5, bypassing anarci)
        },
    }
    CHAIN_TYPE = "heavy"

    def __init__(
        self,
        alpha: float = 0.5,
        weights_dir: str = "/scratch/users/stephen.lu/projects/protevo/checkpoints/oracle_weights",
        max_length: int = 512,
        device: str = "cuda",
    ):
        """Initialize random oracle.

        Parameters
        ----------
        alpha : float
            Mixing ratio between random and LLM oracle (0 to 1)
            alpha=0: pure LLM oracle
            alpha=1: pure random ByteNet
            alpha=0.5: balanced mixture
        weights_dir : str
            Directory containing oracle weights
        max_length : int
            Maximum sequence length for ByteNet, defaults to 512
        device : str
            Device to run inference on, defaults to 'cuda'
        """
        super().__init__(device=device)

        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha
        self.weights_dir = weights_dir
        self.max_length = max_length

        # Initialize LLM oracle
        self.llm_oracle = LLMOracle(weights_dir=weights_dir, device=device)

        # Initialize random ByteNet
        rand_bytenet_ckpt_path = os.path.join(weights_dir, "rand_bytenet_model_ckpt.pth")
        self.rand_oracle = ByteNet(
            n_tokens=self.llm_oracle.tokenizer.vocab_size,
            d_embedding=32,
            d_model=64,
            n_layers=3,
            kernel_size=3,
            r=1,
        )

        if os.path.exists(rand_bytenet_ckpt_path):
            self.rand_oracle.load_state_dict(
                torch.load(
                    rand_bytenet_ckpt_path,
                    map_location=self.device,
                    weights_only=True,  # Safe for pure state dicts
                )
            )
        else:
            raise FileNotFoundError(f"ByteNet checkpoint not found at {rand_bytenet_ckpt_path}")

        self.rand_oracle.to(self.device)
        self.rand_oracle.eval()

        # Load normalization statistics
        llm_stats_path = os.path.join(weights_dir, "llm_oracle_loss_stats.pth")
        rand_stats_path = os.path.join(weights_dir, "rand_oracle_loss_stats.pth")

        self.llm_oracle_obj_loss = torch.load(
            llm_stats_path, map_location=self.device, weights_only=True
        )
        self.rand_oracle_obj_loss = torch.load(
            rand_stats_path, map_location=self.device, weights_only=True
        )

        self.llm_oracle_obj_loss_mean = torch.mean(self.llm_oracle_obj_loss)
        self.rand_oracle_obj_loss_mean = torch.mean(self.rand_oracle_obj_loss)

        self.llm_oracle_obj_loss_std = torch.std(self.llm_oracle_obj_loss)
        self.rand_oracle_obj_loss_std = torch.std(self.rand_oracle_obj_loss)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    @property
    def seed_data(self):
        """Return the seed data dictionary with sequences and precomputed fitnesses."""
        return self.SEED_DATA

    @property
    def seed_sequences(self):
        """Return list of seed sequences for backward compatibility."""
        return [data["sequence"] for data in self.SEED_DATA.values()]

    @property
    def seed_fitnesses(self):
        """Return list of precomputed seed fitnesses (None if not yet computed)."""
        return [data["fitness"] for data in self.SEED_DATA.values()]

    @property
    def chain_type(self):
        """Return the antibody chain type (heavy, light, or both)."""
        return self.CHAIN_TYPE

    @property
    def higher_is_better(self):
        """Return True if higher fitness values are better, False otherwise."""
        return False  # Normalized loss: lower is better

    def get_seed_fitness(self, seed_id: str) -> float:
        """Get precomputed fitness for a specific seed without re-running inference.

        Parameters
        ----------
        seed_id : str
            Seed identifier (e.g., 'seed_0', 'seed_1')

        Returns
        -------
        float or None
            Precomputed fitness value, or None if not yet computed

        Raises
        ------
        ValueError
            If seed_id is not valid
        """
        if seed_id not in self.SEED_DATA:
            raise ValueError(
                f"Unknown seed_id: {seed_id}. Available: {list(self.SEED_DATA.keys())}"
            )
        fitness = self.SEED_DATA[seed_id]["fitness"]
        if fitness is None:
            import warnings

            warnings.warn(
                f"Fitness for {seed_id} is None. "
                "This should not happen as all seed fitnesses have been precomputed."
            )
        return fitness

    def _pad_tokens(self, token: torch.Tensor) -> torch.Tensor:
        """Pad token sequence to max_length."""
        num_25_to_add = self.max_length - len(token)
        if num_25_to_add > 0:
            token = torch.cat([token, torch.tensor([25] * num_25_to_add)])
        return token

    def predict(self, sequence: str) -> float:
        """Predict fitness for a single sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence as a string

        Returns
        -------
        float
            Normalized weighted fitness score
        """
        return self.predict_batch([sequence])[0]

    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """Predict fitness for a batch of sequences.

        Parameters
        ----------
        sequences : List[str]
            List of protein sequences as strings

        Returns
        -------
        np.ndarray
            Array of normalized weighted fitness scores, shape (n_sequences,)
        """
        # Get LLM scores if needed
        if self.alpha == 1:
            llm_score = None
        else:
            llm_score = self.llm_oracle.predict_batch(sequences)

        # Align sequences for ByteNet
        aligned_sequences = [align_sequence(seq) for seq in sequences]

        # Tokenize aligned sequences
        tokens = [
            self.llm_oracle.tokenizer(" ".join(seq), return_tensors="pt").input_ids[0][1:-1]
            for seq in aligned_sequences
        ]

        # Pad tokens
        tokens = [self._pad_tokens(token) for token in tokens]

        one_t = torch.ones(1, dtype=torch.long)

        # Add special tokens
        tokens = [
            torch.cat(
                [
                    self.llm_oracle.tokenizer.bos_token_id * one_t,
                    token,
                    self.llm_oracle.tokenizer.seq_sep_token_id * one_t,
                ]
            )
            for token in tokens
        ]
        tokens = [token[: self.max_length] for token in tokens]

        # Compute random oracle scores
        rand_score = []
        for i in range(len(tokens)):
            curr_input_id = tokens[i].clone().detach()
            curr_label = tokens[i].clone().detach()

            with torch.no_grad():
                outputs = self.rand_oracle(curr_input_id.to(self.device).unsqueeze(0))

            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = curr_label[..., 1:].contiguous()

            loss = self.loss_fn(shift_logits.squeeze(), shift_labels.to(self.device))
            rand_score.append(loss.item())

        # Combine scores with normalization
        if self.alpha == 1:
            total_score = torch.tensor(rand_score)
        else:
            # Normalize both scores to z-scores and combine
            rand_norm = (
                torch.tensor(rand_score) - self.rand_oracle_obj_loss_mean
            ) / self.rand_oracle_obj_loss_std
            llm_norm = (
                torch.tensor(llm_score) - self.llm_oracle_obj_loss_mean
            ) / self.llm_oracle_obj_loss_std
            total_score = self.alpha * rand_norm + (1 - self.alpha) * llm_norm

        return total_score.numpy()

    def __repr__(self):
        return (
            f"RandOracle(alpha={self.alpha}, device='{self.device}', max_length={self.max_length})"
        )
