"""LLM-based oracle for antibody fitness prediction.

Uses a fine-tuned Llama-based model trained on clonal family data
to predict sequence fitness via cross-entropy loss.
"""

import os
from typing import List

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from .base import BaseOracle


class LLMOracle(BaseOracle):
    """LLM-based fitness oracle using cross-entropy as fitness metric.

    This oracle uses a causal language model fine-tuned on antibody clonal
    families. Lower cross-entropy indicates better fitness.

    Output:
        Cross-entropy loss (lower is better)

    Chain type:
        Heavy chain only

    Seed sequences:
        Default sequences from clone_train_data.csv (indices 0-1)
    """

    # Seed sequences from CloneBO optimization (first 2 from clone_train_data.csv)
    # Dictionary mapping seed IDs to sequence and precomputed fitness
    SEED_DATA = {
        "seed_0": {
            "sequence": "QITLKESGPTLVKPTQTLTLTCTFSGFSLNTRGVSVAWIRQPPGKALDWLALIYWNDDQHYSPSLNNRVTITKDTSKNQVVLKMTNMDYVDTGTDYCVYKKSECGTTGCCYPFYYYYYMDVWGKGTTVNVSS",
            "fitness": 0.22852992,  # Cross-entropy loss (computed on A100)
        },
        "seed_1": {
            "sequence": "QITLKESGPTLVKPTQTLTLTCIFSGFSPNTRGVSVAWIRQPPGKALVWLALLYWNDDQHYIPSLNYKVTITKDTSNNHVVLTMTNMDYVDTGTYYCDYSKNECGTTGCFYPFYYYYYMDVWGKGTTVTVSS",
            "fitness": 0.45083448,  # Cross-entropy loss (computed on A100)
        },
    }
    CHAIN_TYPE = "heavy"

    def __init__(
        self,
        weights_dir: str = "/scratch/users/stephen.lu/projects/protevo/checkpoints/oracle_weights",
        max_length: int = 2048,
        device: str = "cuda",
    ):
        """Initialize LLM oracle.

        Parameters
        ----------
        weights_dir : str
            Directory containing vocab.txt and model weights
        max_length : int
            Maximum sequence length, defaults to 2048
        device : str
            Device to run inference on, defaults to 'cuda'
        """
        super().__init__(device=device)
        self.weights_dir = weights_dir
        self.max_length = max_length

        # Initialize tokenizer
        vocab_path = os.path.join(weights_dir, "vocab.txt")
        self.tokenizer = transformers.BertTokenizerFast(
            vocab_file=vocab_path,
            do_lower_case=False,
        )

        # Set special tokens
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.bos_token_id = self.tokenizer.cls_token_id
        self.tokenizer.eos_token = self.tokenizer.sep_token
        self.tokenizer.eos_token_id = self.tokenizer.sep_token_id

        self.tokenizer.hv_token = "[AbHC]"
        self.tokenizer.hv_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.hv_token)
        self.tokenizer.lv_token = "[AbLC]"
        self.tokenizer.lv_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.lv_token)
        self.tokenizer.seq_sep_token = "[ClSep]"
        self.tokenizer.seq_sep_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.seq_sep_token
        )

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained("CloneBO/OracleLM")
        self.model.to(self.device)
        self.model.eval()

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
        """Return list of precomputed seed fitnesses."""
        return [data["fitness"] for data in self.SEED_DATA.values()]

    @property
    def chain_type(self):
        """Return the antibody chain type (heavy, light, or both)."""
        return self.CHAIN_TYPE

    @property
    def higher_is_better(self):
        """Return True if higher fitness values are better, False otherwise."""
        return False  # Cross-entropy loss: lower is better

    def get_seed_fitness(self, seed_id: str) -> float:
        """Get precomputed fitness for a specific seed without re-running inference.

        Parameters
        ----------
        seed_id : str
            Seed identifier (e.g., 'seed_0', 'seed_1')

        Returns
        -------
        float
            Precomputed fitness value
        """
        if seed_id not in self.SEED_DATA:
            raise ValueError(
                f"Unknown seed_id: {seed_id}. Available: {list(self.SEED_DATA.keys())}"
            )
        return self.SEED_DATA[seed_id]["fitness"]

    def predict(self, sequence: str) -> float:
        """Predict fitness for a single sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence as a string

        Returns
        -------
        float
            Cross-entropy loss (lower is better)
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
            Array of cross-entropy losses, shape (n_sequences,)
        """
        assert isinstance(sequences, list), "Input should be a list of string sequences"
        assert all(isinstance(seq, str) for seq in sequences), "All sequences should be strings"

        # Tokenize sequences
        tokens = [
            self.tokenizer(" ".join(seq), return_tensors="pt").input_ids[0][1:-1]
            for seq in sequences
        ]

        one_t = torch.ones(1, dtype=torch.long)

        # Add special tokens
        tokens = [
            torch.cat(
                [
                    self.tokenizer.bos_token_id * one_t,
                    token,
                    self.tokenizer.seq_sep_token_id * one_t,
                ]
            )
            for token in tokens
        ]
        tokens = [token[: self.max_length] for token in tokens]

        # Compute losses
        list_of_loss = []
        for i in range(len(tokens)):
            curr_input_id = tokens[i].clone().detach()
            curr_label = tokens[i].clone().detach()

            with torch.no_grad():
                outputs = self.model(
                    curr_input_id.to(self.device).unsqueeze(0),
                    labels=curr_label.to(self.device),
                )

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = curr_label[..., 1:].contiguous()

            loss = self.loss_fn(shift_logits.squeeze(), shift_labels.to(self.device))
            list_of_loss.append(loss.item())

        return np.array(list_of_loss)

    def __repr__(self):
        return f"LLMOracle(device='{self.device}', max_length={self.max_length})"
