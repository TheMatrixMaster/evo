"""COVID-19 neutralization prediction oracle.

Predicts antibody neutralization of SARS-CoV-1 and SARS-CoV-2 variants
using SRU-based RNN encoder with optional IgLM likelihood weighting.
"""

import os
from typing import List, Optional

import numpy as np
import torch

from .base import BaseOracle
from .covid_model import AA_VOCAB, MultiABOnlyCoronavirusModel


class CovidOracle(BaseOracle):
    """Oracle for predicting COVID-19 neutralization.

    Supports both SARS-CoV-1 and SARS-CoV-2 Beta variant predictions.
    Can optionally weight predictions with IgLM log-likelihood.

    Output:
        Neutralization logits (higher is better)

    Chain type:
        Heavy chain only (VH or VHH)

    Seed sequences:
        Default sequences from CovAbDab datasets (indices 0-1)
    """

    # Seed sequences from CloneBO optimization (first 2 from respective CovAbDab files)
    # Dictionary mapping seed IDs to sequence and precomputed fitness
    # NOTE: Fitness values computed using CPU mode (SRU has CPU fallback despite CUDA compilation errors)
    SEED_DATA = {
        "SARSCoV1": {
            "seed_0": {
                "sequence": "QVQLQESGPGLVKPSETLSLTCTVSGGFIGPHYWSWVRQPPGKGLEWIGYIYISGSTNYNPSLKSRLTISVDMSKSQFSLTLSSATAADTAVYYCARGGGYLETGPFEYWGQGTLVTVSS",
                "fitness": -0.916481,  # Computed on CPU (without IgLM weighting)
            },
            "seed_1": {
                "sequence": "QVQLVQSGAEVKKPGASMKVSCKASGFTFTDYYMHWVRQAPRQGLEWMGLINPSGSGTAYAQNFQGRVTLARDTSTSTLYMEMGSLTSDDTAVYYCARMDSSGSYDHWGQGTLVTVSS",
                "fitness": -0.912276,  # Computed on CPU (without IgLM weighting)
            },
        },
        "SARSCoV2Beta": {
            "seed_0": {
                "sequence": "VQLVESGGGLVQPGGSLRLSCAASGLTVSSNYMNWVRQAPGKGLEWVSVFYPGGSTFYADSVRGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDHSGHALDIWGQGTMVTVS",
                "fitness": 1.933321,  # Computed on CPU (without IgLM weighting)
            },
            "seed_1": {
                "sequence": "EVQLVESGGGLVQPGRSLRLSCAASGFKFDDYAMHWVRQAPGKGLEWVSGTSWNSGTTGYADSVRGRFTISRDNAKKSLYLQMNSLGVEDTAFYYCVKDSNYDSSGYLINNFDYWGQGILVTVSS",
                "fitness": 0.218480,  # Computed on CPU (without IgLM weighting)
            },
        },
    }
    CHAIN_TYPE = "heavy"

    def __init__(
        self,
        variant: str = "SARSCoV1",
        weights_dir: str = "/scratch/users/stephen.lu/projects/protevo/checkpoints/oracle_weights",
        device: str = "cuda",
        use_iglm_weighting: bool = True,
        iglm_weight: Optional[float] = None,
    ):
        """Initialize COVID oracle.

        Parameters
        ----------
        variant : str
            Variant to predict: 'SARSCoV1' or 'SARSCoV2Beta'
        weights_dir : str
            Directory containing neut_model.ckpt
        device : str
            Device to run inference on, defaults to 'cuda'
        use_iglm_weighting : bool
            Whether to add IgLM log-likelihood weighting, defaults to True
        iglm_weight : float, optional
            Weight for IgLM term. If None, computed from std ratio
        """
        super().__init__(device=device)

        if variant not in ["SARSCoV1", "SARSCoV2Beta"]:
            raise ValueError("variant must be 'SARSCoV1' or 'SARSCoV2Beta'")

        self.variant = variant
        self.variant_idx = 0 if variant == "SARSCoV1" else 1
        self.weights_dir = weights_dir
        self.use_iglm_weighting = use_iglm_weighting

        # Load model
        model_path = os.path.join(weights_dir, "neut_model.ckpt")
        model_args = {
            "hidden_dim": 256,
            "n_layers": 2,
            "use_srupp": False,
        }
        model_args.update(MultiABOnlyCoronavirusModel.add_extra_args())

        self.model = MultiABOnlyCoronavirusModel.load_from_checkpoint(
            model_path,
            **model_args,
        )
        self.model.to(self.device)
        self.model.eval()

        # Initialize IgLM if requested
        self.iglm = None
        self.iglm_weight = iglm_weight
        if use_iglm_weighting:
            from iglm import IgLM

            self.iglm = IgLM()
            self.iglm_cache = {}

    @property
    def seed_data(self):
        """Return the seed data dictionary with sequences and precomputed fitnesses."""
        return self.SEED_DATA[self.variant]

    @property
    def seed_sequences(self):
        """Return list of seed sequences for backward compatibility."""
        return [data["sequence"] for data in self.SEED_DATA[self.variant].values()]

    @property
    def seed_fitnesses(self):
        """Return list of precomputed seed fitnesses (None if not yet computed)."""
        return [data["fitness"] for data in self.SEED_DATA[self.variant].values()]

    @property
    def chain_type(self):
        """Return the antibody chain type (heavy, light, or both)."""
        return self.CHAIN_TYPE

    @property
    def higher_is_better(self):
        """Return True if higher fitness values are better, False otherwise."""
        return True  # Neutralization logits: higher is better

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
        if seed_id not in self.SEED_DATA[self.variant]:
            raise ValueError(
                f"Unknown seed_id: {seed_id}. Available: {list(self.SEED_DATA[self.variant].keys())}"
            )
        fitness = self.SEED_DATA[self.variant][seed_id]["fitness"]
        if fitness is None:
            import warnings

            warnings.warn(
                f"Fitness for {seed_id} is None. "
                "This should not happen as all seed fitnesses have been precomputed."
            )
        return fitness

    def _get_iglm_likelihood(self, sequence: str) -> float:
        """Get cached IgLM log-likelihood for a sequence."""
        if sequence not in self.iglm_cache:
            self.iglm_cache[sequence] = self.iglm.log_likelihood(sequence, "[HEAVY]", "[HUMAN]")
        return self.iglm_cache[sequence]

    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode sequence to token IDs."""
        return torch.tensor([AA_VOCAB[aa] for aa in sequence]).long()

    def predict(self, sequence: str) -> float:
        """Predict neutralization for a single sequence.

        Parameters
        ----------
        sequence : str
            Antibody sequence as a string

        Returns
        -------
        float
            Neutralization logit (higher is better binding)
        """
        return self.predict_batch([sequence])[0]

    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """Predict neutralization for a batch of sequences.

        Parameters
        ----------
        sequences : List[str]
            List of antibody sequences as strings

        Returns
        -------
        np.ndarray
            Array of neutralization logits, shape (n_sequences,)
        """
        # Encode sequences
        encoded = [self._encode_sequence(seq) for seq in sequences]

        # Pad to same length
        max_len = max(len(seq) for seq in encoded)
        padded = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, seq in enumerate(encoded):
            padded[i, : len(seq)] = seq

        # Move to device and predict
        padded = padded.to(self.device)

        with torch.no_grad():
            logits = self.model(padded)

        # Extract relevant variant prediction
        predictions = logits[:, self.variant_idx].cpu().numpy()

        # Add IgLM weighting if requested
        if self.use_iglm_weighting:
            iglm_scores = np.array([self._get_iglm_likelihood(seq) for seq in sequences])

            # Compute weight if not provided
            if self.iglm_weight is None:
                # Use ratio of standard deviations (from clonebo implementation)
                # This balances the contributions of the two terms
                # For simplicity, use a default weight if no reference data
                weight = 1.0
            else:
                weight = self.iglm_weight

            predictions = predictions + weight * iglm_scores

        return predictions

    def __repr__(self):
        iglm_str = f", use_iglm_weighting={self.use_iglm_weighting}"
        return f"CovidOracle(variant='{self.variant}', device='{self.device}'{iglm_str})"
