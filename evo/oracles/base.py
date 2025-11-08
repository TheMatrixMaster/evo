"""Base oracle class for protein fitness prediction.

Defines the common interface for all protein fitness oracles.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseOracle(ABC):
    """Abstract base class for protein fitness oracles.

    All oracle implementations should inherit from this class and implement
    the predict() and predict_batch() methods.

    The oracle should be callable, allowing predictions via:
        fitness = oracle(sequence)
        fitnesses = oracle(sequences)
    """

    def __init__(self, device: str = "cuda"):
        """Initialize the oracle.

        Parameters
        ----------
        device : str
            Device to run inference on ('cuda' or 'cpu'), defaults to 'cuda'
        """
        self.device = device

    @abstractmethod
    def predict(self, sequence: str) -> float:
        """Predict fitness for a single protein sequence.

        Parameters
        ----------
        sequence : str
            Protein sequence as a string

        Returns
        -------
        float
            Predicted fitness score
        """

    @abstractmethod
    def predict_batch(self, sequences: List[str]) -> np.ndarray:
        """Predict fitness for a batch of protein sequences.

        Parameters
        ----------
        sequences : List[str]
            List of protein sequences as strings

        Returns
        -------
        np.ndarray
            Array of predicted fitness scores, shape (n_sequences,)
        """

    def __call__(self, sequences: Union[str, List[str]]) -> Union[float, np.ndarray]:
        """Make the oracle callable for convenient inference.

        Parameters
        ----------
        sequences : str or List[str]
            Single sequence or list of sequences

        Returns
        -------
        float or np.ndarray
            Fitness prediction(s)
        """
        if isinstance(sequences, str):
            return self.predict(sequences)
        else:
            return self.predict_batch(sequences)

    def __repr__(self):
        return f"{self.__class__.__name__}(device='{self.device}')"
