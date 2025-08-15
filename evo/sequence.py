from typing import List, Tuple
import re
from itertools import product, zip_longest
import pandas as pd
import numpy as np

_FASTA_VOCAB = "ARNDCQEGHILKMFPSTWYV"
_DNA_VOCAB = "ACGT"


def remove_spaces(seqs: List[str]) -> np.ndarray:
    return np.array([''.join(s.strip().split(' ')) for s in seqs])


def add_spaces(seqs: List[str]) -> np.ndarray:
    return np.array([' '.join(list(s)) for s in seqs])


def single_substitution_names(sequence: str, vocab=_FASTA_VOCAB) -> List[str]:
    """Returns the names of all single mutants of a sequence."""
    mutants = []
    for (i, wt), mut in product(enumerate(sequence), vocab):
        if wt == mut:
            continue
        mutant = f"{wt}{i + 1}{mut}"
        mutants.append(mutant)
    return mutants


def single_deletion_names(sequence: str) -> List[str]:
    """Returns the names of all single deletions of a sequence."""
    mutants = []
    for i in range(len(sequence)):
        mutant = f"{sequence[i]}{i + 1}-"
        mutants.append(mutant)
    return mutants


def single_insertion_names(sequence: str, vocab=_FASTA_VOCAB) -> List[str]:
    """Returns the names of all single insertions of a sequence."""
    mutants = []
    for i in range(len(sequence) + 1):
        for mut in vocab:
            mutant = f"-{i + 1}{mut}"
            mutants.append(mutant)
    return mutants


def split_mutant_name(mutant: str) -> Tuple[str, int, str]:
    """Splits a mutant name into the wildtype, position, and mutant."""
    return mutant[0], int(mutant[1:-1]), mutant[-1]


def sort_mutation_names(mutant: str) -> str:
    """Sorts mutation names in a sequence from greatest to smallest position."""
    delimiters = [",", r"\+", ":"]
    expression = re.compile("|".join(delimiters))
    if mutant.upper() == "WT":
        return mutant
    if expression.search(mutant):
        mutants = expression.split(mutant)
        mutants = sorted(mutants, key=lambda x: int(x[1:-1]), reverse=True)
        return ','.join(mutants)
    return mutant


def make_mutation(sequence: str, mutant: str, start_ind: int = 1) -> str:
    """Makes a mutation on a particular sequence. Multiple mutations may be separated
    by ',', ':', or '+', characters.
    """
    if len(mutant) == 0:
        return sequence
    mutant = sort_mutation_names(mutant)
    delimiters = [",", r"\+", ":"]
    expression = re.compile("|".join(delimiters))
    if mutant.upper() == "WT":
        return sequence
    if expression.search(mutant):
        mutants = expression.split(mutant)
        for mutant in mutants:
            sequence = make_mutation(sequence, mutant)
        return sequence
    else:
        wt, pos, mut = split_mutant_name(mutant)
        pos -= start_ind
        if pos < 0 or pos > len(sequence):
            raise ValueError(f"Position {pos} out of bounds for sequence of length {len(sequence)}.")
        if wt == "-":   # insertion
            return sequence[:pos] + mut + sequence[pos:]
        if mut == "-":  # deletion
            assert sequence[pos] == wt
            return sequence[:pos] + sequence[pos + 1:]
        else:           # substitution
            assert sequence[pos] == wt
            return sequence[:pos] + mut + sequence[pos + 1:]


def create_mutant_df(sequence: str, subs_only=False) -> pd.DataFrame:
    """Create a dataframe with mutant names and sequences"""
    names, types = ["WT"], [None]
    subs = single_substitution_names(sequence)
    names += subs
    types += ["substitution"] * len(subs)
    if not subs_only:
        ins = single_insertion_names(sequence)
        dels = single_deletion_names(sequence)
        names += ins + dels
        types += ["insertion"] * len(ins) + ["deletion"] * len(dels)
    sequences = [sequence] + [make_mutation(sequence, mut) for mut in names[1:]]
    return pd.DataFrame({"mutant": names, "sequence": sequences, "type": types})


def seqdiff(seq1: str, seq2: str) -> str:
    diff = []
    for aa1, aa2 in zip_longest(seq1, seq2, fillvalue="-"):
        if aa1 == aa2:
            diff.append(" ")
        else:
            diff.append("|")
    out = f"{seq1}\n{''.join(diff)}\n{seq2}"
    return out


def to_pivoted_mutant_df(df: pd.DataFrame) -> pd.DataFrame:
    df["wt_aa"] = df["mutant"].str.get(0)
    df["mut_aa"] = df["mutant"].str.get(-1)
    df["Position"] = df["mutant"].str.slice(1, -1).astype(int)
    df = df.drop(columns="mutant").pivot(
        index="mut_aa", columns=["Position", "wt_aa"]
    )
    df = df.loc[list(_FASTA_VOCAB)]
    return df


def pivoted_mutant_df(sequence: str, scores: np.ndarray) -> pd.DataFrame:
    index = pd.Index(list(_FASTA_VOCAB), name="mut_aa")
    columns = pd.MultiIndex.from_arrays(
        [list(range(1, len(sequence) + 1)), list(sequence)], names=["Position", "wt_aa"]
    )
    df = pd.DataFrame(
        data=scores,
        index=index,
        columns=columns,
    )
    return df


# Example usage:
if __name__ == "__main__":
    
    # Write a simple test to make sure that create_mutant_df() works as expected
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    df = create_mutant_df(sequence)
    print(df.head())
