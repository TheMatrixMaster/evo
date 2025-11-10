import gc
from multiprocessing import cpu_count, get_context
from typing import Dict, List, Optional

import numpy as np
from abnumber import Chain

from .sequence import backtranslate, remove_spaces, translate_sequence

# V gene sequences for Koenig et al. obtained from DASM
IGHV3_23_04_SEQ = """GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCTTTAGCAGCTATGCCATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCTCAGCTATTAGTGGTAGTGGTGGTAGCACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCAGAGACAATTCCAAGAACACGCTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAAAGA"""
IGKV1_39_01_SEQ = """GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGAGCATTAGCAGCTATTTAAATTGGTATCAGCAGAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATGCTGCATCCAGTTTGCAAAGTGGGGTCCCATCAAGGTTCAGTGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGAGTTACAGTACCCCTCC"""

# V and J Gene sequences for Koenig et al. obtained from Aakarsh
KOENIG_IGH_CON_SEQ = """GAGGTGCAGCTGGTGGAGTCTGGGGGAGGCTTGGTACAGCCTGGGGGGTCCCTGAGACTCTCCTGTGCAGCCTCTGGATTCACCATTAGCGACTATTGGATACACTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGGTCGCAGGTATTACTCCTGCTGGTGGTTACACATACTACGCAGACTCCGTGAAGGGCCGGTTCACCATCTCCGCAGACACTTCCAAGAACACGGCGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCCGTATATTACTGTGCGAGATTCGTGTTCTTCCTGCCCTACGCCATGGACTACTGGGGCCAGGGAACCCTGGTCACCGTCTCCTCA"""
KOENIG_IGK_CON_SEQ = """GACATCCAGATGACCCAGTCTCCATCCTCCCTGTCTGCATCTGTAGGAGACAGAGTCACCATCACTTGCCGGGCAAGTCAGGACGTTAGCACCGCTGTAGCTTGGTATCAGCAGAAACCAGGGAAAGCCCCTAAGCTCCTGATCTATTCTGCATCCTTTTTGTATAGTGGGGTCCCATCAAGGTTCAGTGGCAGTGGATCTGGGACAGATTTCACTCTCACCATCAGCAGTCTGCAACCTGAAGATTTTGCAACTTACTACTGTCAACAGAGTTACACTACCCCTCCCACGTTCGGCCAAGGGACCAAGGTGGAAATCAAACGT"""


def get_cdr(seq: str) -> List[str]:
    cdrs = Chain(remove_spaces([seq])[0], scheme="imgt")
    return [cdrs.cdr1_seq, cdrs.cdr2_seq, cdrs.cdr3_seq]


def get_frs(seq: str) -> List[str]:
    frs = Chain(remove_spaces([seq])[0], scheme="imgt")
    return [frs.fr1_seq, frs.fr2_seq, frs.fr3_seq, frs.fr4_seq]


def create_region_masks(sequence: str, scheme="imgt") -> Dict[str, np.ndarray]:
    """Create boolean masks (numpy arrays) for antibody regions.

    Returns a dict mapping region names to numpy boolean arrays of length equal to
    the sequence (True where the residue belongs to that region).
    """
    seq = remove_spaces([sequence])[0]

    # try:
    chain = Chain(seq, scheme=scheme)
    # except Exception as e:
    #     print(f"Error parsing sequence with Chain: {e}. Attempting multiple_domains.")
    #     chains = Chain.multiple_domains(seq, scheme=scheme)
    #     chain = chains[0]

    seq_len = len(chain)

    # Define all region keys you want in the final output
    region_keys = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4", "CDR_overall", "FR_overall"]

    # Initialize numpy boolean arrays
    masks: Dict[str, np.ndarray] = {key: np.zeros(seq_len, dtype=bool) for key in region_keys}

    # Iterate through the sequence positions one time and fill masks
    for i, (pos_name, position) in enumerate(chain.positions.items()):
        # region_name may be None for some position objects; guard against that
        region_name = pos_name.get_region()

        if region_name and region_name in masks:
            masks[region_name][i] = True

        # Use getattr to be robust if the Chain position object varies
        if pos_name.is_in_cdr():
            masks["CDR_overall"][i] = True
        else:
            masks["FR_overall"][i] = True

    return masks


def _safe_create_region_masks(args):
    """Wrapper for create_region_masks that returns (index, result_or_error)."""
    idx, seq, scheme = args
    try:
        result = create_region_masks(seq, scheme=scheme)
        return idx, result, None
    except Exception as e:
        return idx, None, str(e)


def compute_region_masks_batch(
    sequences: List[str],
    num_workers: Optional[int] = None,
    chunksize: int = 100,
    scheme: str = "imgt",
    show_progress: bool = True,
    raise_on_error: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """Compute region masks for a batch of sequences using multiprocessing.

    Args:
        sequences: List of antibody amino acid sequences (heavy or light chains)
        num_workers: Number of worker processes (defaults to cpu_count())
        chunksize: Number of sequences to process per worker chunk
        scheme: Numbering scheme for abnumber (default: 'imgt')
        show_progress: Show progress bar with tqdm (default: True)
        raise_on_error: Raise exception on first error, otherwise skip failed sequences (default: False)

    Returns:
        List of mask dictionaries, one per input sequence. Each dict maps
        region names to boolean numpy arrays. Failed sequences return None at their index.

    Example:
        >>> heavy_chains = ["EVQLV...", "QVQLQ..."]
        >>> masks = compute_region_masks_batch(heavy_chains, num_workers=8)
        >>> cdr3_mask = masks[0]['CDR3']  # boolean array for first sequence
    """
    if num_workers is None:
        num_workers = cpu_count()

    if len(sequences) == 0:
        return []

    # Single sequence - no need for multiprocessing
    if len(sequences) == 1:
        return [create_region_masks(sequences[0], scheme=scheme)]

    # Prepare work items: (index, sequence, scheme)
    work_items = [(i, seq, scheme) for i, seq in enumerate(sequences)]

    # Initialize results list with None placeholders
    results = [None] * len(sequences)
    error_count = 0

    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        iterator = pool.imap(_safe_create_region_masks, work_items, chunksize=chunksize)

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(sequences), desc="Computing masks")
            except ImportError:
                print(f"Processing {len(sequences)} sequences with {num_workers} workers...")

        for idx, result, error in iterator:
            if error is not None:
                error_count += 1
                if raise_on_error:
                    raise RuntimeError(f"Error processing sequence {idx}: {error}")
                if show_progress and error_count <= 5:  # Show first 5 errors
                    print(f"\nWarning: Failed to process sequence {idx}: {error}")
            else:
                results[idx] = result

    if error_count > 0:
        print(
            f"\nCompleted with {error_count}/{len(sequences)} failed sequences (returned as None)"
        )

    # clean up memory
    del ctx
    gc.collect()

    return results


def backtranslate_with_v_gene(aa_sequence: str, v_gene_seq: str) -> str:
    """Backtranslate protein sequence using V gene sequence where possible."""
    # Truncate v_gene_seq to codon boundary and translate
    v_gene_seq = v_gene_seq[: len(v_gene_seq) - len(v_gene_seq) % 3]
    v_gene_aa = translate_sequence(v_gene_seq)
    consensus_popular_nt = backtranslate(aa_sequence)

    result = ""
    for i, aa in enumerate(aa_sequence):
        if i < len(v_gene_aa) and aa == v_gene_aa[i]:
            result += v_gene_seq[i * 3 : i * 3 + 3]
        else:
            result += consensus_popular_nt[i * 3 : i * 3 + 3]

    assert translate_sequence(result) == aa_sequence
    return result
