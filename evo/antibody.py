import gc
from multiprocessing import cpu_count, get_context, Pool
from typing import Dict, List, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
from anarci import anarci
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
    """Create boolean masks for antibody regions matching original sequence length."""
    seq = remove_spaces([sequence])[0]
    chain = Chain(seq, scheme=scheme)
    
    region_keys = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4", "CDR_overall", "FR_overall"]
    masks: Dict[str, np.ndarray] = {key: np.zeros(len(seq), dtype=bool) for key in region_keys}

    # Find where chain sequence starts in original
    offset = seq.find(str(chain.seq))
    if offset == -1:
        raise ValueError("Chain sequence not found in original")

    # Map chain positions to original sequence indices
    for i, (pos_name, _) in enumerate(chain.positions.items()):
        seq_idx = offset + i
        region_name = pos_name.get_region()
        
        if region_name in masks:
            masks[region_name][seq_idx] = True
        
        if pos_name.is_in_cdr():
            masks["CDR_overall"][seq_idx] = True
        else:
            masks["FR_overall"][seq_idx] = True
    
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


def parse_anarci_output(anarci_result, scheme_length=149):
    """
    Parses the raw ANARCI output to construct a fixed-length AHo sequence.
    
    Args:
        anarci_result: The tuple (numbering, alignment_details, hit_tables) for a SINGLE sequence.
        scheme_length: AHo uses 149 positions.
    
    Returns:
        A string of length 149 with the aligned sequence, or None if no domain found.
    """
    numbering, alignment_details, hit_tables = anarci_result

    # numbering is a list of domains found. If empty, no antibody domain was found.
    if not numbering:
        return None

    # We typically take the first domain found (index 0) which is the most significant V-domain
    # domain_numbering is a list of tuples: ((pos_id, insertion_code), residue_char)
    domain_numbering = numbering[0][0][0] 
    
    # Initialize a list of gaps
    aligned_seq = ['-'] * scheme_length

    # AHo numbering in ANARCI typically goes from 1 to 149.
    # The 'pos_id' in ANARCI for AHo is the actual position number.
    for (pos_id, insertion_code), residue in domain_numbering:
        # AHo strictly shouldn't have insertion codes if used correctly for standard Ig,
        # but ANARCI returns them in specific formats. 
        # We assume standard integer positions for the AHo scaffolding.
        
        # Adjust 1-based index to 0-based index
        index = pos_id - 1
        
        if 0 <= index < scheme_length:
            aligned_seq[index] = residue
            
    return "".join(aligned_seq)


def process_chunk(chunk_of_sequences):
    """
    Worker function to process a batch of sequences.
    
    Args:
        chunk_of_sequences: List of tuples [('ID_1', 'SEQ_1'), ('ID_2', 'SEQ_2'), ...]
        
    Returns:
        List of results: [('ID', 'ALIGNED_SEQ'), ...]
    """
    # Run ANARCI on the whole chunk at once (more efficient than 1 by 1)
    # output=False prevents printing to stdout, returns python objects instead
    results = anarci(chunk_of_sequences, scheme="aho", output=False, assign_germline=False)
    
    # Unpack results: results is a tuple (numbering_list, details_list, hit_list)
    numbering_list, _, _ = results
    
    processed_data = []
    
    # Iterate through the original input sequences and match with results
    for i, (seq_id, _) in enumerate(chunk_of_sequences):
        # Extract the specific result for this sequence
        # We reconstruct the tuple structure ANARCI would have returned for a single item
        single_result = (
            [numbering_list[i]] if numbering_list[i] else [], 
            None, 
            None
        )
        
        aligned_seq = parse_anarci_output(single_result)
        
        if aligned_seq:
            processed_data.append((seq_id, aligned_seq))
        else:
            # Handle cases where alignment failed (not an antibody or poor quality)
            processed_data.append((seq_id, None))
            
    return processed_data


def parallel_align_sequences(raw_sequences, n_jobs=None, chunk_size=100, verbose=False):
    """
    Parallel alignment with smooth, per-sequence progress tracking.
    
    Args:
        raw_sequences: List of tuples [('ID', 'SEQ'), ...]
        n_jobs: Number of cores (default: all - 1)
        chunk_size: Number of sequences per worker batch. 
                    Smaller = smoother progress bar but slightly more overhead.
                    100-500 is usually a sweet spot for ANARCI.
    Returns:
        List of tuples [('ID', 'ALIGNED_SEQ'), ...] if input is a single sequence,
        List of tuples [(('ID_1', 'ID_2'), 'ALIGNED_SEQ'), ...] if input is a paired sequence
    """
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    # 1. Create many fixed-size chunks
    # This ensures that workers report back frequently
    chunks = [raw_sequences[i:i + chunk_size] for i in range(0, len(raw_sequences), chunk_size)]
    
    if verbose:
        print(f"Aligning {len(raw_sequences)} sequences on {n_jobs} cores...")
        print(f"Work split into {len(chunks)} batches (approx {chunk_size} seqs/batch).")
    
    results = []
    
    # 2. Setup the Pool and Progress Bar
    # total=len(raw_sequences) lets the bar represent ACTUAL sequences, not chunks
    pbar = tqdm(total=len(raw_sequences), unit="seq", desc="ANARCI Alignment")
    
    with Pool(processes=n_jobs) as pool:
        # 3. Use imap_unordered
        # imap_unordered yields results as soon as *any* worker finishes a batch.
        # This prevents the progress bar from stalling if chunk #1 is slower than chunk #2.
        for batch_result in pool.imap_unordered(process_chunk, chunks):
            
            # Aggregate results
            results.extend(batch_result)
            
            # Update progress bar by the ACTUAL number of sequences in this batch
            pbar.update(len(batch_result))
            
    pbar.close()
    
    return results


# --- usage example ---
if __name__ == "__main__":
    # Example raw sequences (Heavy and Light)
    # Note: AHo works for both Heavy and Light chains without changing parameters
    raw_input = [
        ("H_chain_1", "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYYYYGLDVWGQGTTVTVSS"),
        ("L_chain_1", "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKVPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQKYNSAPLTFGGGTKVEIK"),
        ("Trastuzumab_H", "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"),
    ]

    # Run alignment
    aligned_data = parallel_align_sequences(raw_input)
    
    # Run alignment without parallelization
    # aligned_data = process_chunk(raw_input)

    # Display
    print("\n--- Results ---")
    for seq_id, seq_aho in aligned_data:
        if seq_aho:
            print(f">{seq_id}\n{seq_aho} (Length: {len(seq_aho)})")
        else:
            print(f">{seq_id}\nAlignment Failed")

    # Optional: Convert to DataFrame for easy saving
    df = pd.DataFrame(aligned_data, columns=['Id', 'AHo_Sequence'])
    print(df.head())

    breakpoint()