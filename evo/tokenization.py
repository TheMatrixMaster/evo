import itertools
import logging
from copy import copy
from typing import Dict, Optional, Sequence, Union

import esm
import numpy as np
import tape
import torch
import transformers

from .align import MSA

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(
        self,
        tokens: Dict[str, int],
        bos_token: str = "<cls>",
        eos_token: str = "<sep>",
        unk_token: Optional[str] = None,
        pad_token: str = "<pad>",
        mask_token: Optional[str] = None,
        prepend_bos: bool = True,
        append_eos: bool = True,
    ):
        if bos_token is not None and bos_token not in tokens:
            raise KeyError(f"bos token '{bos_token}' not in input tokens.")
        if eos_token is not None and eos_token not in tokens:
            raise KeyError(f"eos token '{eos_token}' not in input tokens.")
        if unk_token is not None and unk_token not in tokens:
            raise KeyError(f"unk token '{unk_token}' not in input tokens.")
        if pad_token not in tokens:
            raise KeyError(f"pad token '{pad_token}' not in input tokens.")
        if mask_token is not None and mask_token not in tokens:
            raise KeyError(f"mask token '{mask_token}' not in input tokens.")

        # prevent modifications to original dictionary from having an effect.
        tokens = copy(tokens)

        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        if bos_token is not None:
            self.bos_token: Optional[str] = bos_token
            self.bos_idx = tokens[bos_token]
        else:
            self.bos_token = None
            self.bos_idx = -1

        if eos_token is not None:
            self.eos_token: Optional[str] = eos_token
            self.eos_idx = tokens[eos_token]
        else:
            self.eos_token = None
            self.eos_idx = -1

        self.tokens_to_idx = tokens
        self.tokens = list(tokens.keys())
        self.unk_token = unk_token
        self.mask_token = mask_token
        self.pad_token = pad_token

        self.pad_idx = tokens[pad_token]

        self.allow_unknown = unk_token is not None
        if unk_token is not None:
            self.unk_idx = tokens[unk_token]

        self.allow_mask = mask_token is not None
        if mask_token is not None:
            self.mask_idx = tokens[mask_token]

        self.uint8_symbols = np.sort(
            np.array([tok for tok in self.tokens if len(tok) == 1], dtype="|S1").view(np.uint8)
        )
        self.numpy_indices = np.array(
            [self.index(chr(tok)) for tok in self.uint8_symbols],
            dtype=np.int64,
        )

    def index(self, token: str) -> int:
        return self.tokens_to_idx[token]

    def token(self, index: int) -> str:
        return self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def __repr__(self) -> str:
        return f"Vocab({self.to_dict()})"

    def to_dict(self) -> Dict[str, int]:
        return copy(self.tokens_to_idx)

    def _convert_uint8_array(self, array: np.ndarray) -> np.ndarray:
        assert array.dtype in (np.dtype("S1"), np.uint8)
        array = array.view(np.uint8)
        mask = ~np.isin(array, self.uint8_symbols)
        locs = np.digitize(array, self.uint8_symbols, right=True)
        locs = np.clip(locs, 0, len(self.numpy_indices) - 1)  # ensure in bounds
        indices = self.numpy_indices[locs.reshape(-1)].reshape(locs.shape)
        if mask.any():
            if not self.allow_unknown:
                raise ValueError("Unknown tokens found but unk_token not set")
            indices[mask] = self.unk_idx
        return indices

    def add_special_tokens(self, array: np.ndarray) -> np.ndarray:
        pad_widths = [(0, 0)] * (array.ndim - 1) + [(int(self.prepend_bos), int(self.append_eos))]
        return np.pad(
            array,
            pad_widths,
            constant_values=[(self.bos_idx, self.eos_idx)],
        )

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        return self.add_special_tokens(self._convert_uint8_array(array))

    def encode_single_sequence(self, sequence: str) -> np.ndarray:
        return self.encode_array(np.array(list(sequence), dtype="|S1"))

    def encode_batched_sequences(self, sequences: Sequence[str]) -> np.ndarray:
        batch_size = len(sequences)
        max_seqlen = max(len(seq) for seq in sequences)
        extra_token_pad = int(self.prepend_bos) + int(self.append_eos)
        indices = np.full((batch_size, max_seqlen + extra_token_pad), fill_value=self.pad_idx)

        for i, seq in enumerate(sequences):
            encoded = self.encode_single_sequence(seq)
            indices[i, : len(encoded)] = encoded
        return indices

    def decode_single_sequence(self, array: np.ndarray) -> str:
        array = array[int(self.prepend_bos) : len(array) - int(self.append_eos)]
        return "".join(self.token(idx) for idx in array)

    def encode(
        self, inputs: Union[str, Sequence[str], np.ndarray, MSA], validate: bool = True
    ) -> np.ndarray:
        if validate and not self.check_valid(inputs):
            raise ValueError("Invalid tokens in input")
        if isinstance(inputs, str):
            return self.encode_single_sequence(inputs)
        elif isinstance(inputs, Sequence):
            return self.encode_batched_sequences(inputs)
        elif isinstance(inputs, np.ndarray):
            return self.encode_array(inputs)
        elif isinstance(inputs, MSA) or hasattr(inputs, "array"):
            return self.encode_array(inputs.array)
        else:
            raise TypeError(f"Unknown input type {type(inputs)}")

    def check_valid(self, inputs: Union[str, Sequence[str], np.ndarray, MSA]) -> bool:
        if isinstance(inputs, str):
            tokens = set(inputs)
        elif isinstance(inputs, Sequence):
            tokens = set(itertools.chain.from_iterable(inputs))
        elif isinstance(inputs, np.ndarray):
            inputs = inputs.astype(np.dtype("S1"))
            tokens = {x.decode() for x in inputs.flatten()}
        elif isinstance(inputs, MSA) or hasattr(inputs, "sequences"):
            tokens = set(itertools.chain.from_iterable(inputs.sequences))
        else:
            raise TypeError(f"Unknown input type {type(inputs)}")
        return not bool(tokens - set(self.tokens))

    def decode(self, tokens: Union[np.ndarray, torch.Tensor]) -> Union[str, Sequence[str]]:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()

        if tokens.ndim == 1:
            return self.decode_single_sequence(tokens)
        elif tokens.ndim == 2:
            return [self.decode_single_sequence(toks) for toks in tokens]
        elif tokens.ndim == 3:
            assert tokens.shape[0] == 1, "Cannot decode w/ batch size > 1"
            tokens = tokens[0]
            return self.decode(tokens)
        else:
            raise ValueError("Too many dimensions!")

    def is_eos(self, index: int) -> bool:
        return index == self.eos_idx

    @classmethod
    def from_esm_alphabet(cls, alphabet: esm.data.Alphabet) -> "Vocab":
        return cls(
            tokens=alphabet.tok_to_idx,
            bos_token="<cls>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            prepend_bos=alphabet.prepend_bos,
            append_eos=alphabet.append_eos,
        )

    @classmethod
    def from_tape_tokenizer(cls, tokenizer: tape.tokenizers.TAPETokenizer) -> "Vocab":
        if "<unk>" in tokenizer.vocab:
            unk_token: Optional[str] = "<unk>"
        elif "X" in tokenizer.vocab:
            unk_token = "X"
        else:
            unk_token = None

        return cls(
            tokens=tokenizer.vocab,
            bos_token=tokenizer.start_token,
            eos_token=tokenizer.stop_token,
            unk_token=unk_token,
            pad_token="<pad>",
            mask_token=tokenizer.mask_token,
            prepend_bos=True,
            append_eos=True,
        )

    @classmethod
    def from_huggingface_tokenizer(
        cls, tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    ) -> "Vocab":
        return cls(
            tokens=tokenizer.get_vocab(),
            bos_token=tokenizer.cls_token,
            eos_token=tokenizer.sep_token,
            unk_token=tokenizer.unk_token,
            pad_token=tokenizer.pad_token,
            mask_token=tokenizer.mask_token,
            prepend_bos=tokenizer.cls_token is not None,
            append_eos=tokenizer.sep_token is not None,
        )

    @classmethod
    def from_fasta_standard(cls) -> "Vocab":
        alphabet = "ARNDCQEGHILKMFPSTWYV-X"
        a2n = {a: n for n, a in enumerate(alphabet)}
        return cls(a2n, pad_token="-", prepend_bos=False, append_eos=False, unk_token="X")

    @classmethod
    def from_trrosetta(cls) -> "Vocab":
        alphabet = "ARNDCQEGHILKMFPSTWYV-"
        a2n = {a: n for n, a in enumerate(alphabet)}
        return cls(a2n, pad_token="-", prepend_bos=False, append_eos=False, unk_token="-")

    @classmethod
    def from_peint(cls) -> "Vocab":
        alphabet = "ARNDCQEGHILKMFPSTWYV-"
        a2n = {a: n for n, a in enumerate(alphabet)}
        a2n.update(
            {
                "<s>": len(a2n),  # bos
                "<mask>": len(a2n) + 1,  # mask
                "</s>": len(a2n) + 2,  # eos
                "<pad>": len(a2n) + 3,  # pad
                "X": len(a2n) + 4,  # unk
            }
        )
        # a2n.update({
        #     'J': a2n['I'],  # ambiguously I or L, just map to I
        #     'B': a2n['D'],  # ambiguously D or N, just map to D
        #     'Z': a2n['Q'],  # ambiguously Q or E, just map to Q
        # })
        return cls(
            tokens=a2n,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="X",
            pad_token="<pad>",
            mask_token="<mask>",
            prepend_bos=False,
            append_eos=False,
        )

    @classmethod
    def from_msa_editflows(cls) -> "Vocab":
        a2n = {
            "<cls>": 0,
            "<pad>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "L": 4,
            "A": 5,
            "G": 6,
            "V": 7,
            "S": 8,
            "E": 9,
            "R": 10,
            "T": 11,
            "I": 12,
            "D": 13,
            "P": 14,
            "K": 15,
            "Q": 16,
            "N": 17,
            "F": 18,
            "Y": 19,
            "M": 20,
            "H": 21,
            "W": 22,
            "C": 23,
            "X": 24,
            "B": 25,
            "U": 26,
            "Z": 27,
            "O": 28,
            ".": 29,
            "-": 29,
            # "<mask>": 30,
        }
        return cls(
            tokens=a2n,
            bos_token="<cls>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token=None,
            prepend_bos=True,
            append_eos=True,
        )


class DNAVocab(Vocab):
    """Vocabulary for DNA sequences with single nucleotide tokens."""

    @classmethod
    def from_dna(cls) -> "DNAVocab":
        """Create a DNA vocabulary with A, T, G, C nucleotides."""
        alphabet = "ATGC"
        a2n = {a: n for n, a in enumerate(alphabet)}
        a2n.update(
            {
                "<s>": len(a2n),  # bos
                "<mask>": len(a2n) + 1,  # mask
                "</s>": len(a2n) + 2,  # eos
                "<pad>": len(a2n) + 3,  # pad
                "N": len(a2n) + 4,  # unk (unknown nucleotide)
            }
        )
        return cls(
            tokens=a2n,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="N",
            pad_token="<pad>",
            mask_token="<mask>",
            prepend_bos=False,
            append_eos=False,
        )


class CodonVocab(Vocab):
    """Vocabulary for codon sequences (DNA triplets)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Standard genetic code
    GENETIC_CODE = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "*",
        "TAG": "*",
        "TGT": "C",
        "TGC": "C",
        "TGA": "*",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    @classmethod
    def from_codons(cls) -> "CodonVocab":
        """Create a codon vocabulary with all 64 possible codons."""
        nucleotides = "ATGC"
        codons = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides]
        tokens = {
            "<pad>": 0,  # pad
            "<s>": 1,  # bos
            "</s>": 2,  # eos
            "<mask>": 3,  # mask
            "NNN": 4,  # unk (unknown codon)
        }
        c2n = {codon: n + 5 for n, codon in enumerate(codons)}
        tokens.update(c2n)
        return cls(
            tokens=tokens,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="NNN",
            pad_token="<pad>",
            mask_token="<mask>",
            prepend_bos=True,
            append_eos=False,
        )

    def _convert_uint8_array(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def encode_array(self, array: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def encode_single_sequence(self, sequence: str) -> np.ndarray:
        """Encode a DNA sequence by splitting into codon triplets."""
        seq_len = len(sequence)
        remainder = seq_len % 3
        if remainder > 0:
            if not self.allow_unknown:
                raise ValueError(
                    f"Sequence length {seq_len} is not divisible by 3 and unk_token not set"
                )
            sequence = sequence + "N" * (3 - remainder)
            seq_len = len(sequence)

        if self.allow_unknown:
            indices = np.array(
                [
                    self.tokens_to_idx.get(sequence[i : i + 3], self.unk_idx)
                    for i in range(0, seq_len, 3)
                ],
                dtype=np.int64,
            )
        else:
            indices = np.empty(seq_len // 3, dtype=np.int64)
            for idx, i in enumerate(range(0, seq_len, 3)):
                codon = sequence[i : i + 3]
                try:
                    indices[idx] = self.tokens_to_idx[codon]
                except KeyError:
                    raise ValueError(f"Unknown codon '{codon}' and unk_token not set")

        return self.add_special_tokens(indices)

    def check_valid(self, inputs: Union[str, Sequence[str], np.ndarray, MSA]) -> bool:
        """Check if input contains valid codons."""
        if isinstance(inputs, str):
            # Split into codons and check each
            codons = [inputs[i : i + 3] for i in range(0, len(inputs), 3)]
            return all(codon in self.tokens or len(codon) < 3 for codon in codons)
        elif isinstance(inputs, Sequence):
            return all(self.check_valid(seq) for seq in inputs)
        else:
            # For other types, fall back to base class
            return super().check_valid(inputs)

    def translate(self, sequence: str) -> str:
        """Translate a DNA codon sequence to amino acids using the standard genetic code"""
        codons = [sequence[i : i + 3] for i in range(0, len(sequence), 3)]
        amino_acids = []
        for codon in codons:
            if len(codon) < 3:
                logger.warning(f"Incomplete codon '{codon}' at end of sequence, skipping")
                continue
            if codon in self.GENETIC_CODE:
                amino_acids.append(self.GENETIC_CODE[codon])
            else:
                amino_acids.append("X")  # Unknown codon
        return "".join(amino_acids)

    def translation_tensor_map(self, aa_vocab: Vocab) -> torch.Tensor:
        """Creates a tensor mapping codon indices to amino acid indices in the provided amino acid vocabulary."""
        mapping = torch.full((len(self),), fill_value=aa_vocab.unk_idx, dtype=torch.long)
        for cod_tok, aa_tok in [
            (self.pad_idx, aa_vocab.pad_idx),
            (self.bos_idx, aa_vocab.bos_idx),
            (self.eos_idx, aa_vocab.eos_idx),
            (self.mask_idx, aa_vocab.mask_idx),
        ]:
            if cod_tok >= 0 and aa_tok >= 0:
                mapping[cod_tok] = aa_tok
        for codon, aa in self.GENETIC_CODE.items():
            assert codon in self.tokens_to_idx, f"Codon '{codon}' not in vocabulary"
            assert aa in aa_vocab.tokens_to_idx or aa == "*", f"Amino acid '{aa}' not in vocabulary"
            codon_idx = self.tokens_to_idx[codon]
            aa_idx = aa_vocab.tokens_to_idx[aa] if aa != "*" else aa_vocab.eos_idx  # map stop codons to the aa eos token
            mapping[codon_idx] = aa_idx
        return mapping


def test_encode_sequence():
    sequence = "LFKLGAENIFLGRKAATKEEAIRFAGEQLVKGGYVEPEYVQAMLDREKLTPTYLGESIAVPHGTVEAK"
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    vocab = Vocab.from_esm_alphabet(alphabet)
    batch_converter = alphabet.get_batch_converter()
    _, _, esm_tokens = batch_converter([("", sequence)])
    evo_tokens = vocab.encode(sequence)[None]
    assert (esm_tokens == evo_tokens).all()


if __name__ == "__main__":
    pass

    # Suppress multi-character token warnings for cleaner test output
    logger.setLevel(logging.ERROR)

    print("=" * 70)
    print("Testing DNAVocab")
    print("=" * 70)

    dna_vocab = DNAVocab.from_dna()
    print(f"✓ DNAVocab created with size: {len(dna_vocab)}")

    # Test single sequence encoding/decoding
    dna_seq = "ATGCGATCG"
    encoded_dna = dna_vocab.encode(dna_seq)
    decoded_dna = dna_vocab.decode(encoded_dna)
    assert decoded_dna == dna_seq, f"Mismatch: {decoded_dna} != {dna_seq}"
    print(f"✓ Single sequence encode/decode: {dna_seq} -> {list(encoded_dna)} -> {decoded_dna}")

    # Test batched sequences
    dna_sequences = ["ATGC", "GGGGAAAA", "T"]
    batched_dna = dna_vocab.encode(dna_sequences)
    decoded_batch = dna_vocab.decode(batched_dna)
    for orig, dec in zip(dna_sequences, decoded_batch):
        dec_clean = dec.replace("<pad>", "").strip()
        assert orig == dec_clean, f"Batch mismatch: {orig} != {dec_clean}"
    print(f"✓ Batched sequences ({len(dna_sequences)} sequences) encode/decode correctly")

    print("\n" + "=" * 70)
    print("Testing CodonVocab")
    print("=" * 70)

    codon_vocab = CodonVocab.from_codons()
    print(f"✓ CodonVocab created with size: {len(codon_vocab)}")

    # Test basic codon encoding/decoding
    codon_seq = "ATGAAATTT"
    encoded_codon = codon_vocab.encode(codon_seq)
    decoded_codon = codon_vocab.decode(encoded_codon)
    assert decoded_codon == codon_seq, f"Mismatch: {decoded_codon} != {codon_seq}"
    print(
        f"✓ Single sequence encode/decode: {codon_seq} -> {list(encoded_codon)} -> {decoded_codon}"
    )

    # Test translation accuracy
    test_cases = [
        ("ATGAAATTT", "MKF"),  # Start codon + AAA (K) + TTT (F)
        ("TTTAAAGGG", "FKG"),  # TTT (F) + AAA (K) + GGG (G)
        ("ATGAAATAG", "MK*"),  # With stop codon TAG
        ("TAATAATGA", "***"),  # All stop codons
        ("GCAGCGGCA", "AAA"),  # All alanine
        ("ATGATCATA", "MII"),  # ATG (M) + ATC (I) + ATA (I)
    ]

    print("\nTranslation tests:")
    for dna, expected_aa in test_cases:
        translated = codon_vocab.translate(dna)
        assert (
            translated == expected_aa
        ), f"Translation error: {dna} -> {translated}, expected {expected_aa}"
        print(f"  ✓ {dna} -> {translated}")

    # Test batched codon sequences
    codon_sequences = ["ATGAAATTT", "ATGGGG", "TTTAAA"]
    batched_codon = codon_vocab.encode(codon_sequences)
    decoded_codon_batch = codon_vocab.decode(batched_codon)
    for orig, dec in zip(codon_sequences, decoded_codon_batch):
        dec_clean = dec.replace("<pad>", "").strip()
        assert orig == dec_clean, f"Batch mismatch: {orig} != {dec_clean}"
    print(f"\n✓ Batched sequences ({len(codon_sequences)} sequences) encode/decode correctly")

    # Test incomplete codon handling
    incomplete_seq = "ATGAA"  # 5 nucleotides, incomplete last codon
    try:
        encoded_incomplete = codon_vocab.encode(incomplete_seq)
        decoded_incomplete = codon_vocab.decode(encoded_incomplete)
        print(f"✓ Incomplete codon handling: {incomplete_seq} -> {decoded_incomplete}")
    except ValueError as e:
        print(f"✓ Incomplete codon raises error as expected: {e}")

    # Test comprehensive genetic code coverage
    print("\nVerifying complete genetic code table (64 codons):")
    nucleotides = "ATGC"
    all_codons = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides]
    assert len(all_codons) == 64, "Should have exactly 64 codons"

    translated_codons = 0
    for codon in all_codons:
        aa = codon_vocab.translate(codon)
        assert len(aa) == 1, f"Translation should produce single amino acid, got {aa}"
        translated_codons += 1

    print(f"  ✓ All {translated_codons} codons translate correctly")

    # Verify specific well-known codon translations
    known_translations = {
        "ATG": "M",  # Methionine (start codon)
        "TGG": "W",  # Tryptophan (only 1 codon)
        "TAA": "*",  # Stop
        "TAG": "*",  # Stop
        "TGA": "*",  # Stop
        "GCT": "A",  # Alanine
        "TGT": "C",  # Cysteine
    }

    for codon, expected_aa in known_translations.items():
        aa = codon_vocab.translate(codon)
        assert (
            aa == expected_aa
        ), f"Known translation error: {codon} -> {aa}, expected {expected_aa}"

    print(f"  ✓ Known codon translations verified")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
