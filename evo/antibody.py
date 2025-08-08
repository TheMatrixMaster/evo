from typing import List
from abnumber import Chain

from .sequence import remove_spaces


def get_cdr(seq: str) -> List[str]:
    cdrs = Chain(remove_spaces([seq])[0], scheme='imgt')
    return [cdrs.cdr1_seq, cdrs.cdr2_seq, cdrs.cdr3_seq]


def get_frs(seq: str) -> List[str]:
    frs = Chain(remove_spaces([seq])[0], scheme='imgt')
    return [frs.fr1_seq, frs.fr2_seq, frs.fr3_seq, frs.fr4_seq]


