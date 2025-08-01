import contextlib
import functools
from typing import Callable, Generator, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
from tqdm.auto import trange

from .tokenization import Vocab

TensorLike = TypeVar("TensorLike", np.ndarray, torch.Tensor)
T = TypeVar("T")


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward. Taken from github.com/pytorch/fairseq"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def coerce_numpy(func: Callable) -> Callable:
    @functools.wraps(func)
    def make_torch_args(*args, **kwargs):
        is_numpy = False
        update_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.from_numpy(arg)
                is_numpy = True
            update_args.append(arg)
        update_kwargs = {}
        for kw, arg in kwargs.items():
            if isinstance(args, np.ndarray):
                arg = torch.from_numpy(arg)
                is_numpy = True
            update_kwargs[kw] = arg

        output = func(*update_args, **update_kwargs)

        if is_numpy:
            output = recursive_make_numpy(output)

        return output

    return make_torch_args


def recursive_make_torch(item):
    if isinstance(item, np.ndarray):
        return torch.from_numpy(item)
    elif isinstance(item, (tuple, list)):
        return type(item)(recursive_make_torch(el) for el in item)
    elif isinstance(item, dict):
        return {kw: recursive_make_torch(arg) for kw, arg in item.items()}
    else:
        return item


def recursive_make_numpy(item):
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy()
    elif isinstance(item, (tuple, list)):
        return type(item)(recursive_make_numpy(el) for el in item)
    elif isinstance(item, dict):
        return {kw: recursive_make_numpy(arg) for kw, arg in item.items()}
    else:
        return item


def collate_tensors(sequences: Sequence[TensorLike], constant_value=0, dtype=None) -> TensorLike:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array


def collate_list_of_dicts(
    sequences: Sequence[dict],
    batch_keys: Sequence[str],
    constant_value=0,
) -> dict:
    col_batch = {}
    for key in batch_keys:
        if isinstance(sequences[0][key], (float, int)):
            col_batch[key] = torch.tensor([item[key] for item in sequences])
        elif isinstance(sequences[0][key], (np.ndarray, torch.Tensor)):
            col_batch[key] = collate_tensors(
                [item[key] for item in sequences], constant_value=constant_value
            )
        else:
            col_batch[key] = [item[key] for item in sequences]
    return col_batch


def mask_tensor(
    x: torch.Tensor,
    vocab: Vocab,
    mask_prob: float = 0.15,
    random_token_prob: float = 0.1,
    leave_unmasked_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    special_tokens = [vocab.bos_idx, vocab.eos_idx, vocab.pad_idx, vocab.mask_idx]
    special_tokens = torch.Tensor(special_tokens).to(x.device).to(x.dtype)
    keep_mask = torch.isin(x, special_tokens)
    random_probs = torch.rand_like(x, dtype=torch.float)
    random_probs[keep_mask] = 1.0  # Ensure special tokens are not masked
    do_mask = random_probs < mask_prob
    tgt = x.masked_fill(~do_mask, vocab.pad_idx)
    mask_with_token = random_probs < (mask_prob * (1 - leave_unmasked_prob))
    src = x.masked_fill(mask_with_token, vocab.mask_idx)
    mask_with_random = random_probs < (mask_prob * random_token_prob)
    rand_tokens = torch.randint_like(src, len(vocab))
    src[mask_with_random] = rand_tokens[mask_with_random]
    return src, tgt


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


@coerce_numpy
def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


def batched_iterator(
    data: Sequence[T],
    batch_size: int,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> Generator[Sequence[T], None, None]:
    num_examples = len(data)
    iterator = trange if verbose else range
    for start in iterator(0, num_examples, batch_size):
        batch = data[start : start + batch_size]
        if device is not None and isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        yield batch
