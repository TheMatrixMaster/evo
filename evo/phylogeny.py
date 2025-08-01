from typing import List

import numpy as np


def get_quantization_points_from_geometric_grid(
    quantization_grid_center: float = 0.03,
    quantization_grid_step: float = 1.1,
    quantization_grid_num_steps: int = 64,
) -> List[str]:
    """
    Generates a list of quantization points as strings using a geometric grid.

    Args:
        quantization_grid_center (float, optional): The center value of the geometric grid. Defaults to 0.03.
        quantization_grid_step (float, optional): The multiplicative step between grid points. Defaults to 1.1.
        quantization_grid_num_steps (int, optional): Number of steps to take in each direction from the center. Defaults to 64.

    Returns:
        List[str]: List of quantization points formatted as strings.
    """
    quantization_points = [
        ("%.8f" % (quantization_grid_center * quantization_grid_step**i))
        for i in range(-quantization_grid_num_steps, quantization_grid_num_steps + 1, 1)
    ]
    return quantization_points


def get_quantile_idx(quantiles: List[float], t: float) -> int:
    """Returns the quantile index that time t falls in.

    Args:
        quantiles (List[float]): List of quantile boundaries (length N).
        t (float): Value to locate.

    Returns:
        int: Index i such that t in [quantiles[i], quantiles[i+1]), or edge index if out of bounds.
    """
    if t < quantiles[0]:
        return 0
    elif t > quantiles[-1]:
        return len(quantiles) - 2

    idx_to_insert_t = np.searchsorted(quantiles, t, "right")
    return idx_to_insert_t - 1


VALID_BINS = np.array([float(f) for f in get_quantization_points_from_geometric_grid()])
