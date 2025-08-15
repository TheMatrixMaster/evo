from typing import List

import numpy as np
import pandas as pd
from ete3 import Tree


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


def df_to_ete3_tree(df: pd.DataFrame) -> Tree:
    """
    Convert a dataframe of edges (parent_name, child_name, branch_length)
    into an ete3 Tree object in 'format=3'.

    The tree is assumed to have exactly one root: a node that does not appear as any child's name.
    """

    # Create a lookup for all nodes
    # We'll store node_name -> ete3.Tree instance
    nodes = {}

    # Create all nodes, but do not connect them yet
    for parent_name, child_name, branch_length in df[
        ["parent_name", "child_name", "branch_length"]
    ].itertuples(index=False):
        if parent_name not in nodes:
            nodes[parent_name] = Tree(name=parent_name)  # parent node
        if child_name not in nodes:
            nodes[child_name] = Tree(name=child_name)  # child node

    # Connect child to parent, setting the branch length (dist) on the child side
    # We'll also track who is the parent of whom in parent_of
    parent_of = {}
    for parent_name, child_name, branch_length in df[
        ["parent_name", "child_name", "branch_length"]
    ].itertuples(index=False):
        parent_node = nodes[parent_name]
        child_node = nodes[child_name]
        # Set branch length on the child
        child_node.dist = branch_length
        # Attach child to parent's children
        parent_node.add_child(child_node)
        parent_of[child_name] = parent_name

    # Find the root as the node that never appears as a child
    all_nodes = set(nodes.keys())
    all_children = set(parent_of.keys())
    root_candidates = all_nodes - all_children
    if len(root_candidates) != 1:
        raise ValueError(f"Expected exactly 1 root, found: {root_candidates}")
    root_name = root_candidates.pop()

    # Return the root as an ete3 Tree
    # By construction, nodes[root_name] is now the root and has all children below it
    tree = nodes[root_name]

    # If you want to ensure 'format=3' usage downstream, you can do:
    #   newick_str = tree.write(format=3)
    #   tree = Tree(newick_str, format=3)
    # But typically, once you have 'tree' as a Tree object, you can simply
    # use `tree.write(format=3)` or pass format=3 when you want to export.
    return tree
