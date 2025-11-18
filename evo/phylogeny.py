import subprocess
from pathlib import Path
from typing import List, Tuple, Union

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


# R script that reads newick from stdin or file, outputs CSV to stdout
R_SCRIPT = """
library(ape)
args <- commandArgs(trailingOnly = TRUE)

# Read tree from file or stdin
if (length(args) > 0 && args[1] != "-") {
    tree <- read.tree(args[1])
} else {
    tree <- read.tree(file("stdin"))
}

# Compute all pairwise node distances
dist <- dist.nodes(tree)

# Get node labels
n_tips <- length(tree$tip.label)
n_internal <- tree$Nnode

if (is.null(tree$node.label) || all(tree$node.label == "")) {
    internal_labels <- paste0("internal_", seq_len(n_internal) - 1)
} else {
    internal_labels <- tree$node.label
    empty_idx <- which(internal_labels == "" | is.na(internal_labels))
    internal_labels[empty_idx] <- paste0("internal_", empty_idx - 1)
}

all_labels <- c(tree$tip.label, internal_labels)
rownames(dist) <- all_labels
colnames(dist) <- all_labels

# Write to stdout
write.csv(dist, file = stdout(), quote = FALSE)
"""


def get_patristic_distances(
    newick_input: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute patristic distance matrix for all nodes (tips + internal).

    Uses R's ape::dist.nodes() - transfers data via stdout (no temp files for output).

    Parameters
    ----------
    newick_input : str or Path
        Newick string or path to Newick file.

    Returns
    -------
    distances : np.ndarray
        Symmetric matrix of pairwise patristic distances.
    labels : np.ndarray
        Node labels in matrix order (tips first, then internal nodes).

    Examples
    --------
    >>> newick = "((A:0.1,B:0.2)AB:0.3,(C:0.4,D:0.5)CD:0.6)root;"
    >>> dist, labels = get_patristic_distances(newick)
    >>> i, j = np.where(labels == 'A')[0][0], np.where(labels == 'B')[0][0]
    >>> print(f"Distance A-B: {dist[i,j]}")  # 0.3
    """
    newick_str = str(newick_input)
    is_file = Path(newick_str).is_file()

    # Build R command
    if is_file:
        # Pass file path as argument
        cmd = ["Rscript", "-e", R_SCRIPT, newick_str]
        stdin_data = None
    else:
        # Pass newick string via stdin
        cmd = ["Rscript", "-e", R_SCRIPT, "-"]
        stdin_data = newick_str

    # Run R
    try:
        result = subprocess.run(cmd, input=stdin_data, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("Rscript not found. Install R and ensure it's in your PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"R failed:\n{e.stderr}")

    # Parse CSV from stdout
    lines = result.stdout.strip().split("\n")

    # First line is header with labels
    header = lines[0].split(",")[1:]  # Skip empty first column
    labels = np.array([label.strip('"') for label in header])

    # Remaining lines are data
    n = len(labels)
    distances = np.zeros((n, n), dtype=np.float64)

    for i, line in enumerate(lines[1:]):
        values = line.split(",")[1:]  # Skip row label
        distances[i] = [float(v) for v in values]

    return distances, labels


def get_distance(newick_input: Union[str, Path], node1: str, node2: str) -> float:
    """
    Get patristic distance between two specific nodes.

    Parameters
    ----------
    newick_input : str or Path
        Newick string or path to Newick file.
    node1, node2 : str
        Node names.

    Returns
    -------
    distance : float
        Patristic distance between the two nodes.
    """
    distances, labels = get_patristic_distances(newick_input)

    idx1 = np.where(labels == node1)[0]
    idx2 = np.where(labels == node2)[0]

    if len(idx1) == 0:
        raise ValueError(f"Node '{node1}' not found. Available: {labels.tolist()}")
    if len(idx2) == 0:
        raise ValueError(f"Node '{node2}' not found. Available: {labels.tolist()}")

    return distances[idx1[0], idx2[0]]
