import shutil
from typing import Tuple

import numpy as np
from biotite.application.dssp import DsspApp
from biotite.structure.io import pdb


def pdb_path_to_biotite_atom_array(file_path):
    """Returns an atom array stack - if only one chain, can take just the first of the tuple to get the atom array."""
    pdb_file = pdb.PDBFile.read(file_path)
    atom_array = pdb.get_structure(pdb_file)
    return atom_array


def get_mkdssp_path() -> str:
    mkdssp_in_path = shutil.which("mkdssp")
    if mkdssp_in_path:
        return mkdssp_in_path
    else:
        return "mkdssp not found in PATH. Please install `conda install salilab::dssp` or update your PATH."


DSSP_PATH = get_mkdssp_path()


def pdb_path_to_secondary_structure(pdb_path) -> Tuple[float, float, str]:
    structure_atom_array = pdb_path_to_biotite_atom_array(pdb_path)
    out = DsspApp.annotate_sse(structure_atom_array, DSSP_PATH)
    dssp_annotation = "".join(out)
    alpha_percentage = np.array([x == "H" for x in dssp_annotation]).sum() / len(dssp_annotation)
    beta_percentage = np.array([x == "E" for x in dssp_annotation]).sum() / len(dssp_annotation)
    return alpha_percentage, beta_percentage, dssp_annotation
