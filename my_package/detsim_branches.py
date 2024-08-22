"""
! All the pwd used in the following are in the context of my own analysis performed with the CCIN2P3 and cannot be accessed for unauthorized people.

Loading DETSIM Files, Trees, and Branches

This script provides utility functions for loading ROOT files, extracting trees, and retrieving branches and attributes
from the DETSIM simulation data. It includes error handling to ensure that the file and tree loading processes work
as expected.

Created by: Lukas Péron
Last Update: 22/08/2024

Overview:
---------
The script includes the following functions:
1. `load_file(number)`: Opens a DETSIM ROOT file based on a provided identifier number.
2. `load_branches(tree, *kwargs)`: Retrieves specified branches from a given tree.
3. `load_attributes(entry, *kwargs)`: Extracts attributes from a specific tree entry.

Error Handling:
---------------
- The `load_file` function checks if the ROOT file is successfully opened. If the file cannot be opened or is corrupted
  (zombie file), an error is raised with a descriptive message.
- The `load_branches` and `load_attributes` functions assume that the provided branch or attribute names are valid. 
  If invalid names are provided, the function will raise an `AttributeError` or return `None`, depending on the ROOT framework's behavior.

Dependencies:
-------------
- ROOT

Instructions:
-------------
1. Ensure that ROOT is installed and configured in your environment.
2. Use the `load_file` function to open the desired DETSIM ROOT file.
3. Use `load_branches` and `load_attributes` to extract the necessary data from the tree.
"""

from ROOT import *

def load_file(number):
    """
    Opens a DETSIM ROOT file based on the provided identifier number.

    Parameters:
    -----------
    number : int or str
        Identifier number used to locate the specific DETSIM ROOT file.

    Returns:
    --------
    TFile
        The opened ROOT file.

    Raises:
    -------
    RuntimeError
        If the file cannot be opened or is corrupted (zombie file).
    """
    file = TFile.Open(f"/sps/atlas/s/stark/pourLukas2/Fornax_2021/detsim/all/user-detsim-Fornax2021_27Msun_40kpc_NMO_J23.1-{number}.root")
    if not file or file.IsZombie():
        raise RuntimeError(f"Error: Could not open the file user-detsim-Fornax2021_27Msun_40kpc_NMO_J23.1-{number}.root.")
    else:
        return file

def load_branches(tree, *kwargs):
    """
    Retrieves specified branches from a given tree.

    Parameters:
    -----------
    tree : TTree
        The ROOT TTree from which branches will be extracted.
    kwargs : str
        Names of the branches to be retrieved.

    Returns:
    --------
    list of TBranch
        A list of TBranch objects corresponding to the specified branch names.
    """
    return [tree.GetBranch(name) for name in kwargs]

def load_attributes(entry, *kwargs):
    """
    Extracts attributes from a specific tree entry.

    Parameters:
    -----------
    entry : TTree entry
        The entry from which attributes will be extracted.
    kwargs : str
        Names of the attributes to be retrieved.

    Returns:
    --------
    list
        A list of attribute values corresponding to the specified names.
    """
    return [getattr(entry, name) for name in kwargs]
