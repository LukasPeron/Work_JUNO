"""
Create Edge Index for Graph Neural Networks (GNN)

This script creates edge indices for Graph Neural Networks (GNN) by calculating the spatial and temporal distances 
between PMTs (Photomultiplier Tubes) in the JUNO experiment. It processes event data from DETSIM files, computes 
nearest neighbors based on arc distances and time differences, and saves the edge indices for each event.

Created by: Lukas Peron
Last Update: 01/09/2024

Overview:
---------
The script performs the following steps:
1. Load PMT location data and event data from text files.
2. Compute the arc distance between PMT locations for each event to find spatial neighbors.
3. Calculate the time difference between PMTs to find temporal neighbors.
4. Generate and save edge indices for spatial and temporal graphs based on the nearest neighbors.

Error Handling:
---------------
- The script assumes that the input files exist and are formatted correctly. If files are missing or corrupted, 
  the numpy file loading functions (`np.genfromtxt` and `np.loadtxt`) will raise an `IOError`.
- The calculations assume valid data with the correct shapes; otherwise, numpy functions may raise `ValueError` 
  or `IndexError` if the shapes do not align for matrix operations.
- The script assumes valid integer input from the user for `num_file`. If invalid input is provided, a `ValueError` 
  will be raised when converting the input to an integer.

Dependencies:
-------------
- numpy

Instructions:
-------------
1. Ensure that numpy is installed and configured in your environment.
2. Run the script and provide a valid file number when prompted.
3. The script will generate and save edge indices for spatial and temporal graphs in the specified directory.
"""

import numpy as np

pwd = "/sps/l2it/lperon/JUNO/txt/data_profiling/"
# Load the data
pmt_loc = np.genfromtxt("/pbs/home/l/lperon/work_JUNO/JUNO_info/PMTPos_CD_LPMT.csv", usecols=(0, 1, 2, 3))

def arc_dist(pmt1, pmt2):
    """
    Compute the arc distance between pairs of PMT locations.

    Parameters:
    -----------
    pmt1 : numpy.ndarray
        A 2D array representing the positions of PMTs.
    pmt2 : numpy.ndarray
        A 2D array representing the positions of PMTs.

    Returns:
    --------
    numpy.ndarray
        A 2D array containing the arc distances between all pairs of PMTs.

    Notes:
    ------
    The arc distance is calculated based on the angular separation of PMTs in 3D space, 
    assuming they are positioned on a sphere.
    """
    # R is the square root of the product of norms
    R = np.sqrt(np.linalg.norm(pmt1, axis=1)[:, None] * np.linalg.norm(pmt2, axis=1)[None, :])
    # Calculate the dot products and then compute the arccos
    cos_theta = np.einsum('ij,kj->ik', pmt1, pmt2) / (R ** 2)
    # Compute arc distance
    dist = R * np.arccos(np.clip(cos_theta, -1.0, 1.0))
    # Set the diagonal to Inf
    np.fill_diagonal(dist, np.inf)
    return dist

def main():
    """
    Main function to create edge indices for GNN based on spatial and temporal distances.

    Parameters:
    -----------
    None

    Returns:
    --------
    None

    Raises:
    -------
    IOError
        If the input files are missing or corrupted.
    ValueError
        If the shapes of the arrays do not align for matrix operations or if the input file number is invalid.
    IndexError
        If the indices calculated are out of bounds for the arrays.
    """
    try:
        num_file = int(input("Enter file number: "))
    except ValueError:
        raise ValueError("Invalid input: Please enter an integer value for the file number.")

    # Load event data from the specified file
    try:
        X_train = np.loadtxt(pwd + f"elecsim_data_file{num_file}.txt")
    except IOError:
        raise IOError(f"Error loading file: elecsim_data_file{num_file}.txt not found or is corrupted.")

    for num_event in range(len(X_train)):
        event = X_train[num_event]
        lst_pmt_on = np.where(event[::3] == 1)[0]
        positions = pmt_loc[lst_pmt_on, 1:]

        # Calculate spatial neighbors using arc distances
        try:
            dist_matrix = arc_dist(positions, positions)
            nearest_indices = np.argpartition(dist_matrix, 6, axis=1)[:, :6]
            row_indices = np.repeat(np.arange(dist_matrix.shape[0])[:, None], 6, axis=1)
            nearest = np.stack((row_indices, nearest_indices), axis=-1)
            nearest = np.reshape(nearest, (len(nearest) * 6, 2))
            np.savetxt(pwd + "graph_edge/" + f"graph_edge_spat_file{num_file}_event{num_event}.txt", nearest)
        except (ValueError, IndexError) as e:
            print(f"Error calculating spatial neighbors for event {num_event} in file {num_file}: {e}")
            continue

        # Calculate temporal neighbors using time differences
        try:
            lst_t0 = event[lst_pmt_on * 3 + 2]
            diff_t0 = np.abs(lst_t0[:, None] - lst_t0[None, :])
            np.fill_diagonal(diff_t0, np.inf)
            nearest_indices_time = np.argpartition(diff_t0, 6, axis=1)[:, :6]
            row_indices_time = np.repeat(np.arange(diff_t0.shape[0])[:, None], 6, axis=1)
            nearest_time = np.stack((row_indices_time, nearest_indices_time), axis=-1)
            nearest_time = np.reshape(nearest_time, (len(nearest_time) * 6, 2))
            np.savetxt(pwd + "graph_edge/" + f"graph_edge_temp_file{num_file}_event{num_event}.txt", nearest_time)
        except (ValueError, IndexError) as e:
            print(f"Error calculating temporal neighbors for event {num_event} in file {num_file}: {e}")
            continue

if __name__ == "__main__":
    main()