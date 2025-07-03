# Load necessary libraries
import os
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import warnings
from scipy.spatial.transform import Rotation as R
from Bio.PDB import PDBParser
from Bio import PDB

warnings.filterwarnings("ignore")

# --- PDB Extraction Function ---
def extract_ca_coordinates(pdb_file):
    """
    Loads a PDB file and extracts C-alpha coordinates for each standard amino acid,
    returning them as a list of NumPy arrays, where each array is (1, 3) for a single residue.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    
    coords_per_residue = [] 
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True) and residue.has_id('CA'):
                    coords_per_residue.append(np.array([residue['CA'].coord]))
    
    return coords_per_residue


# --- calculate_rmsd_after_superposition function ---
def calculate_rmsd_after_superposition(coords1, coords2):
    """
    Calculates the Root Mean Square Deviation (RMSD) between two sets of 3D coordinates
    after optimally superposing coords2 onto coords1 using the Kabsch algorithm (via scipy).

    Args:
        coords1 (np.ndarray): N x 3 array of 3D coordinates (reference).
        coords2 (np.ndarray): M x 3 array of 3D coordinates (to be superposed).

    Returns:
        float: The minimal RMSD between the two sets of coordinates.
    """
    # Robustness checks for empty arrays
    if coords1.shape[0] == 0 and coords2.shape[0] == 0:
        return 0.0 # Both empty, perfectly "aligned" (no cost)
    elif coords1.shape[0] == 0 or coords2.shape[0] == 0:
        # One is empty, the other is not. This should incur a high cost.
        return 1000.0 # A high penalty, indicating a fundamental mismatch
    
    # Handle single-point comparison directly (avoids R.align_vectors issues for N=1)
    if coords1.shape[0] == 1 and coords2.shape[0] == 1:
        return np.linalg.norm(coords1[0] - coords2[0]) # Euclidean distance for 1 point is its RMSD

    # Standard sanity check for 3D coordinates
    if coords1.shape[1] != 3 or coords2.shape[1] != 3:
        raise ValueError("Input coordinate arrays must be (N, 3) or (M, 3).")

    # Center the coordinates
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    centered_coords1 = coords1 - centroid1
    centered_coords2 = coords2 - centroid2

    # --- Check for degenerate (all points identical / zero length after centering) inputs ---
    # If the sum of squared magnitudes of the centered vectors is effectively zero,
    # it means all points are identical, and alignment is trivial (RMSD is 0).
    is_coords1_degenerate = np.isclose(np.sum(centered_coords1**2), 0.0)
    is_coords2_degenerate = np.isclose(np.sum(centered_coords2**2), 0.0)

    if is_coords1_degenerate and is_coords2_degenerate:
        return 0.0 # Both sets of points are effectively identical (or collapsed to a single point)
    elif is_coords1_degenerate or is_coords2_degenerate:
        # One set is degenerate (e.g., all its atoms are at the same coordinate),
        # while the other is not. This is a significant structural mismatch.
        return 1000.0 # High penalty

    # Find the optimal rotation (will only be called if N > 1 AND not degenerate)
    rotation, rmsd = R.align_vectors(centered_coords2, centered_coords1)

    return rmsd


# --- dtw_with_rmsd_cost function ---
def frechet_with_rmsd_cost(seq1_coords, seq2_coords):
    """
    Performs Frechet on two sequences of 3D backbone coordinates,
    using RMSD as the local cost metric between corresponding residues.

    Args:
        seq1_coords (list of np.ndarray): List where each element is a (1, 3) or (N_atoms, 3)
                                          numpy array representing the coordinates for one residue/segment.
                                          For simple C-alpha, it's (1, 3).
        seq2_coords (list of np.ndarray): Similar list for the second sequence.

    Returns:
        tuple: (dtw_cost, warping_path)
            dtw_cost (float): The total accumulated cost of the optimal warping path.
            warping_path (list): A list of (index_seq1, index_seq2) tuples representing
                                 the optimal alignment path.
    """

    n = len(seq1_coords)
    m = len(seq2_coords)

    # Initialize the cost matrix
    D = np.full((n, m), np.inf)

    # Fill D[0][0]
    D[0, 0] = calculate_rmsd_after_superposition(seq1_coords[0].reshape(-1, 3), seq2_coords[0].reshape(-1, 3))

    # Fill first column
    for i in range(1, n):
        cost = calculate_rmsd_after_superposition(seq1_coords[i].reshape(-1, 3), seq2_coords[0].reshape(-1, 3))
        D[i, 0] = max(D[i - 1, 0], cost)

    # Fill first row
    for j in range(1, m):
        cost = calculate_rmsd_after_superposition(seq1_coords[0].reshape(-1, 3), seq2_coords[j].reshape(-1, 3))
        D[0, j] = max(D[0, j - 1], cost)

    # Fill the rest of the matrix
    for i in range(1, n):
        for j in range(1, m):
            cost = calculate_rmsd_after_superposition(seq1_coords[i].reshape(-1, 3), seq2_coords[j].reshape(-1, 3))
            min_prev = min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
            D[i, j] = max(cost, min_prev)

    # Optional: Traceback to get the warping path
    path = []

    total_cost = D[n-1, m-1]

    return total_cost, path


# --- Worker function for parallel processing ---
# This function must be defined at the top level (not inside if __name__ == "__main__":) so it can be pickled and sent to other processes.
def _compare_pair(args):
    pdb_id1, pdb_id2, coords1, coords2 = args
    try:
        frechet_distance, _ = frechet_with_rmsd_cost(coords1, coords2)
        return {'PDB1': pdb_id1, 'PDB2': pdb_id2, 'Frechet_Distance': frechet_distance}
    except Exception as e:
        # Handle potential errors during comparison, return NaN or a high penalty
        print(f"Error comparing {pdb_id1} and {pdb_id2}: {e}")
        return {'PDB1': pdb_id1, 'PDB2': pdb_id2, 'Frechet_Distance': np.nan}
    
# --- Main Automation Block ---
if __name__ == "__main__":
    # Antibodies folder
    pdb_folder_path = "../antibodies/antibodies" 

    if not os.path.exists(pdb_folder_path):
        print(f"Error: PDB folder not found at '{pdb_folder_path}'. Please check the path.")
        exit()

    # Get list of all PDB files
    pdb_files = [f for f in os.listdir(pdb_folder_path) if f.endswith('.pdb')]
    pdb_files.sort() # Ensure consistent order

    print(f"Found {len(pdb_files)} PDB files in '{pdb_folder_path}'.")
    if len(pdb_files) < 2:
        print("Need at least two PDB files for comparison. Exiting.")
        exit()

    # Cache extracted coordinates to avoid re-parsing PDBs for each comparison
    # This step will still run sequentially first.
    parsed_pdb_coords = {}
    print("Step 1/2: Extracting C-alpha coordinates from all PDB files (sequential)...")
    for pdb_file_name in tqdm(pdb_files, desc="Extracting Coordinates"):
        pdb_full_path = os.path.join(pdb_folder_path, pdb_file_name)
        pdb_id = os.path.splitext(pdb_file_name)[0]
        coords = extract_ca_coordinates(pdb_full_path)
        if len(coords) > 0:
            parsed_pdb_coords[pdb_id] = coords
        else:
            print(f"Skipping {pdb_id} due to empty or problematic C-alpha extraction.")

    valid_pdb_ids = list(parsed_pdb_coords.keys())
    print(f"Successfully extracted C-alpha coordinates for {len(valid_pdb_ids)} valid PDBs.")

    # Prepare tasks for the multiprocessing pool
    tasks = []
    for i in range(len(valid_pdb_ids)):
        pdb_id1 = valid_pdb_ids[i]
        coords1 = parsed_pdb_coords[pdb_id1]
        for j in range(i + 1, len(valid_pdb_ids)): # Compare each unique pair
            pdb_id2 = valid_pdb_ids[j]
            coords2 = parsed_pdb_coords[pdb_id2]
            tasks.append((pdb_id1, pdb_id2, coords1, coords2)) # Store arguments as a tuple

    total_comparisons = len(tasks)
    print(f"Step 2/2: Starting pairwise Frechet comparisons ({total_comparisons} in total) using multiprocessing...")

    # Determine the number of CPU cores to use
    num_processes = os.cpu_count()
    if num_processes is None: # Fallback for systems that don't report CPU count
        num_processes = 4 # Default to 4 if not detected
    print(f"Using {num_processes} CPU cores for parallel processing.")

    comparison_results = []
    # Use multiprocessing Pool to distribute tasks
    with multiprocessing.Pool(processes=4) as pool:
        # imap_unordered is good for progress bars as it yields results as they complete
        for result in tqdm(pool.imap_unordered(_compare_pair, tasks), total=total_comparisons, desc="Frechet Comparisons"):
            if result is not None: # Collect results, skipping any None from errors
                comparison_results.append(result)

    # Store results in a Pandas DataFrame
    frechet_df = pd.DataFrame(comparison_results)

    # Save to CSV
    output_csv_path = "../output/pairwise_frechet_distances_parallel.csv"
    frechet_df.to_csv(output_csv_path, index=False)
    print(f"\nAll pairwise Frechet distances saved to '{output_csv_path}'")

    # Display a sample of the results
    print("\nSample of Pairwise Frechet Distances:")
    print(frechet_df.head())

    # Create and save the square distance matrix
    pdb_ids_list = sorted(list(parsed_pdb_coords.keys())) # Ensure consistent order
    distance_matrix = np.full((len(pdb_ids_list), len(pdb_ids_list)), np.nan) # Use nan for self-comparison/uncomputed

    # Create mapping from PDB ID to index
    pdb_id_to_idx = {pdb_id: i for i, pdb_id in enumerate(pdb_ids_list)}

    # Fill diagonal with 0.0 (distance to self)
    np.fill_diagonal(distance_matrix, 0.0)

    for _, row in frechet_df.iterrows():
        idx1 = pdb_id_to_idx[row['PDB1']]
        idx2 = pdb_id_to_idx[row['PDB2']]
        distance_matrix[idx1, idx2] = row['Frechet_Distance']
        distance_matrix[idx2, idx1] = row['Frechet_Distance'] # Symmetric matrix

    # Save the distance matrix
    np.save("../output/frechet_distance_matrix_parallel.npy", distance_matrix)
    print(f"Freceht distance matrix (NumPy array) saved to 'frechet_distance_matrix_parallel.npy'")
    
    print("\nParallel processing complete!")