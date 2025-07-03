import numpy as np
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from Bio.PDB import PDBParser # Biopython parser for PDB files

# --- Core Utility Functions from User's Input ---

def get_ca_coords_with_residues(pdb_path):
    """
    Parses a PDB file to extract the 3D coordinates of C-alpha atoms
    and their corresponding residue IDs.

    Args:
        pdb_path (str): The full path to the PDB file.

    Returns:
        tuple: A tuple containing:
               - np.array: A 2D numpy array where each row is the [x, y, z]
                           coordinate of a C-alpha atom.
               - list: A list of residue IDs (integers) corresponding to the coordinates.
               Returns (empty_np_array, empty_list) if parsing fails or no CA atoms found.
    """
    parser = PDBParser(QUIET=True) # QUIET=True suppresses warnings
    try:
        structure = parser.get_structure('structure', pdb_path)
    except Exception as e:
        print(f"Error parsing PDB {pdb_path}: {e}")
        return np.array([]), []

    coords = []
    residue_ids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue: # Check if the C-alpha atom exists for the residue
                    coords.append(residue['CA'].get_coord())
                    # residue.get_id()[1] typically gets the residue sequence number
                    residue_ids.append(residue.get_id()[1])
    
    return np.array(coords), residue_ids

def c_rmsd(P, Q):
    """
    Computes the c-RMSD (coordinate RMSD) between two sets of 3D points P and Q
    after optimal rigid-body superposition using the Kabsch algorithm.
    This is the user's custom implementation of the Kabsch algorithm.

    Args:
        P (np.array): A 2D numpy array of coordinates for the first set of points (Nx3).
        Q (np.array): A 2D numpy array of coordinates for the second set of points (Nx3).
                      P and Q must have the same number of points (N).

    Returns:
        float: The RMSD value after optimal superposition.
    """
    # Ensure inputs are numpy arrays
    P = np.asarray(P)
    Q = np.asarray(Q)

    # Check for empty inputs after potential conversion (should be handled upstream too)
    if P.size == 0 or Q.size == 0:
        return 1000.0 # Return a high penalty if no points to compare

    # Ensure same number of points for c-RMSD
    if P.shape != Q.shape:
        raise ValueError("Input point sets P and Q must have the same shape for c-RMSD calculation.")

    # Compute the centroids
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)

    # Center the point clouds
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    # Compute covariance matrix H
    H = np.dot(P_centered.T, Q_centered)

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Ensure a right-handed coordinate system (determinant = +1)
    # This ensures that the rotation matrix does not include a reflection.
    d = np.linalg.det(np.dot(Vt.T, U.T))
    D = np.diag([1, 1, np.sign(d)]) # Use sign(d) for the last diagonal element

    # Compute optimal rotation matrix R
    R_matrix = np.dot(Vt.T, np.dot(D, U.T))

    # Rotate P and compute RMSD
    P_rot = np.dot(P_centered, R_matrix)
    rmsd = np.sqrt(np.mean(np.sum((P_rot - Q_centered) ** 2, axis=1)))
    
    return rmsd

def calculate_matched_residue_rmsd(coords1, residue_ids1, coords2, residue_ids2):
    """
    Calculates the RMSD between two protein structures by first finding common
    residues (based on residue IDs) and then applying the c_rmsd function
    on the corresponding C-alpha coordinates.

    Args:
        coords1 (np.array): C-alpha coordinates for the first structure.
        residue_ids1 (list): Residue IDs for the first structure.
        coords2 (np.array): C-alpha coordinates for the second structure.
        residue_ids2 (list): Residue IDs for the second structure.

    Returns:
        float: The c-RMSD value for the matched residues. Returns 1000.0
               if no common residues are found or an error occurs.
    """
    # Create dictionaries for quick lookup of coordinates by residue ID
    X_dict = dict(zip(residue_ids1, coords1))
    Y_dict = dict(zip(residue_ids2, coords2))

    # Find common residue IDs
    common_residues = sorted(list(set(residue_ids1) & set(residue_ids2)))

    if not common_residues:
        # No common residues, cannot calculate RMSD meaningfully
        return 1000.0 

    # Extract coordinates for matched residues, ensuring order
    X_matched = np.array([X_dict[r] for r in common_residues])
    Y_matched = np.array([Y_dict[r] for r in common_residues])

    # Compute and return c-RMSD
    try:
        return c_rmsd(X_matched, Y_matched)
    except Exception as e:
        print(f"Error in c_rmsd for matched residues: {e}")
        return 1000.0 # High penalty on error


# --- Worker function for parallel processing ---
# This function must be defined at the top level so it can be pickled and sent to other processes.
def _compare_pair_c_rmsd(args):
    pdb_id1, pdb_id2, coords1, residue_ids1, coords2, residue_ids2 = args
    
    c_rmsd_val = np.nan
    try:
        c_rmsd_val = calculate_matched_residue_rmsd(coords1, residue_ids1, coords2, residue_ids2)
    except Exception as e:
        print(f"Error calculating c-RMSD for {pdb_id1} and {pdb_id2}: {e}")
        
    return {'PDB1': pdb_id1, 'PDB2': pdb_id2, 'c_RMSD': c_rmsd_val}

# --- Main Automation Block ---
if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your PDB files
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

    # Cache extracted coordinates and residue IDs
    parsed_pdb_data = {} # Will store {'pdb_id': (coords_np_array, residue_ids_list)}
    print("Step 1/2: Extracting C-alpha coordinates and residue IDs from all PDB files (sequential)...")
    for pdb_file_name in tqdm(pdb_files, desc="Extracting Data"):
        pdb_full_path = os.path.join(pdb_folder_path, pdb_file_name)
        pdb_id = os.path.splitext(pdb_file_name)[0]
        coords, res_ids = get_ca_coords_with_residues(pdb_full_path)
        if coords.size > 0: # Check if any coordinates were extracted
            parsed_pdb_data[pdb_id] = (coords, res_ids)
        else:
            print(f"Skipping {pdb_id} due to empty or problematic C-alpha extraction.")

    valid_pdb_ids = list(parsed_pdb_data.keys())
    print(f"Successfully extracted data for {len(valid_pdb_ids)} valid PDBs.")

    # Prepare tasks for the multiprocessing pool
    tasks = []
    for i in range(len(valid_pdb_ids)):
        pdb_id1 = valid_pdb_ids[i]
        coords1, res_ids1 = parsed_pdb_data[pdb_id1]
        for j in range(i + 1, len(valid_pdb_ids)): # Compare each unique pair
            pdb_id2 = valid_pdb_ids[j]
            coords2, res_ids2 = parsed_pdb_data[pdb_id2]
            tasks.append((pdb_id1, pdb_id2, coords1, res_ids1, coords2, res_ids2)) # Store arguments as a tuple

    total_comparisons = len(tasks)
    print(f"Step 2/2: Starting pairwise c-RMSD comparisons ({total_comparisons} in total) using multiprocessing...")

    # Determine the number of CPU cores to use
    num_processes = os.cpu_count()
    if num_processes is None:
        num_processes = 4
    print(f"Using {num_processes} CPU cores for parallel processing.")

    comparison_results = []
    # Use multiprocessing Pool to distribute tasks
    with multiprocessing.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(_compare_pair_c_rmsd, tasks), total=total_comparisons, desc="c-RMSD Comparisons"):
            if result is not None:
                comparison_results.append(result)

    # Store results in a Pandas DataFrame
    comparison_df = pd.DataFrame(comparison_results)

    # Save to CSV
    output_csv_path = "../outputs/pairwise_c_rmsd_distances_parallel.csv"
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"\nAll pairwise c-RMSD distances saved to '{output_csv_path}'")

    # Display a sample of the results
    print("\nSample of Pairwise c-RMSD Distances:")
    print(comparison_df.head())

    print("\nCreating and saving square distance matrix for c-RMSD...")
    pdb_ids_list = sorted(list(parsed_pdb_data.keys())) # Ensure consistent order
    
    # Initialize c-RMSD distance matrix
    c_rmsd_matrix = np.full((len(pdb_ids_list), len(pdb_ids_list)), np.nan)
    np.fill_diagonal(c_rmsd_matrix, 0.0) # Distance to self is 0

    # Create mapping from PDB ID to index
    pdb_id_to_idx = {pdb_id: i for i, pdb_id in enumerate(pdb_ids_list)}

    for _, row in comparison_df.iterrows():
        idx1 = pdb_id_to_idx[row['PDB1']]
        idx2 = pdb_id_to_idx[row['PDB2']]
        
        c_rmsd_matrix[idx1, idx2] = row['c_RMSD']
        c_rmsd_matrix[idx2, idx1] = row['c_RMSD'] # Symmetric

    # Save the distance matrix
    np.save("../outputs/c_rmsd_distance_matrix_parallel.npy", c_rmsd_matrix)
    print(f"c-RMSD distance matrix (NumPy array) saved to '../outputs/c_rmsd_distance_matrix_parallel.npy'")

    print("\nParallel processing complete!")
