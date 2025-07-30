"""
Permutation Analysis Module

This module implements Mantel tests for analyzing the relationship between 
neural representations and various facial features (centers, sizes, ages, genders).
The analysis includes both RDM-based and MDS-based comparisons using permutation testing.
"""

# Standard library imports
import os
import json
import logging

# Third-party imports
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from tqdm import tqdm

# Local imports
from utils.utils import retrieve_roi_mask
from utils.config import load_config
from rsa.create_rdm import create_rdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# CORE MANTEL TEST FUNCTIONS
# =========================================================================

def mantel_test_feature(
    X: np.ndarray, 
    feature: np.ndarray, 
    B: int = 2000, 
    metric: str = "euclidean"
) -> dict:
    """
    Perform Mantel test between feature space X and a specific feature vector.
    
    This function computes the correlation between distance matrices derived from
    the feature space X and the feature vector, then tests significance using
    permutation testing.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        feature: Feature vector of shape (n_samples,) or (n_samples, n_features)
        B: Number of bootstrap/permutation iterations for significance testing
        metric: Distance metric to use (default: "euclidean")
        
    Returns:
        dict: Results containing observed correlation, p-value, and permutation correlations
    """
    X = np.asarray(X)
    feature = np.asarray(feature)

    assert X.shape[0] == feature.shape[0], "X and feature must have same number of samples"

    # Ensure feature is 2D
    if feature.ndim == 1:
        feature_vals = feature.reshape(-1, 1)
    else:
        feature_vals = feature

    n = X.shape[0]

    # Compute distance matrices
    D_X = squareform(pdist(X, metric=metric))
    D_F = squareform(pdist(feature_vals, metric=metric))

    # Extract upper triangular indices (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    d_vec = D_X[iu]
    f_vec = D_F[iu]

    # Compute observed correlation
    r_obs, _ = spearmanr(d_vec, f_vec)

    # Permutation testing
    r_perm = np.empty(B)
    for b in range(B):
        perm = np.random.permutation(n)
        D_F_perm = D_F[perm][:, perm]
        f_p = D_F_perm[iu]
        r_perm[b], _ = spearmanr(d_vec, f_p)

    # Calculate p-value using permutation distribution
    p_value = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1) / (B + 1)

    return {"r_obs": r_obs, "p_value": p_value, "r_perm": r_perm}


def mantel_test_feature_rdm(
    D_X: np.ndarray, 
    feature: np.ndarray, 
    B: int = 2000, 
    metric: str = "euclidean"
) -> dict:
    """
    Perform Mantel test using a precomputed distance matrix (RDM) and feature vector.
    
    This function is similar to mantel_test_feature but takes a precomputed
    distance matrix as input instead of computing it from features.
    
    Args:
        D_X: Precomputed distance matrix (condensed format from pdist)
        feature: Feature vector of shape (n_samples,) or (n_samples, n_features)  
        B: Number of bootstrap/permutation iterations for significance testing
        metric: Distance metric to use for feature distances (default: "euclidean")
        
    Returns:
        dict: Results containing observed correlation, p-value, and permutation correlations
    """
    feature = np.asarray(feature)
    D_X = squareform(D_X)  # Convert to square matrix format
    
    assert (
        D_X.shape[0] == feature.shape[0] and D_X.shape[1] == D_X.shape[0]
    ), "D_X and feature must have compatible dimensions; D_X should be square"

    # Ensure feature is 2D
    if feature.ndim == 1:
        feature_vals = feature.reshape(-1, 1)
    else:
        feature_vals = feature

    n = D_X.shape[0]

    # Compute feature distance matrix
    D_F = squareform(pdist(feature_vals, metric=metric))

    # Extract upper triangular indices (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    d_vec = D_X[iu]
    f_vec = D_F[iu]

    # Compute observed correlation
    r_obs, _ = spearmanr(d_vec, f_vec)

    # Permutation testing
    r_perm = np.empty(B)
    for b in range(B):
        perm = np.random.permutation(n)
        D_F_perm = D_F[perm][:, perm]
        f_p = D_F_perm[iu]
        r_perm[b], _ = spearmanr(d_vec, f_p)

    # Calculate p-value using permutation distribution
    p_value = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1) / (B + 1)

    return {"r_obs": r_obs, "p_value": p_value, "r_perm": r_perm}


# =========================================================================
# FEATURE EXTRACTION AND PROCESSING
# =========================================================================

def extract_facial_features(metadata: list, subject_i: int) -> dict:
    """
    Extract facial features from detection results for a given subject.
    
    Args:
        metadata: List of image identifiers
        subject_i: Subject ID
        
    Returns:
        dict: Dictionary containing extracted features (centers, sizes, ages, genders, eye_coordinates)
    """
    # Load face detection results
    face_det_path = f"data/labels/subj_{subject_i:02d}/face_detection_result.json"
    face_det_path_shared = f"data/labels/shared/face_detection_result.json"

    with open(face_det_path, "r") as f:
        face_detections = json.load(f)

    with open(face_det_path_shared, "r") as f:
        face_detections_shared = json.load(f)

    # Combine detections from both sources
    face_detections += face_detections_shared

    # Filter detections to match metadata
    filtered = [x for x in face_detections if x["file_name"][:-4] in metadata]
    for x in filtered:
        x["cocoId"] = x["file_name"][:-4]

    # Initialize feature lists
    centers = []
    eye_coordinates = []
    feature_array_age = []
    sizes = []
    genders = []

    # Extract features for each image in metadata
    for entry in metadata:
        # Find matching detection record
        matches = [x for x in filtered if x["cocoId"] == entry]
        assert len(matches) == 1, f"Expected exactly one match for {entry}, got {len(matches)}"
        match = matches[0]
        assert len(match["detection"]) == 1, "Expected exactly one bounding box per image"

        # Extract bounding box center and size
        x1, y1, x2, y2 = match["detection"][0]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        centers.append(center)
        
        width = x2 - x1
        height = y2 - y1
        sizes.append(np.array([width, height]))

        # Extract demographic features
        feature_array_age.append(match["age"])
        genders.append(match["gender"])

        # Extract eye landmark coordinates
        lm = match["landmarks"][0]
        left_eye = np.array(lm[38])
        right_eye = np.array(lm[88])
        eye_coordinates.append([left_eye, right_eye])

    return {
        "centers": np.array(centers),
        "sizes": np.array(sizes),
        "eye_coordinates": np.array(eye_coordinates),
        "ages": np.array(feature_array_age),
        "genders": np.array(genders)
    }


def run_mantel_test(
    D_X: np.ndarray,
    X: np.ndarray, 
    subject_i: int, 
    metadata: list, 
    B: int = 2000, 
    feature_selection: str = "centers"
) -> dict:
    """
    Run Mantel test for a specific feature selection on given data.
    
    Args:
        D_X: Precomputed distance matrix (condensed format)
        X: Feature matrix for MDS-based analysis
        subject_i: Subject ID
        metadata: List of image identifiers
        B: Number of permutation iterations
        feature_selection: Feature type to analyze ("centers", "sizes", "ages", "genders", "eye")
        
    Returns:
        dict: Results for both RDM and MDS analyses
    """
    logging.info(f"Running Mantel test for subject {subject_i}, feature: {feature_selection}")
    
    # Extract all facial features
    features_dict = extract_facial_features(metadata, subject_i)
    
    # Select the specified feature
    if feature_selection == "eye":
        features = features_dict["eye_coordinates"]
    else:
        features = features_dict[feature_selection]

    # Run Mantel tests for both RDM and MDS
    result_RDM = mantel_test_feature_rdm(D_X, features, B=B)
    result_MDS = mantel_test_feature(X, features, B=B)

    return {
        "MDS": {
            "subject": subject_i,
            "r_obs": result_MDS["r_obs"],
            "p_value": result_MDS["p_value"],
            "feature": feature_selection,
        },
        "RDM": {
            "subject": subject_i,
            "r_obs": result_RDM["r_obs"],
            "p_value": result_RDM["p_value"],
            "feature": feature_selection,
        },
    }


# =========================================================================
# HIGHER-LEVEL ANALYSIS FUNCTIONS
# =========================================================================

def run_mantel_test_for_subject_mask(
    subject_i: int, 
    mask: int, 
    set_to_take_rsa: str, 
    t_test_threshold: float, 
    B: int = 2000, 
    feature_selection: str = "centers", 
    config=None
) -> list:
    """
    Load data and run Mantel test for a specific subject and ROI mask.
    
    Args:
        subject_i: Subject ID
        mask: ROI mask value
        set_to_take_rsa: Dataset identifier for RSA analysis
        t_test_threshold: Threshold for t-test based masking
        B: Number of permutation iterations
        feature_selection: Feature type to analyze
        config: Configuration object
        
    Returns:
        list: Results for both RDM and MDS analyses
    """
    logging.info(f"Processing subject {subject_i}, mask {mask}, feature: {feature_selection}")
    
    # Generate RDM if needed
    rand = 0
    create_rdm(config, [subject_i], mask, f"subj_{subject_i:02d}", t_test_threshold, randomization=True)

    # Load precomputed MDS coordinates
    X = np.load(
        f"data/mds/{set_to_take_rsa}/"
        f"subj_{subject_i:02d}/subj_{subject_i:02d}/"
        f"mask_{mask}_averaged_both_rand_{rand}_mds.npy"
    )

    # Load precomputed RDM
    D_X = np.load(
        f"data/rdm/{set_to_take_rsa}/"
        f"subj_{subject_i:02d}/subj_{subject_i:02d}/"
        f"mask_{mask}_averaged_both_rand_{rand}_rdm.npy"
    )

    # Load metadata
    metadata = np.load(
        f"data/rdm/{set_to_take_rsa}/" 
        f"subj_{subject_i:02d}/subj_{subject_i:02d}/" 
        "metadata.npy"
    )

    # Run Mantel test
    res = run_mantel_test(
        D_X, X, subject_i, metadata, B=B, feature_selection=feature_selection
    )

    # Format results with mask information
    return [
        {
            "type": "MDS",
            "subject": res["MDS"]["subject"],
            "mask": mask,
            "r_obs": res["MDS"]["r_obs"],
            "p_value": res["MDS"]["p_value"],
            "feature": res["MDS"]["feature"],
        },
        {
            "type": "RDM",
            "subject": res["RDM"]["subject"],
            "mask": mask,
            "r_obs": res["RDM"]["r_obs"],
            "p_value": res["RDM"]["p_value"],
            "feature": res["RDM"]["feature"],
        },
    ]


def generate_results():
    """
    Generate comprehensive Mantel test results for all subjects, masks, and features.
    
    This function processes all subjects (1-8), extracts ROI masks, and runs
    Mantel tests for all feature types, saving results to an Excel file.
    """
    logging.info("Starting comprehensive Mantel test analysis")
    
    config = load_config("config.yaml")
    results_path = os.path.join(
        config.directories.output_dir, 
        config.saved_results_paths.permutation_analysis_excel
    )

    results = []
    feature_selections = ["centers", "sizes", "ages", "genders"]

    # Process each subject
    for subj in range(1, 9):
        logging.info(f"Processing subject {subj}")
        
        # Retrieve ROI mask for this subject
        subj_mask = retrieve_roi_mask(config, subj, f"subj_{subj:02d}", False, 3.0)
        mask_values = np.unique(subj_mask).tolist()
        mask_values.remove(0)  # Remove background mask value

        # Process each mask value
        for mask in tqdm(mask_values, desc=f"Subject {subj} masks"):
            # Process each feature type
            for feature_selection in feature_selections:
                try:
                    r = run_mantel_test_for_subject_mask(
                        subj, 
                        mask, 
                        f"subj_{subj:02d}",  # set_to_take_rsa
                        3.0,  # t_test_threshold
                        B=2000, 
                        feature_selection=feature_selection,
                        config=config
                    )
                    results += r
                except Exception as e:
                    logging.error(f"Error processing subject {subj}, mask {mask}, feature {feature_selection}: {e}")
                    continue

        # Save intermediate results after each subject
        df_results = pd.DataFrame(results)
        df_results.to_excel(results_path, index=False)
        logging.info(f"Saved intermediate results after subject {subj}")

    # Save final results
    df_results = pd.DataFrame(results)
    df_results.to_excel(results_path, index=False)
    logging.info(f"Final results saved to {results_path}")
    logging.info(f"Total results generated: {len(results)}")


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    """Main execution function."""
    logging.info("Starting permutation analysis pipeline")
    generate_results()
    logging.info("Permutation analysis pipeline complete")


if __name__ == "__main__":
    main()