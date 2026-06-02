import glob
import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from utils.config import Configuration, load_config


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =============================================================================
# HELPER FUNCTIONS - DATA LOADING
# =============================================================================


def _load_missing_subjects(config: Configuration) -> Dict:
    """
    Load the missing subjects mapping from JSON file.

    Args:
        config: Configuration object containing directory paths

    Returns:
        Dictionary mapping filenames to missing subject lists
    """
    missing_subjects_file = os.path.join(
        config.directories.excel_files_target_dir, "missing_subjects.json"
    )

    try:
        with open(missing_subjects_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Missing subjects file not found: {missing_subjects_file}")
        return {}


def _load_ttest_results(
    config: Configuration, subject_id: int, shared_set: bool
) -> np.ndarray:
    """
    Load t-test results for a specific subject.

    Args:
        config: Configuration object containing directory paths
        subject_id: Subject identifier number
        shared_set: Whether to use shared dataset structure

    Returns:
        Array of t-test results with shape (n_voxels, 2): [t_value, p_value]

    Raises:
        FileNotFoundError: If t-test results file doesn't exist
    """
    if shared_set:
        t_results_dir = os.path.join(config.directories.t_test_results_dir, "shared")
    else:
        t_results_dir = os.path.join(
            config.directories.t_test_results_dir, f"subj_{subject_id:02d}"
        )

    t_results_filepath = os.path.join(
        t_results_dir, f"result_subj_{subject_id:02d}.npy"
    )

    if not os.path.exists(t_results_filepath):
        raise FileNotFoundError(f"T-test results file not found: {t_results_filepath}")

    return np.load(t_results_filepath)


def _load_subject_excel_data(
    config: Configuration, subject_id: int, shared_set: bool, dataset_type: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load Excel data for a specific subject and dataset type.

    Args:
        config: Configuration object containing directory paths
        subject_id: Subject identifier number
        shared_set: Whether to use shared dataset structure
        dataset_type: Either 'face' or 'non_face'

    Returns:
        Tuple of (DataFrame, list of processed filenames)
    """
    if dataset_type == "face":
        config_attr = config.dataset_creation.subset_animate_face_final
    else:  # non_face
        config_attr = config.dataset_creation.subset_animate_non_face_final

    if shared_set:
        excel_path = os.path.join(
            config.directories.excel_files_target_dir,
            "shared",
            config_attr,
        )
    else:
        excel_path = os.path.join(
            config.directories.excel_files_target_dir,
            f"subj_{subject_id:02d}",
            config_attr,
        )

    excel_data = pd.read_excel(excel_path)
    processed_filenames = [
        e.split("/")[1].split(".")[0] for e in excel_data["file_name"].tolist()
    ]

    return excel_data, processed_filenames


def _load_beta_arrays_for_file(
    config: Configuration,
    file_name: str,
    subject_id: int,
    n_samples: int,
    rng: random.Random,
    missing_subjects: Dict,
    is_shared_file: bool = False,
) -> List[np.ndarray]:
    """
    Load beta arrays for a specific file and subject.

    Args:
        config: Configuration object containing directory paths
        file_name: Name of the file to load
        subject_id: Subject identifier number
        n_samples: Number of samples to select
        rng: Random number generator for shuffling
        missing_subjects: Dictionary of missing subjects
        is_shared_file: Whether this file is from shared dataset

    Returns:
        List of beta arrays, up to n_samples length
    """
    # Check if subject is missing for this file
    if file_name in missing_subjects:
        if str(subject_id) in missing_subjects[file_name]:
            logging.info(
                f"Skipping {file_name} - missing in betas data for subject {subject_id}"
            )
            return []

    # Construct file path
    path = os.path.join(config.directories.image_betas_dir, file_name)
    subj_path = os.path.join(path, f"subj_{subject_id:02d}")
    files = glob.glob(os.path.join(subj_path, "*.npy"))

    # Load and validate arrays
    arrays = []
    for file in files:
        array = np.load(file)
        is_nan = np.isnan(np.sum(array))

        if is_nan:
            if is_shared_file:
                # Expected for shared files, skip
                continue
            else:
                raise ValueError(f"NaN values found in non-shared file: {file}")

        arrays.append(array)

    # Check if we have enough samples
    if len(arrays) < n_samples:
        if is_shared_file:
            # Expected for shared files, return empty
            return []
        else:
            logging.warning(
                f"Insufficient samples for {file_name}: {len(arrays)} < {n_samples}"
            )
            return []

    # Randomly select n_samples
    rng.shuffle(arrays)
    return arrays[:n_samples]


# =============================================================================
# HELPER FUNCTIONS - OUTLIER DETECTION
# =============================================================================


def _filter_voxels_by_threshold(
    ttest_results: np.ndarray, t_threshold: float
) -> np.ndarray:
    """
    Filter voxels based on t-value threshold.

    Args:
        ttest_results: Array of t-test results (n_voxels, 2)
        t_threshold: Minimum t-value threshold

    Returns:
        Array of voxel indices that pass the threshold
    """
    return np.where(ttest_results[:, 0] > t_threshold)[0]


def _compute_outlier_scores_for_voxel(
    pos_values: np.ndarray, neg_values: np.ndarray, voxel_index: int
) -> List[List]:
    """
    Compute outlier scores for all samples in a specific voxel.

    Args:
        pos_values: Positive group values for this voxel
        neg_values: Negative group values for this voxel
        voxel_index: Index of the current voxel

    Returns:
        List of outlier detection results for all samples
    """
    results = []

    # Compute centroids for both groups
    pos_centroid = np.mean(pos_values)
    neg_centroid = np.mean(neg_values)

    # Process positive group samples
    for sample_idx, value in enumerate(pos_values):
        own_distance = abs(value - pos_centroid)
        other_distance = abs(value - neg_centroid)
        score = other_distance - own_distance
        results.append(
            [voxel_index, "positive", sample_idx, own_distance, other_distance, score]
        )

    # Process negative group samples
    for sample_idx, value in enumerate(neg_values):
        own_distance = abs(value - neg_centroid)
        other_distance = abs(value - pos_centroid)
        score = other_distance - own_distance
        results.append(
            [voxel_index, "negative", sample_idx, own_distance, other_distance, score]
        )

    return results


def _save_outlier_results(
    config: Configuration, subject_id: int, shared_set: bool, results: np.ndarray
) -> str:
    """
    Save outlier detection results to file.

    Args:
        config: Configuration object containing directory paths
        subject_id: Subject identifier number
        shared_set: Whether using shared dataset structure
        results: Array of outlier detection results

    Returns:
        Path to saved results file
    """
    if shared_set:
        results_dir = os.path.join(config.directories.t_test_results_dir, "shared")
    else:
        results_dir = os.path.join(
            config.directories.t_test_results_dir, f"subj_{subject_id:02d}"
        )

    os.makedirs(results_dir, exist_ok=True)

    results_filepath = os.path.join(
        results_dir, f"voxel_outlier_result_subj_{subject_id:02d}.npy"
    )

    np.save(results_filepath, results)
    return results_filepath


# =============================================================================
# HELPER FUNCTIONS - T-TESTING
# =============================================================================


def _load_dataset_filenames(
    config: Configuration, subject_id: int, augment_shared_set: bool
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load and prepare dataset filenames for t-testing.

    Args:
        config: Configuration object
        subject_id: Subject identifier number
        augment_shared_set: Whether to augment with shared dataset

    Returns:
        Tuple of (positive_filenames, negative_filenames, shared_filenames)
    """
    # Load shared positive set
    shared_excel, shared_filenames = _load_subject_excel_data(
        config, subject_id, shared_set=True, dataset_type="face"
    )

    # Load subject-specific positive set
    positive_excel, positive_filenames = _load_subject_excel_data(
        config, subject_id, shared_set=False, dataset_type="face"
    )

    # Combine positive sets if augmenting
    if augment_shared_set:
        positive_filenames = positive_filenames + shared_filenames

    # Load negative set and filter out animate faces
    negative_excel, _ = _load_subject_excel_data(
        config, subject_id, shared_set=False, dataset_type="non_face"
    )

    # Filter out animate_faces from negative set
    filtered_negative = negative_excel[negative_excel["label"] != "animate_faces"]
    negative_filenames = [
        e.split("/")[1].split(".")[0] for e in filtered_negative["file_name"].tolist()
    ]

    logging.info(
        f"Removed {len(negative_excel) - len(filtered_negative)} animate_faces samples"
    )

    return positive_filenames, negative_filenames, shared_filenames


def _load_beta_data_for_group(
    config: Configuration,
    filenames: List[str],
    subject_id: int,
    n_samples: int,
    rng: random.Random,
    missing_subjects: Dict,
    shared_filenames: List[str],
) -> np.ndarray:
    """
    Load beta data for a group of files.

    Args:
        config: Configuration object
        filenames: List of filenames to process
        subject_id: Subject identifier number
        n_samples: Number of samples per file
        rng: Random number generator
        missing_subjects: Dictionary of missing subjects
        shared_filenames: List of shared filenames for identification

    Returns:
        Array of beta data with shape (n_total_samples, n_voxels)
    """
    group_data = []

    for filename in filenames:
        is_shared = filename in shared_filenames
        arrays = _load_beta_arrays_for_file(
            config, filename, subject_id, n_samples, rng, missing_subjects, is_shared
        )

        # Add all samples from this file
        group_data.extend(arrays)

    if not group_data:
        raise ValueError("No valid data loaded for group")

    return np.array(group_data)


def _compute_voxelwise_ttest(
    positive_data: np.ndarray, negative_data: np.ndarray
) -> np.ndarray:
    """
    Compute voxelwise t-tests between positive and negative groups.

    Args:
        positive_data: Positive group data (n_pos_samples, n_voxels)
        negative_data: Negative group data (n_neg_samples, n_voxels)

    Returns:
        Array of t-test results (n_voxels, 2): [t_statistic, p_value]
    """
    n_voxels = positive_data.shape[1]
    results = []

    for i in tqdm(range(n_voxels), desc="Computing t-tests"):
        positive_voxel = positive_data[:, i]
        negative_voxel = negative_data[:, i]

        t_statistic, p_value = stats.ttest_ind(
            positive_voxel,
            negative_voxel,
            equal_var=False,
        )

        results.append([t_statistic, p_value])

    return np.array(results)


def _save_ttest_results(
    config: Configuration, subject_id: int, shared_set: bool, results: np.ndarray
) -> str:
    """
    Save t-test results to file.

    Args:
        config: Configuration object
        subject_id: Subject identifier number
        shared_set: Whether using shared dataset structure
        results: Array of t-test results

    Returns:
        Path to saved results file
    """
    if shared_set:
        results_dir = os.path.join(config.directories.t_test_results_dir, "shared")
    else:
        results_dir = os.path.join(
            config.directories.t_test_results_dir, f"subj_{subject_id:02d}"
        )

    os.makedirs(results_dir, exist_ok=True)

    results_filepath = os.path.join(results_dir, f"result_subj_{subject_id:02d}.npy")
    np.save(results_filepath, results)

    return results_filepath


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================


def set_voxelwise_outlier_detection(
    config: Configuration,
    subjs_to_use: List[int],
    shared_set: bool,
    t_threshold: float = 2.0,
) -> None:
    """
    Perform voxelwise outlier detection for specified subjects.

    This function:
    1. Loads previously computed voxelwise t-test results
    2. Selects voxels with t-values above threshold
    3. Loads positive and negative beta data
    4. Computes group centroids for each selected voxel
    5. Calculates outlier scores for each sample
    6. Saves results as [voxel_index, group, sample_index, own_distance, other_distance, score]

    Args:
        config: Configuration object containing all necessary paths
        subjs_to_use: List of subject IDs to process
        shared_set: Whether to use shared dataset structure
        t_threshold: T-value threshold for voxel selection (default: 2.0)
    """
    # Load missing subjects information
    missing_subjects = _load_missing_subjects(config) if shared_set else {}

    # Process each subject
    for subject_id in tqdm(subjs_to_use, desc="Processing subjects"):
        logging.info(
            f"Performing voxelwise outlier detection for subject {subject_id:02d}"
        )

        try:
            # Load t-test results
            ttest_results = _load_ttest_results(config, subject_id, shared_set)

            # Filter voxels by threshold
            selected_voxels = _filter_voxels_by_threshold(ttest_results, t_threshold)
            logging.info(
                f"Subject {subject_id:02d}: {len(selected_voxels)} voxels pass t > {t_threshold}"
            )

            # Load dataset information
            _, pos_filenames = _load_subject_excel_data(
                config, subject_id, shared_set, "face"
            )
            _, neg_filenames = _load_subject_excel_data(
                config, subject_id, shared_set, "non_face"
            )

            # Load beta data for positive group
            pos_data = []
            for filename in pos_filenames:
                arrays = _load_beta_arrays_for_file(
                    config,
                    filename,
                    subject_id,
                    n_samples=1,
                    rng=random.Random(0),
                    missing_subjects=missing_subjects,
                )
                pos_data.extend(arrays)
            pos_data = np.array(pos_data) if pos_data else np.array([]).reshape(0, -1)

            # Load beta data for negative group
            neg_data = []
            for filename in neg_filenames:
                arrays = _load_beta_arrays_for_file(
                    config,
                    filename,
                    subject_id,
                    n_samples=1,
                    rng=random.Random(0),
                    missing_subjects=missing_subjects,
                )
                neg_data.extend(arrays)
            neg_data = np.array(neg_data) if neg_data else np.array([]).reshape(0, -1)

            # Perform outlier detection for each selected voxel
            all_outlier_results = []
            for voxel_idx in tqdm(selected_voxels, desc="Processing voxels"):
                pos_voxel_values = pos_data[:, voxel_idx]
                neg_voxel_values = neg_data[:, voxel_idx]

                voxel_results = _compute_outlier_scores_for_voxel(
                    pos_voxel_values, neg_voxel_values, voxel_idx
                )
                all_outlier_results.extend(voxel_results)

            # Save results
            results_array = np.array(all_outlier_results)
            results_path = _save_outlier_results(
                config, subject_id, shared_set, results_array
            )
            logging.info(f"Saved outlier results to {results_path}")

        except Exception as e:
            logging.error(f"Error processing subject {subject_id:02d}: {str(e)}")
            continue


def set_t_testing(
    config: Configuration,
    subjs_to_use: List[int],
    shared_set: bool,
    augment_shared_set: bool,
    n_samples_per_file: int = 2,
    random_seed: int = 0,
) -> None:
    """
    Perform t-testing analysis for specified subjects.

    This function:
    1. Loads positive (face) and negative (non-face) datasets
    2. Samples multiple instances per image for robust statistics
    3. Performs voxelwise t-tests between groups
    4. Saves results for further analysis

    Args:
        config: Configuration object containing all necessary paths
        subjs_to_use: List of subject IDs to process
        shared_set: Whether to use shared dataset structure (currently disabled)
        augment_shared_set: Whether to augment subject data with shared dataset
        n_samples_per_file: Number of samples to use per image file (default: 2)
        random_seed: Random seed for reproducible sampling (default: 0)

    Raises:
        Exception: If shared_set is True (not currently supported)
    """
    if shared_set:
        raise Exception("Shared set processing not implemented in current version")

    # Initialize random number generator
    rng = random.Random(random_seed)

    # Load missing subjects information
    missing_subjects = _load_missing_subjects(config)

    # Process each subject
    for subject_id in tqdm(subjs_to_use, desc="Processing subjects"):
        logging.info(f"Generating t-test data for subject {subject_id:02d}")
        logging.info(f"Augmenting with shared set: {augment_shared_set}")

        try:
            # Load dataset filenames
            pos_filenames, neg_filenames, shared_filenames = _load_dataset_filenames(
                config, subject_id, augment_shared_set
            )

            logging.info(f"Positive files: {len(pos_filenames)}")
            logging.info(f"Negative files: {len(neg_filenames)}")

            # Load beta data for both groups
            positive_data = _load_beta_data_for_group(
                config,
                pos_filenames,
                subject_id,
                n_samples_per_file,
                rng,
                missing_subjects,
                shared_filenames,
            )

            negative_data = _load_beta_data_for_group(
                config,
                neg_filenames,
                subject_id,
                n_samples_per_file,
                rng,
                missing_subjects,
                shared_filenames,
            )

            logging.info(f"Positive data shape: {positive_data.shape}")
            logging.info(f"Negative data shape: {negative_data.shape}")

            # Compute voxelwise t-tests
            logging.info("Computing voxelwise statistics...")
            ttest_results = _compute_voxelwise_ttest(positive_data, negative_data)

            # Save results
            results_path = _save_ttest_results(
                config, subject_id, shared_set, ttest_results
            )
            logging.info(f"Saved t-test results to {results_path}")

        except Exception as e:
            logging.error(f"Error processing subject {subject_id:02d}: {str(e)}")
            continue
