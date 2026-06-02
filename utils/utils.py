"""
Utility Functions Module

This module provides core utility functions for data loading, ROI mask processing,
and beta coefficient retrieval for the NSD analysis pipeline. It includes functions
for subject list processing, ROI mask loading, and comprehensive beta data loading
with support for train/test splits and data augmentation.
"""

# Standard library imports
import glob
import os
import logging

# Third-party imports
import pandas as pd
import numpy as np
import nibabel as nib

# Local imports
from utils.config import Configuration
from t_testing.clean_roi_mask import modify_mask_with_ttest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# LOGGING AND UTILITY FUNCTIONS
# =========================================================================


def logging_message(step: int, message: str) -> str:
    """
    Create standardized logging message for pipeline steps.

    Args:
        step: Pipeline step number
        message: Message content

    Returns:
        str: Formatted log message
    """
    return f"Pipeline (S={step}): {message}"


def subjects_list_unifier(subjects_list: list, reduce_shared: bool) -> list:
    """
    Unify and validate subject list specification.

    Handles the special "shared" keyword by expanding it to all subjects (1-8)
    when reduce_shared is True, or keeping it as-is when False.

    Args:
        subjects_list: List containing subject IDs and/or "shared"
        reduce_shared: If True, expand "shared" to subjects 1-8

    Returns:
        list: Processed subject list

    Raises:
        AssertionError: If invalid subjects are specified
    """
    # Validate subject list contains only allowed values
    valid_subjects = {"shared", 1, 2, 3, 4, 5, 6, 7, 8}
    invalid_subjects = set(subjects_list) - valid_subjects
    assert not invalid_subjects, f"Invalid subjects specification: {subjects_list}"

    if reduce_shared:
        if "shared" in subjects_list:
            return list(range(1, 8 + 1))
        else:
            return subjects_list
    else:
        return subjects_list


# =========================================================================
# ROI MASK PROCESSING FUNCTIONS
# =========================================================================


def retrieve_roi_mask(
    config: Configuration,
    subject: int,
    subj_to_check: str,
    take_cleaned_roi: bool,
    t_test_threshold: float,
    sub_filename: str = "cleanedrois",
) -> np.ndarray:
    """
    Retrieve ROI mask for a specific subject with optional t-test cleaning.

    Args:
        config: Configuration object containing directory paths
        subject: Subject ID to load
        subj_to_check: Subject identifier string for file paths
        take_cleaned_roi: Whether to use t-test cleaned ROI masks
        t_test_threshold: Threshold for t-test cleaning
        sub_filename: Filename identifier for cleaned ROI files

    Returns:
        np.ndarray: Combined left and right hemisphere ROI mask

    Raises:
        Exception: If attempting to use non-cleaned ROI masks
    """
    logging.info(f"Loading ROI mask with cleaning: {take_cleaned_roi}")

    if take_cleaned_roi:
        # Apply t-test cleaning to generate mask files
        modify_mask_with_ttest(
            config, t_test_threshold, False, subject, sub_filename=sub_filename
        )

        mask_path_lh = os.path.join(
            config.directories.t_test_roi_dir,
            subj_to_check,
            f"lh.subj{subject:02d}.{sub_filename}_{t_test_threshold}.mgz",
        )
        mask_path_rh = os.path.join(
            config.directories.t_test_roi_dir,
            subj_to_check,
            f"rh.subj{subject:02d}.{sub_filename}_{t_test_threshold}.mgz",
        )
    else:
        raise Exception("Using non-cleaned ROI masks is not recommended")
        # Legacy code kept for reference but not used
        # mask_path_lh = os.path.join(
        #     config.directories.t_test_roi_dir,
        #     subj_to_check,
        #     f"lh.subj{subject:02d}.testrois.mgz",
        # )
        # mask_path_rh = os.path.join(
        #     config.directories.t_test_roi_dir,
        #     subj_to_check,
        #     f"rh.subj{subject:02d}.testrois.mgz",
        # )

    logging.info(f"Loading mask from:\n{mask_path_lh}\n{mask_path_rh}")

    # Load and combine hemisphere masks
    mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()
    mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()
    mask = np.concatenate((mask_lh, mask_rh)).astype(int)

    return mask


def retrieve_roi_mask_extended(
    config: Configuration,
    subject: int,
    subj_to_check: str,
    take_cleaned_roi: bool,
    sub_filename: str = "cleanedrois",
    t_test_threshold: float = 2.0,
) -> tuple[np.ndarray, int]:
    """
    Retrieve ROI mask with additional left hemisphere size information.

    Extended version of retrieve_roi_mask that also returns the size of the
    left hemisphere mask for proper indexing in downstream analyses.

    Args:
        config: Configuration object containing directory paths
        subject: Subject ID to load
        subj_to_check: Subject identifier string for file paths
        take_cleaned_roi: Whether to use t-test cleaned ROI masks
        sub_filename: Filename identifier for cleaned ROI files
        t_test_threshold: Threshold for t-test cleaning

    Returns:
        tuple: (combined_mask, left_hemisphere_size)
            - combined_mask: Combined left and right hemisphere ROI mask
            - left_hemisphere_size: Number of voxels in left hemisphere

    Raises:
        Exception: If attempting to use non-cleaned ROI masks
    """
    logging.info(f"Loading extended ROI mask with cleaning: {take_cleaned_roi}")

    if take_cleaned_roi:
        # Apply t-test cleaning to generate mask files
        modify_mask_with_ttest(
            config, t_test_threshold, False, subject, sub_filename=sub_filename
        )

        mask_path_lh = os.path.join(
            config.directories.t_test_roi_dir,
            subj_to_check,
            f"lh.subj{subject:02d}.{sub_filename}.mgz",
        )
        mask_path_rh = os.path.join(
            config.directories.t_test_roi_dir,
            subj_to_check,
            f"rh.subj{subject:02d}.{sub_filename}.mgz",
        )
    else:
        raise Exception("Using non-cleaned ROI masks is not recommended")
        # Legacy code kept for reference
        # mask_path_lh = os.path.join(
        #     config.directories.t_test_roi_dir,
        #     subj_to_check,
        #     f"lh.subj{subject:02d}.testrois.mgz",
        # )
        # mask_path_rh = os.path.join(
        #     config.directories.t_test_roi_dir,
        #     subj_to_check,
        #     f"rh.subj{subject:02d}.testrois.mgz",
        # )

    logging.info(f"Loading extended mask from:\n{mask_path_lh}\n{mask_path_rh}")

    # Load hemisphere masks separately to get left hemisphere size
    mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()
    mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()

    # Combine masks and return with left hemisphere size
    mask = np.concatenate((mask_lh, mask_rh)).astype(int)
    return mask, mask_lh.size


def filter_roi_mask(roi_value: int, mask: np.ndarray) -> tuple:
    """
    Filter ROI mask to find indices of voxels with a specific ROI value.

    Args:
        roi_value: ROI label value to filter for
        mask: ROI mask array

    Returns:
        tuple: Indices where mask equals roi_value
    """
    indices = np.where(mask == roi_value)
    return indices


# =========================================================================
# BETA COEFFICIENT LOADING FUNCTIONS
# =========================================================================


def retrieve_stacked_betas(
    config: Configuration,
    subj: int,
    train_mode: str,
    sample: int,
    test: bool = False,
    subj_to_check: str = "shared",
    label_subset_name: str = None,
    only_face_set: bool = True,
    randomization: bool = False,
    augment_shared_set: bool = False,
    seed_offset: int = 0,
) -> tuple[np.ndarray, list, list]:
    """
    Comprehensive beta coefficient loading function with train/test support.

    This unified function handles both training and test data loading with multiple
    modes of operation, data augmentation, and quality control measures.

    Args:
        config: Configuration object containing paths and parameters
        subj: Subject ID to load
        train_mode: Mode for training data ("single", "multiple", "averaged")
        sample: Sample index for single mode
        test: If True, loads test data (last clean sample per image)
        subj_to_check: Subject identifier for file paths
        label_subset_name: Override for label subset filename
        only_face_set: If True, only load face stimuli
        randomization: If True, randomize trial order
        augment_shared_set: If True, include shared dataset
        seed_offset: Offset for randomization seed

    Returns:
        tuple: (stacked_data, image_ids, mds_mapping)
            - stacked_data: Beta coefficients array
            - image_ids: List of image identifiers
            - mds_mapping: List of MDS indices for each sample

    Raises:
        ValueError: If insufficient clean data is found
        AssertionError: If trial indices don't match or data contains NaN
    """
    assert train_mode in (
        "single",
        "multiple",
        "averaged",
    ), f"Invalid train_mode: {train_mode}"

    logging.info(
        f"Loading beta coefficients - Subject: {subj}, Mode: {train_mode}, Test: {test}"
    )

    # Load behavioral TSV for trial matching
    nsd_dir = os.path.join(
        config.nsd_project.nsd_data_dir, config.nsd_project.nsd_subdir
    )
    tsv_path = os.path.join(
        nsd_dir,
        config.nsd_project.label_subdir,
        "ppdata",
        f"subj{subj:02d}",
        "behav",
        "responses.tsv",
    )
    tsv_data = pd.read_csv(tsv_path, sep="\t")

    # Determine label subset to load
    if label_subset_name is None:
        label_subset_name = config.dataset_creation.subset_animate_face_final
    logging.info(f"Loading labels from: {label_subset_name}")

    # Load base dataset
    set_excel_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_check, label_subset_name
    )

    # Optionally augment with shared dataset
    if augment_shared_set:
        logging.info("Augmenting with shared dataset")
        shared_path = os.path.join(
            config.directories.excel_files_target_dir, "shared", label_subset_name
        )
        shared_df = pd.read_excel(shared_path)
        base_df = pd.concat(
            [pd.read_excel(set_excel_path), shared_df], ignore_index=True
        )
        logging.info(f"Dataset sizes - Shared: {len(shared_df)}, Total: {len(base_df)}")
    else:
        logging.info("Not using shared dataset augmentation")
        temp = pd.read_excel(set_excel_path)
        shared_df = pd.DataFrame(
            columns=temp.columns
        )  # Empty shared_df for later reference
        base_df = temp

    # Optionally add non-face stimuli
    if not only_face_set:
        nonface_path = os.path.join(
            config.directories.excel_files_target_dir,
            subj_to_check,
            config.dataset_creation.subset_animate_non_face_final,
        )
        nonface_df = pd.read_excel(nonface_path)
        base_df = pd.concat([base_df, nonface_df], ignore_index=True)
        logging.info(f"Added non-face stimuli. Total dataset size: {len(base_df)}")

    # Extract image identifiers
    coco_ids = [f"{int(x):012d}" for x in base_df["cocoId"].tolist()]
    nsd_ids = base_df["nsdId"].tolist()

    # Initialize output containers
    data = []
    image_ids = []
    mds_mapping = []
    betas_dir = config.directories.image_betas_dir

    logging.info(f"Processing {len(coco_ids)} images")

    # Process each image
    for mds_index, entry in enumerate(coco_ids):
        cocoId = int(entry)

        # Find trial-matched beta files
        tsv_matches = tsv_data.loc[
            tsv_data["73KID"] == nsd_ids[mds_index] + 1
        ].index.tolist()

        # Find available beta files for this image
        npy_files = sorted(
            glob.glob(
                os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
            )
        )

        # Apply randomization if requested (training only)
        if randomization:
            seed = subj * 10 + int(entry) + seed_offset
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(npy_files))
            npy_files = [npy_files[i] for i in perm]

        npy_files_cleaned = npy_files.copy()

        # Validate trial indices match TSV data
        for npy_path in npy_files:
            trial_id = int(os.path.basename(npy_path)[11:-4])
            assert trial_id in tsv_matches, (
                f"Trial index mismatch for {npy_path}. "
                f"Trial {trial_id} not found in TSV matches: {tsv_matches}"
            )

        # Clean out files with NaN values
        n_usable = len(npy_files)
        faulty = 0

        # Iterate backwards to safely remove items during iteration
        for i in range(len(npy_files) - 1, -1, -1):
            npy_path = npy_files[i]
            arr = np.load(npy_path)
            if np.isnan(arr).any():
                logging.info(f"Removing {npy_path} due to NaN values")
                del npy_files_cleaned[i]
                faulty += 1

        n_usable -= faulty

        # Validate sufficient clean data
        if n_usable == 0:
            if cocoId in shared_df["cocoId"].tolist():
                continue  # Skip shared images with no clean data
            raise ValueError(f"No clean data found for image {entry}")

        # For training, need at least 2 samples for proper train/test split
        if not test and n_usable < 2:
            if cocoId in shared_df["cocoId"].tolist():
                continue  # Skip shared images with insufficient data
            raise ValueError(
                f"Insufficient clean samples for image {entry} (need ≥2, got {n_usable})"
            )

        # Select data based on mode and test flag
        if test:
            # Test mode: always use last clean sample
            picks = [npy_files_cleaned[-1]]
        else:
            # Training modes
            if train_mode == "single":
                picks = [npy_files_cleaned[sample]]
            elif train_mode == "multiple":
                picks = npy_files_cleaned[:-1]  # All but last sample
            else:  # averaged
                picks = npy_files_cleaned[:-1]  # Average all but last sample

        # Load and process selected data
        if test or train_mode == "single":
            # Single sample loading
            arr = np.load(picks[0])
            data.append(arr)
            mds_mapping.append(mds_index)
            image_ids.append(entry)

        elif train_mode == "multiple":
            # Multiple sample loading (each as separate entry)
            for p in picks:
                arr = np.load(p)
                data.append(arr)
                mds_mapping.append(mds_index)
                image_ids.append(entry)

        else:  # averaged
            # Average multiple samples into single entry
            stacked = np.stack([np.load(p) for p in picks])
            mean_arr = np.mean(stacked, axis=0)
            data.append(mean_arr)
            mds_mapping.append(mds_index)
            image_ids.append(entry)

    # Final validation and formatting
    stacked_data = np.stack(data)
    assert not np.isnan(stacked_data).any(), "Final stacked data contains NaN values"
    assert stacked_data.shape[0] == len(image_ids), "Data and image_ids length mismatch"

    logging.info(f"Final data shape: {stacked_data.shape}")
    logging.info(f"Loaded {len(image_ids)} samples successfully")

    return stacked_data, image_ids, mds_mapping


# =========================================================================
# NEGATIVE SET LOADING FUNCTIONS
# =========================================================================


def load_negative_set(
    config: Configuration,
    subj_to_check: str,
    label_subset_name: str = None,
    augment_shared_set: bool = False,
    remove_animate_face: bool = False,
) -> list[tuple[str, list[str]]]:
    """
    Load negative stimulus set with optional filtering and augmentation.

    Loads non-face animate stimuli for contrast analyses, with support for
    shared dataset augmentation and label-based filtering.

    Args:
        config: Configuration object containing paths and dataset parameters
        subj_to_check: Subject identifier for file paths
        label_subset_name: Override filename for negative set (unused, kept for compatibility)
        augment_shared_set: If True, include shared negative stimulus set
        remove_animate_face: If True, exclude animate_face labeled stimuli

    Returns:
        list: List of tuples (name, filenames) where:
            - name: Dataset name identifier
            - filenames: List of stimulus filenames (without path/extension)

    Raises:
        ValueError: If required columns are missing, unsupported labels found,
                   or no samples remain after filtering
    """
    logging.info(f"Loading negative set for {subj_to_check}")

    # Load subject-specific negative set
    subset_fname = config.dataset_creation.subset_animate_non_face_final
    subj_dir = os.path.join(config.directories.excel_files_target_dir, subj_to_check)
    personal_path = os.path.join(subj_dir, subset_fname)

    logging.info(f"Loading personal negative set from: {personal_path}")
    neg_df = pd.read_excel(personal_path)

    # Optionally augment with shared negative set
    if augment_shared_set:
        shared_fname = "shared"
        shared_path = os.path.join(
            config.directories.excel_files_target_dir, shared_fname, subset_fname
        )
        logging.info(f"Augmenting with shared negative set from: {shared_path}")
        shared_df = pd.read_excel(shared_path)
        neg_df = pd.concat([neg_df, shared_df], ignore_index=True)

    # Validate required columns
    if "label" not in neg_df.columns:
        raise ValueError(
            f"Missing 'label' column in {personal_path}. "
            f"Found columns: {list(neg_df.columns)}"
        )

    # Filter out generic 'animate' labels
    initial_count = len(neg_df)
    neg_df = neg_df[neg_df["label"] != "animate"]
    filtered_count = len(neg_df)

    if filtered_count < initial_count:
        logging.info(
            f"Removed {initial_count - filtered_count} generic 'animate' labels"
        )

    # Validate label categories
    allowed_labels = {"animate_persons", "animate_face", "animate_animal"}
    found_labels = set(neg_df["label"].unique())
    unsupported_labels = found_labels - allowed_labels

    if unsupported_labels:
        raise ValueError(
            f"Found unsupported labels: {unsupported_labels}. "
            f"Supported labels: {allowed_labels} (plus 'animate' which is auto-removed)"
        )

    # Optionally remove animate_face labels
    if remove_animate_face:
        pre_removal_count = len(neg_df)
        neg_df = neg_df[neg_df["label"] != "animate_face"]
        post_removal_count = len(neg_df)

        if post_removal_count < pre_removal_count:
            logging.info(
                f"Removed {pre_removal_count - post_removal_count} animate_face labels"
            )

    # Validate remaining data
    if neg_df.empty:
        raise ValueError(
            "No negative examples remain after filtering. "
            "Check label categories and filtering parameters."
        )

    # Extract filenames (remove path and extension)
    filenames = (
        neg_df["file_name"]
        .apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        .tolist()
    )

    # Return standardized format
    dataset_name = label_subset_name or "animate"
    logging.info(f"Loaded {len(filenames)} negative samples with name '{dataset_name}'")

    return [(dataset_name, filenames)]
