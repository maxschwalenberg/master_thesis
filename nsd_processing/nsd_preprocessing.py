"""
NSD Data Processing Module

This module handles preprocessing and extraction of Natural Scenes Dataset (NSD) data.
It includes functionality for loading beta coefficients, standardizing voxel responses,
concatenating sessions, and extracting ROI data with memory-efficient processing.
"""

# Standard library imports
import os
import logging

# Third-party imports
import numpy as np
import nibabel as nib
import h5py as h5
import psutil
from tqdm import tqdm

# Local imports
from utils.config import load_config, Configuration
from utils.utils import subjects_list_unifier, logging_message
from datasetCreation.load_betas import load_betas_subset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load configuration
config = load_config("config.yaml")

# =========================================================================
# GLOBAL PARAMETERS
# =========================================================================

SUBJECTS = range(1, 9)  # Subjects 1 to 8
SESSIONS = range(1, 41)  # Sessions 1 to 40

# Directory configuration
data_dirs = {
    "freesurfer": config.nsd_data.freesurfer_dir,
    "tsv": config.nsd_data.nsddata_responses_tsv_dir,
    "betas": config.nsd_data.nsddata_betas_dir,
    "target": "/media/harveylab/STORAGE1_NA/NSD/full_brain_standardize_no_shift",
}

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================


def get_memory_usage():
    """
    Get current RAM usage statistics.

    Returns:
        dict: Dictionary containing memory usage information in MB and percentage
    """
    process = psutil.Process(os.getpid())
    mem_info = psutil.virtual_memory()

    return {
        "RAM used (MB)": process.memory_info().rss / 1e6,
        "RAM available (MB)": mem_info.available / 1e6,
        "RAM total (MB)": mem_info.total / 1e6,
        "RAM used (%)": mem_info.percent,
    }


# =========================================================================
# DATA LOADING FUNCTIONS
# =========================================================================


def load_v1_rois(subject):
    """
    Load V1 ROI data for a specific subject.

    Args:
        subject: Subject ID (integer)

    Returns:
        np.ndarray: Array of voxel indices belonging to V1 ROIs (labels 1 and 2)
    """
    logging.info(f"Loading V1 ROIs for subject {subject}")

    # Define paths for left and right hemisphere ROI files
    lh_path = os.path.join(
        data_dirs["freesurfer"],
        f"subj{subject:02d}",
        "label",
        "customrois",
        f"lh.subj{subject:02d}.testrois.mgz",
    )
    rh_path = os.path.join(
        data_dirs["freesurfer"],
        f"subj{subject:02d}",
        "label",
        "customrois",
        f"rh.subj{subject:02d}.testrois.mgz",
    )

    # Load ROI data from both hemispheres
    lh_img = nib.load(lh_path).get_fdata()
    rh_img = nib.load(rh_path).get_fdata()

    # Concatenate hemispheres and find V1 voxels (labels 1 and 2)
    v1_rois = np.concatenate((np.squeeze(lh_img), np.squeeze(rh_img)))
    v1_indices = np.where((v1_rois == 1) | (v1_rois == 2))[0]

    logging.info(f"Found {len(v1_indices)} V1 voxels for subject {subject}")
    return v1_indices


def load_betas(subject, session):
    """
    Load beta coefficients for a specific subject and session.

    Args:
        subject: Subject ID (integer)
        session: Session number (integer)

    Returns:
        np.ndarray: Beta coefficients concatenated across hemispheres
                   Shape: (n_trials, n_voxels)
    """
    logging.debug(f"Loading betas for subject {subject}, session {session}")

    # Define directory and file paths
    betas_dir = os.path.join(
        data_dirs["betas"],
        "ppdata",
        f"subj{subject:02d}",
        "nativesurface",
        "betas_fithrf_GLMdenoise_RR",
    )
    beta_lh = os.path.join(betas_dir, f"lh.betas_session{session:02d}.hdf5")
    beta_rh = os.path.join(betas_dir, f"rh.betas_session{session:02d}.hdf5")

    # Load beta data from HDF5 files
    with h5.File(beta_lh, "r") as f:
        betas_lh = f["betas"][:]
    with h5.File(beta_rh, "r") as f:
        betas_rh = f["betas"][:]

    # Concatenate hemispheres along voxel dimension
    betas_combined = np.concatenate((betas_lh, betas_rh), axis=1)

    logging.debug(f"Loaded betas shape: {betas_combined.shape}")
    return betas_combined


# =========================================================================
# DATA PROCESSING FUNCTIONS
# =========================================================================


def process_subject(subject):
    """
    Process a single subject across all sessions with voxel-wise standardization.

    This function loads beta coefficients for each session, applies voxel-wise
    standardization (z-scoring), and saves the processed data.

    Args:
        subject: Subject ID to process
    """
    logging.info(f"Processing Subject {subject}...")

    # Create target directory
    target_dir = os.path.join(data_dirs["target"], f"subj{subject:02d}")
    os.makedirs(target_dir, exist_ok=True)

    # Process each session
    for session in tqdm(SESSIONS, desc=f"Subject {subject} sessions"):
        target_file_path = os.path.join(target_dir, f"betas_session_{session}.npy")

        # Skip if file already exists
        if os.path.exists(target_file_path):
            logging.debug(f"Skipping existing file: {target_file_path}")
            continue

        try:
            # Load beta coefficients
            betas = load_betas(subject, session)

            # Apply voxel-wise standardization
            # Compute standard deviation for each voxel across trials
            voxel_stds = betas.std(axis=0, ddof=0)  # Shape: (n_voxels,)

            # Avoid division by zero (though this should be rare)
            voxel_stds[voxel_stds == 0] = 1.0

            # Standardize each voxel by dividing by its standard deviation
            betas_voxel_scaled = betas / voxel_stds

            # Save processed data
            np.save(target_file_path, betas_voxel_scaled)

            logging.debug(
                f"Processed and saved session {session} for subject {subject}"
            )

        except Exception as e:
            logging.warning(
                f"Failed to process session {session} for subject {subject}: {e}"
            )
            break


def concatenate_subject_sessions(subject):
    """
    Concatenate all processed beta coefficients for a subject across sessions.

    Uses memory mapping to handle large datasets efficiently and saves the result
    in a format compatible with np.load().

    Args:
        subject: Subject ID to process
    """
    logging.info(f"Concatenating sessions for Subject {subject}...")

    target_dir = os.path.join(data_dirs["target"], f"subj{subject:02d}")
    concatenated_file_path = os.path.join(
        target_dir, f"full_betas_subj{subject:02d}.npy"
    )

    # Count available session files
    session_files = [
        f for f in os.listdir(target_dir) if f.startswith("betas_session_")
    ]
    available_n_sessions = len(session_files)

    if available_n_sessions == 0:
        logging.error(f"No session files found for subject {subject}")
        return

    logging.info(f"Found {available_n_sessions} session files for subject {subject}")

    # Load first session to determine dimensions
    first_session_file = os.path.join(target_dir, "betas_session_1.npy")
    if not os.path.exists(first_session_file):
        raise FileNotFoundError(f"Missing file: {first_session_file}")

    # Use memory mapping to avoid RAM overload
    first_betas = np.load(first_session_file, mmap_mode="r")
    n_samples, n_voxels = first_betas.shape

    logging.info(f"Session dimensions: {n_samples} samples, {n_voxels} voxels")

    # Create temporary memory-mapped file for concatenation
    temp_memmap_file = concatenated_file_path + ".tmp"
    total_samples = n_samples * available_n_sessions

    logging.info(f"Creating memory map with shape: ({total_samples}, {n_voxels})")

    concatenated_betas = np.memmap(
        temp_memmap_file,
        dtype=np.float32,
        mode="w+",
        shape=(total_samples, n_voxels),
    )

    # Concatenate session data
    start_idx = 0
    for session in tqdm(
        range(1, available_n_sessions + 1),
        desc=f"Subject {subject}: Concatenating sessions",
    ):
        session_file_path = os.path.join(target_dir, f"betas_session_{session}.npy")

        if os.path.exists(session_file_path):
            # Load session data with memory mapping
            betas = np.load(session_file_path, mmap_mode="r")
            end_idx = start_idx + betas.shape[0]

            # Copy data to concatenated array
            concatenated_betas[start_idx:end_idx] = betas
            start_idx = end_idx
        else:
            logging.warning(f"Warning: {session_file_path} not found. Skipping...")

    # Ensure data is written to disk
    concatenated_betas.flush()

    # Convert memory map to regular NumPy array and save as .npy
    logging.info("Converting memory map to final .npy format...")
    final_array = np.array(concatenated_betas)
    np.save(concatenated_file_path, final_array)

    # Clean up temporary memory map file
    os.remove(temp_memmap_file)
    logging.info(f"Saved file: {concatenated_file_path} in compatible .npy format")

    # Remove individual session files to save space
    logging.info("Cleaning up individual session files...")
    for session in tqdm(SESSIONS, desc=f"Subject {subject}: Deleting session files"):
        session_file_path = os.path.join(target_dir, f"betas_session_{session}.npy")
        try:
            if os.path.exists(session_file_path):
                os.remove(session_file_path)
        except Exception as e:
            logging.warning(f"Failed to delete {session_file_path}: {e}")
            break

    logging.info(f"Concatenation complete for subject {subject}")


# =========================================================================
# MAIN PROCESSING FUNCTION
# =========================================================================


def extract_nsd_data(config: Configuration):
    """
    Main function to extract and process NSD data for specified subjects.

    This function orchestrates the complete data processing pipeline:
    1. Load and standardize beta coefficients for each session
    2. Concatenate sessions for each subject
    3. Extract subject-specific beta subsets

    Args:
        config: Configuration object containing processing parameters
    """
    logging.info("Starting NSD data extraction pipeline")

    # Get list of subjects to process
    subject_list = subjects_list_unifier(
        config.pipeline.step_1_preprocessing.subjects, True
    )

    if config.pipeline.step_1_preprocessing.extract_nsd_data:
        logging.info(logging_message(1, "Starting *NSD preprocessing and extraction*"))

        # Log memory usage before processing
        mem_info = get_memory_usage()
        logging.info(f"Initial memory usage: {mem_info}")

        # Process each subject
        for subject in subject_list:
            logging.info(f"Beginning processing for subject {subject}")

            # Process individual sessions
            process_subject(subject)

            # Concatenate all sessions
            concatenate_subject_sessions(subject)

            # Log memory usage after each subject
            mem_info = get_memory_usage()
            logging.info(f"Memory usage after subject {subject}: {mem_info}")

        logging.info("NSD preprocessing and extraction completed")

    else:
        logging.info(logging_message(1, "Skipping *NSD preprocessing and extraction*"))


# =========================================================================
# EXAMPLE USAGE
# =========================================================================


def example_usage():
    """
    Example usage demonstrating the complete data processing pipeline.

    This function shows how to:
    1. Process multiple subjects
    2. Update configuration parameters
    3. Load beta subsets for further analysis
    """
    logging.info("Running example usage pipeline")

    # Process specified subjects
    for subject in SUBJECTS:
        logging.info(f"Processing subject {subject} in example pipeline")
        process_subject(subject)
        concatenate_subject_sessions(subject)

    # Update configuration for subsequent processing steps
    config.pipeline.step_2_dataset_creation.subjects = [3, 4, 5, 6, 7, 8]
    config.directories.image_betas_dir = (
        "/media/harveylab/STORAGE1_NA/NSD/image_betas_standardize_no_shift"
    )
    config.nsd_data.full_brain_data_dir = (
        "/media/harveylab/STORAGE1_NA/NSD/full_brain_standardize_no_shift"
    )

    # Load beta subsets with updated configuration
    load_betas_subset(config, overwrite=False)

    logging.info("Example usage pipeline completed")
