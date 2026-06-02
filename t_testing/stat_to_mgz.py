import glob
import logging
import os
import re
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
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
# HELPER FUNCTIONS
# =============================================================================


def _construct_output_directory(
    config: Configuration, subdir: str = "mutually_exclusive"
) -> str:
    """
    Create and return the output directory path for processed results.

    Args:
        config: Configuration object containing directory paths
        subdir: Subdirectory name for output

    Returns:
        Path to the created output directory
    """
    output_dir = os.path.join(config.directories.t_test_results_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _load_t_test_results(
    config: Configuration, subj_num: int, subdir: str
) -> np.ndarray:
    """
    Load t-test results for a specific subject and subdirectory.

    Args:
        config: Configuration object containing directory paths
        subj_num: Subject number
        subdir: Subdirectory containing the results

    Returns:
        Loaded t-test results array
    """
    filepath = os.path.join(
        config.directories.t_test_results_dir,
        subdir,
        f"result_subj_{subj_num:02d}.npy",
    )
    return np.load(filepath)


def _apply_mutual_exclusivity_filter(
    face_animate_results: np.ndarray,
    animate_nonanimate_results: np.ndarray,
    threshold: float,
    difference_threshold: float = 3.0,
) -> np.ndarray:
    """
    Apply mutual exclusivity filtering to face-animate t-test results.

    Sets values to 0 where the difference between face-animate and animate-nonanimate
    results is below the specified threshold.

    Args:
        face_animate_results: Face vs animate t-test results
        animate_nonanimate_results: Animate vs non-animate t-test results
        threshold: Minimum threshold for face-animate results
        difference_threshold: Minimum difference threshold between conditions

    Returns:
        Filtered face-animate results array
    """
    filtered_results = face_animate_results.copy()

    for index in tqdm(
        range(animate_nonanimate_results.shape[0]), desc="Applying mutual exclusivity"
    ):
        if face_animate_results[index][0] > threshold:
            # Apply mutual exclusivity based on difference threshold
            if (
                face_animate_results[index][0] - animate_nonanimate_results[index][0]
            ) < difference_threshold:
                filtered_results[index][0] = 0

    return filtered_results


def _get_subject_list(config: Configuration) -> List[str]:
    """
    Generate list of subject identifiers from configuration.

    Args:
        config: Configuration object containing subject numbers

    Returns:
        List of subject identifier strings
    """
    return [f"subj{i:02d}" for i in config.pipeline.step_3_t_testing.subjects]


def _construct_subject_subdir(subj_num: str, subj_to_pick_shared: bool) -> str:
    """
    Construct subject subdirectory path based on sharing configuration.

    Args:
        subj_num: Subject number string (e.g., "01")
        subj_to_pick_shared: Whether to use shared subject data

    Returns:
        Subject subdirectory path
    """
    if subj_to_pick_shared:
        return os.path.join("shared", f"subj_{subj_num}")
    else:
        return f"subj_{subj_num}"


def _get_mask_file_path(config: Configuration, sub: str, hemi: str) -> str:
    """
    Get the appropriate mask file path for a subject and hemisphere.

    Args:
        config: Configuration object containing directory paths
        sub: Subject identifier (e.g., "subj01")
        hemi: Hemisphere identifier ("lh" or "rh")

    Returns:
        Path to the mask file
    """
    mask_dir = os.path.join(
        config.nsd_data.stans_thesis_repo_data, config.nsd_data.mask_data_dir
    )

    # Special handling for subjects 06 and 08
    if sub in ["subj06", "subj08"]:
        return os.path.join(mask_dir, sub, f"{hemi}.{sub}.nans.testrois.mgz")
    else:
        return os.path.join(mask_dir, sub, f"{hemi}.{sub}.testrois.mgz")


def _construct_output_filename(
    config: Configuration,
    sub: str,
    hemi: str,
    mode: str,
    threshold: float,
    label: str,
    shared_string: str,
    clipping_value: Optional[float] = None,
) -> str:
    """
    Construct the output filename for processed t-test results.

    Args:
        config: Configuration object containing directory paths
        sub: Subject identifier
        hemi: Hemisphere identifier
        mode: Processing mode ("absolute" or "signed")
        threshold: Threshold value
        label: Label identifier
        shared_string: Shared data indicator string
        clipping_value: Optional clipping value

    Returns:
        Full path to output file
    """
    base_dir = os.path.join(config.nsd_data.freesurfer_dir, sub, "label")
    threshold_str = str(threshold).replace(".", "_")

    if clipping_value is None:
        filename = (
            f"{hemi}.t_test_{mode}_{threshold_str}_label_{label}_{shared_string}.mgz"
        )
    else:
        filename = f"{hemi}.t_test_{mode}_{threshold_str}_label_{label}_clip_{clipping_value}_{shared_string}.mgz"

    return os.path.join(base_dir, filename)


def _apply_data_scaling(
    data: np.ndarray, clipping_value: float, preserve_zeros: bool = True
) -> np.ndarray:
    """
    Apply linear scaling to data within specified range.

    Args:
        data: Input data array
        clipping_value: Value used for clipping range
        preserve_zeros: Whether to preserve zero values during scaling

    Returns:
        Scaled data array
    """
    # Linear transformation to [0.1, 3.9]
    scale = ((2 * clipping_value - 0.1) - 0.1) / (2 * clipping_value - 0)
    arr_scaled = 0.1 + scale * (data - 0)

    # Correct mean if necessary
    mean_before = np.mean(data)
    mean_after = np.mean(arr_scaled)
    shift = mean_before - mean_after

    # Shift values uniformly to restore original mean
    scaled_data = arr_scaled + shift

    return scaled_data


def _process_hemisphere_data(
    t_data: np.ndarray,
    maskdata_long: np.ndarray,
    hemi_idx: int,
    hemisphere_shapes: List[int],
    mode: str,
    threshold: float,
    thresholding: bool,
    min_value: float,
) -> np.ndarray:
    """
    Process t-test data for a single hemisphere.

    Args:
        t_data: T-test results data
        maskdata_long: Mask data for the hemisphere
        hemi_idx: Hemisphere index (0 for left, 1 for right)
        hemisphere_shapes: List of hemisphere shapes
        mode: Processing mode ("absolute" or "signed")
        threshold: Threshold value for filtering
        thresholding: Whether to apply thresholding
        min_value: Minimum value for signed mode processing

    Returns:
        Processed data array for the hemisphere
    """
    data_out = np.zeros(maskdata_long.shape)

    for i in range(maskdata_long.shape[0]):
        # Calculate index based on hemisphere
        if hemi_idx == 0:
            index = i
        else:
            index = i + hemisphere_shapes[0]

        # Apply thresholding if enabled
        if thresholding and np.abs(t_data[index][0]) < threshold:
            continue

        # Apply mode-specific processing
        if mode == "absolute":
            data_out[i] = np.abs(t_data[index][0])
        else:  # signed mode
            data_out[i] = t_data[index][0] + np.abs(min_value)

    return data_out


def _apply_final_thresholding(
    data_out: np.ndarray,
    t_data: np.ndarray,
    maskdata_long: np.ndarray,
    hemi_idx: int,
    hemisphere_shapes: List[int],
    threshold: float,
    thresholding: bool,
) -> np.ndarray:
    """
    Apply final thresholding to processed data.

    Args:
        data_out: Processed data array
        t_data: Original t-test data
        maskdata_long: Mask data
        hemi_idx: Hemisphere index
        hemisphere_shapes: List of hemisphere shapes
        threshold: Threshold value
        thresholding: Whether to apply thresholding

    Returns:
        Final thresholded data array
    """
    if not thresholding:
        return data_out

    final_data = data_out.copy()

    for i in range(maskdata_long.shape[0]):
        if hemi_idx == 0:
            index = i
        else:
            index = i + hemisphere_shapes[0]

        if np.abs(t_data[index][0]) < threshold:
            final_data[i] = 0

    return final_data


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================


def mutually_exclusive_t_test_export(
    config: Configuration,
    threshold: float,
    face_animate_t_test_subdir: str,
    animate_nonanimate_t_test_subdir: str,
    subj_num: int = 1,
    difference_threshold: float = 3.0,
) -> None:
    """
    Export mutually exclusive t-test results by filtering face-animate results.

    This function loads face vs animate and animate vs non-animate t-test results,
    then applies mutual exclusivity filtering to reduce overlapping activations.

    Args:
        config: Configuration object containing directory paths
        threshold: Minimum threshold for face-animate results
        face_animate_t_test_subdir: Subdirectory containing face vs animate results
        animate_nonanimate_t_test_subdir: Subdirectory containing animate vs non-animate results
        subj_num: Subject number to process (default: 1)
        difference_threshold: Minimum difference threshold between conditions (default: 3.0)
    """
    logging.info(f"Processing mutual exclusivity for subject {subj_num:02d}")

    # Create output directory
    output_dir = _construct_output_directory(config, "mutually_exclusive")

    # Load t-test results
    face_animate_results = _load_t_test_results(
        config, subj_num, face_animate_t_test_subdir
    )
    animate_nonanimate_results = _load_t_test_results(
        config, subj_num, animate_nonanimate_t_test_subdir
    )

    # Apply mutual exclusivity filtering
    filtered_results = _apply_mutual_exclusivity_filter(
        face_animate_results,
        animate_nonanimate_results,
        threshold,
        difference_threshold,
    )

    # Save filtered results
    output_path = os.path.join(output_dir, f"result_subj_{subj_num:02d}.npy")
    np.save(output_path, filtered_results)

    logging.info(f"Saved mutually exclusive results to {output_path}")


def t_test_results_to_mgz(
    config: Configuration,
    mode: str,
    threshold: float,
    label: str,
    subj_to_pick_shared: bool,
    thresholding: bool = True,
    t_test_results_subdir: str = "",
    clipping_value: Optional[float] = None,
) -> None:
    """
    Convert t-test results to MGZ format for visualization in FreeSurfer.

    This function processes t-test results and converts them to MGZ format files
    that can be visualized on brain surfaces in FreeSurfer. Supports both absolute
    and signed modes, with optional thresholding and value clipping.

    Args:
        config: Configuration object containing all necessary paths
        mode: Processing mode - "absolute" for absolute values, "signed" for signed values
        threshold: Statistical threshold for filtering significant voxels
        label: Label identifier for output filenames
        subj_to_pick_shared: Whether to use shared subject data structure
        thresholding: Whether to apply statistical thresholding (default: True)
        t_test_results_subdir: Subdirectory containing t-test results (default: "")
        clipping_value: Optional value for clipping extreme t-statistics

    Raises:
        AssertionError: If mode is not "absolute" or "signed"
        NotImplementedError: If clipping_value is None (auto-scaling not implemented)
    """
    # Validate input parameters
    assert mode in [
        "absolute",
        "signed",
    ], f"Mode must be 'absolute' or 'signed', got '{mode}'"

    if clipping_value is None:
        raise NotImplementedError(
            "Auto-scaling functionality removed. Only works with predefined clipping range."
        )

    # Initialize processing parameters
    hemis = ["lh", "rh"]
    subs = _get_subject_list(config)
    shared_string = "shared" if subj_to_pick_shared else "no_shared"

    # Set value ranges based on clipping
    max_value = clipping_value
    min_value = -clipping_value
    scale = ((2 * clipping_value - 0.1) - 0.1) / (2 * clipping_value - 0)

    # Process each subject
    for sub_i, sub in enumerate(subs):
        logging.info(
            f"Processing {sub} - mode={mode}, thresholding={thresholding}, threshold={threshold}"
        )

        # Construct subject subdirectory
        subj_subdir = _construct_subject_subdir(sub[4:], subj_to_pick_shared)

        # Load t-test data
        t_data_path = os.path.join(
            config.directories.t_test_results_dir,
            subj_subdir,
            f"result_subj_{sub[4:]}.npy",
        )

        logging.info(f"Loading {t_data_path}")
        t_data = np.load(t_data_path)

        # Apply clipping
        t_data = np.clip(t_data, min_value, max_value)

        hemisphere_shapes = []

        # Process each hemisphere
        for hemi_i, hemi in enumerate(hemis):
            # Load mask data
            maskdata_long_file = _get_mask_file_path(config, sub, hemi)
            maskdata_long = nib.load(maskdata_long_file).get_fdata().squeeze()
            hemisphere_shapes.append(maskdata_long.shape[0])

            # Construct output filename
            data_out_file = _construct_output_filename(
                config, sub, hemi, mode, threshold, label, shared_string, clipping_value
            )

            logging.info(f"Saving to {data_out_file}")

            # Process hemisphere data
            data_out = _process_hemisphere_data(
                t_data,
                maskdata_long,
                hemi_i,
                hemisphere_shapes,
                mode,
                threshold,
                thresholding,
                min_value,
            )

            # Apply data scaling
            data_out = _apply_data_scaling(data_out, clipping_value)

            # Apply final thresholding
            data_out = _apply_final_thresholding(
                data_out,
                t_data,
                maskdata_long,
                hemi_i,
                hemisphere_shapes,
                threshold,
                thresholding,
            )

            # Save as MGZ file
            img = nib.Nifti1Image(np.expand_dims(data_out, axis=(1, 2)), np.eye(4))
            nib.loadsave.save(img, data_out_file)

        # Log processing statistics
        if mode == "signed":
            range_max = max(max_value, np.abs(min_value))
            val_range = [-range_max, range_max]
            adjusted_range = [0, 2 * range_max]

            logging.info(f"Data range: {val_range}")
            logging.info(f"Adjusted data range: {adjusted_range}")
            logging.info(f"Maximum value: {max_value}")
            logging.info(f"Minimum value: {min_value}")
        else:
            logging.info(f"Maximum value: {max_value}")
            logging.info(f"Minimum value: {min_value}")

        logging.info("-" * 50)


# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================


def process_multiple_thresholds(
    config: Configuration,
    label_subdirs: List[str],
    modes: List[str],
    thresholds: List[float],
    subj_to_pick_shared: bool = False,
    t_test_subdir: str = "subj_01",
    clipping_value: float = 7.0,
) -> None:
    """
    Process t-test results with multiple thresholds and modes.

    Args:
        config: Configuration object
        label_subdirs: List of label subdirectories to process
        modes: List of processing modes ("absolute", "signed")
        thresholds: List of threshold values to apply
        subj_to_pick_shared: Whether to use shared subject data
        t_test_subdir: T-test results subdirectory
        clipping_value: Value for clipping extreme statistics
    """
    for labels_subdir in label_subdirs:
        for mode in modes:
            # Process without thresholding (threshold = 0.0)
            t_test_results_to_mgz(
                config=config,
                mode=mode,
                threshold=0.0,
                label=labels_subdir,
                subj_to_pick_shared=subj_to_pick_shared,
                thresholding=False,
                t_test_results_subdir=t_test_subdir,
                clipping_value=clipping_value,
            )

            # Process with each threshold
            for threshold in thresholds:
                t_test_results_to_mgz(
                    config=config,
                    mode=mode,
                    threshold=threshold,
                    label=labels_subdir,
                    subj_to_pick_shared=subj_to_pick_shared,
                    thresholding=True,
                    t_test_results_subdir=t_test_subdir,
                    clipping_value=clipping_value,
                )


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """
    Example usage of the t-test processing functions.
    Uncomment and modify the parameters as needed.
    """
    # Example usage:
    # config = load_config("config.yaml")

    # # Process mutual exclusivity
    # mutually_exclusive_t_test_export(
    #     config=config,
    #     threshold=5.0,
    #     face_animate_t_test_subdir="face_animate",
    #     animate_nonanimate_t_test_subdir="animate_nonanimate"
    # )

    # # Process multiple thresholds
    # process_multiple_thresholds(
    #     config=config,
    #     label_subdirs=["face_animate_new"],
    #     modes=["absolute", "signed"],
    #     thresholds=[5.0, 4.0, 3.0, 2.0],
    #     subj_to_pick_shared=False,
    #     t_test_subdir="subj_01",
    #     clipping_value=7.0
    # )

    pass


if __name__ == "__main__":
    main()
