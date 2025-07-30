import logging
import os
from typing import Tuple

import nibabel as nib
import numpy as np

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

def _load_hemisphere_masks(mask_path_lh: str, mask_path_rh: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.dtype]:
    """
    Load left and right hemisphere mask files and return processed data.
    
    Args:
        mask_path_lh: Path to left hemisphere mask file
        mask_path_rh: Path to right hemisphere mask file
        
    Returns:
        Tuple containing:
        - mask_lh: Left hemisphere mask data
        - mask_rh: Right hemisphere mask data
        - mask_img_lh: Left hemisphere NIfTI image object
        - mask_img_rh: Right hemisphere NIfTI image object
        - original_dtype: Original data type of the mask
    """
    # Load left hemisphere mask
    mask_img_lh = nib.load(mask_path_lh)
    mask_lh = mask_img_lh.get_fdata().squeeze()
    original_dtype = mask_img_lh.get_data_dtype()
    
    # Load right hemisphere mask
    mask_img_rh = nib.load(mask_path_rh)
    mask_rh = mask_img_rh.get_fdata().squeeze()
    
    return mask_lh, mask_rh, mask_img_lh, mask_img_rh, original_dtype


def _construct_file_paths(
    config: Configuration,
    subj_to_pick: int,
    subj_to_pick_shared: bool,
    sub_filename: str,
    threshold: float
) -> Tuple[str, str, str, str, str]:
    """
    Construct all necessary file paths for mask processing.
    
    Args:
        config: Configuration object containing directory paths
        subj_to_pick: Subject number to process
        subj_to_pick_shared: Whether to use shared subject data
        sub_filename: Sub-filename for output files
        threshold: Threshold value for filename
        
    Returns:
        Tuple of file paths: (mask_dir, mask_path_lh, mask_path_rh, new_mask_path_lh, new_mask_path_rh)
    """
    # Determine mask directory based on shared/individual subject
    mask_dir = os.path.join(config.directories.t_test_roi_dir)
    if subj_to_pick_shared:
        mask_dir = os.path.join(mask_dir, "shared")
    else:
        mask_dir = os.path.join(mask_dir, f"subj_{subj_to_pick:02d}")
    
    # Construct input mask paths
    mask_path_lh = os.path.join(mask_dir, f"lh.subj{subj_to_pick:02d}.final.mgz")
    mask_path_rh = os.path.join(mask_dir, f"rh.subj{subj_to_pick:02d}.final.mgz")
    
    # Construct output mask paths
    new_mask_path_lh = os.path.join(
        mask_dir, f"lh.subj{subj_to_pick:02d}.{sub_filename}_{threshold}.mgz"
    )
    new_mask_path_rh = os.path.join(
        mask_dir, f"rh.subj{subj_to_pick:02d}.{sub_filename}_{threshold}.mgz"
    )
    
    return mask_dir, mask_path_lh, mask_path_rh, new_mask_path_lh, new_mask_path_rh


def _load_ttest_data(
    config: Configuration,
    subj_to_pick: int,
    subj_to_pick_shared: bool
) -> np.ndarray:
    """
    Load t-test results data for the specified subject.
    
    Args:
        config: Configuration object containing directory paths
        subj_to_pick: Subject number to process
        subj_to_pick_shared: Whether to use shared subject data
        
    Returns:
        T-test data array
    """
    # Determine subdirectory path based on shared/individual subject
    if subj_to_pick_shared:
        subj_subdir = os.path.join("shared", f"subj_{subj_to_pick:02d}")
    else:
        subj_subdir = f"subj_{subj_to_pick:02d}"
    
    # Load t-test results
    t_data_path = os.path.join(
        config.directories.t_test_results_dir,
        subj_subdir,
        f"result_subj_{subj_to_pick:02d}.npy",
    )
    
    return np.load(t_data_path)


def _apply_threshold_to_roi(
    mask: np.ndarray,
    t_data: np.ndarray,
    threshold: float,
    roi_label: int
) -> None:
    """
    Apply threshold to a specific ROI in the mask based on t-test values.
    
    Args:
        mask: Combined mask array (modified in-place)
        t_data: T-test data array
        threshold: Threshold value for filtering
        roi_label: ROI label to process
    """
    # Get indices for this ROI
    roi_indices = np.where(mask == roi_label)[0]
    
    # Extract corresponding t-values (assuming first column contains t-values)
    roi_t_values = t_data[roi_indices, 0]
    
    # Count positive and negative t-values
    pos_count = np.sum(roi_t_values > 0)
    neg_count = np.sum(roi_t_values < 0)
    
    # Determine thresholding strategy based on majority sign
    if pos_count >= neg_count:
        # Majority positive: keep voxels with t >= threshold
        keep_voxels = roi_t_values >= threshold
        logging.info(
            f"ROI {roi_label}: majority positive (pos={pos_count}, neg={neg_count}) "
            f"- keeping voxels with t >= {threshold} --> keep={keep_voxels.sum()}"
        )
    else:
        # Majority negative: keep voxels with t <= -threshold
        keep_voxels = roi_t_values <= -threshold
        logging.info(
            f"ROI {roi_label}: majority negative (pos={pos_count}, neg={neg_count}) "
            f"- keeping voxels with t <= {-threshold} --> keep={keep_voxels.sum()}"
        )
    
    # Set voxels that don't meet condition to 0
    roi_to_zero = roi_indices[~keep_voxels]
    mask[roi_to_zero] = 0


def _save_hemisphere_mask(
    mask_data: np.ndarray,
    original_img: nib.Nifti1Image,
    output_path: str,
    original_dtype: np.dtype,
    hemisphere: str
) -> None:
    """
    Save processed mask data to a NIfTI file.
    
    Args:
        mask_data: Processed mask data
        original_img: Original NIfTI image object for affine transformation
        output_path: Path to save the new mask file
        original_dtype: Original data type to preserve
        hemisphere: Hemisphere identifier for logging ('left' or 'right')
    """
    # Create NIfTI image with proper dimensions and affine transformation
    img = nib.Nifti1Image(
        np.expand_dims(mask_data, axis=(1, 2)),
        original_img.affine,
        dtype=original_dtype,
    )
    
    # Save the image
    nib.save(img, output_path)
    logging.info(f"Saving modified {hemisphere} hemisphere mask to {output_path}")


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def modify_mask_with_ttest(
    config: Configuration,
    threshold: float,
    subj_to_pick_shared: bool,
    subj_to_pick: int,
    sub_filename: str = "cleanedrois",
) -> None:
    """
    Modify existing mask files based on t-test results.
    
    Sets mask values to 0 where t-test values are below the specified threshold.
    The thresholding strategy depends on the majority sign of t-values in each ROI:
    - For ROIs with majority positive t-values: keep voxels with t >= threshold
    - For ROIs with majority negative t-values: keep voxels with t <= -threshold
    
    Args:
        config: Configuration object containing necessary directory paths
        threshold: T-test value threshold for filtering voxels
        subj_to_pick_shared: Whether to use shared subject data directory structure
        subj_to_pick: Subject number to process
        sub_filename: Suffix for output filenames (default: "cleanedrois")
    
    Raises:
        AssertionError: If t-test data shape doesn't match mask shape
        FileNotFoundError: If required input files are not found
    """
    logging.info(f"Processing subject {subj_to_pick} with threshold {threshold}")
    
    # Construct all necessary file paths
    _, mask_path_lh, mask_path_rh, new_mask_path_lh, new_mask_path_rh = _construct_file_paths(
        config, subj_to_pick, subj_to_pick_shared, sub_filename, threshold
    )
    
    # Load hemisphere masks
    mask_lh, mask_rh, mask_img_lh, mask_img_rh, original_dtype = _load_hemisphere_masks(
        mask_path_lh, mask_path_rh
    )
    
    # Store original hemisphere lengths for later splitting
    mask_lh_len = mask_lh.shape[0]
    mask_rh_len = mask_rh.shape[0]
    
    # Concatenate hemispheres to align with t-test data structure
    mask = np.concatenate((mask_lh, mask_rh)).astype(int)
    
    # Load t-test data
    t_data = _load_ttest_data(config, subj_to_pick, subj_to_pick_shared)
    
    # Validate data alignment
    assert t_data.shape[0] == mask.shape[0], (
        f"T-test data shape ({t_data.shape[0]}) doesn't match mask shape ({mask.shape[0]})"
    )
    
    # Process each unique ROI label
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background (label 0)
        
        _apply_threshold_to_roi(mask, t_data, threshold, label)
    
    # Split processed mask back into hemispheres
    new_mask_lh = mask[0:mask_lh_len].copy()
    new_mask_rh = mask[mask_lh_len:].copy()
    
    # Save processed hemisphere masks
    _save_hemisphere_mask(
        new_mask_lh, mask_img_lh, new_mask_path_lh, original_dtype, "left"
    )
    _save_hemisphere_mask(
        new_mask_rh, mask_img_rh, new_mask_path_rh, original_dtype, "right"
    )
    
    logging.info(f"Successfully processed subject {subj_to_pick} with threshold {threshold}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Example usage of the mask modification function.
    Uncomment and modify the parameters as needed.
    """
    # Example usage:
    # config = load_config("config.yaml")
    # modify_mask_with_ttest(
    #     config=config,
    #     threshold=2.0,
    #     subj_to_pick_shared=False,
    #     subj_to_pick=2,
    #     sub_filename="cleanedrois"
    # )
    pass


if __name__ == "__main__":
    main()