"""
RDM Creation Module

This module takes the betas and mask to compute the Representational Dissimilarity Matrix (RDM)
based on the pdist scipy function.

The RDMs are then stored under the projects directory, and can be accessed later on. 
Then, we used these RDM (or ones previously computed) to get the corresponding MDS.
"""

# Standard library imports
import os
import glob
import time
from typing import Union
import logging

# Third-party imports
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

# Local imports
from previousCode.nsddatapaper_rsa.utils.utils import mds
from t_testing.clean_roi_mask import modify_mask_with_ttest
from utils.config import Configuration, load_config
from utils.utils import retrieve_stacked_betas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================

def emd(u, v):
    """
    Custom metric wrapper for Earth Mover's Distance (Wasserstein distance).
    
    Args:
        u, v: Input arrays for distance computation
        
    Returns:
        float: Wasserstein distance between u and v
    """
    return wasserstein_distance(u, v)


def _make_fname(mask_value, mode, hemisphere, rnd, rnd_offset, sample, suffix):
    """
    Create standardized filename for RDM and MDS outputs.
    
    Args:
        mask_value: ROI mask value
        mode: Analysis mode ("averaged" or "single")
        hemisphere: Hemisphere identifier ("lh", "rh", or "both")
        rnd: Whether randomization was used
        rnd_offset: Random seed offset
        sample: Sample number (for single mode)
        suffix: File suffix ("rdm" or "mds")
        
    Returns:
        str: Formatted filename
    """
    parts = [f"mask_{mask_value}", mode]
    if mode == "single":
        parts.append(f"sample_{sample}")
    parts += [hemisphere]
    if rnd:
        parts.append(f"rand_{rnd_offset}")
    return "_".join(parts) + f"_{suffix}.npy"


# =========================================================================
# MAIN RDM CREATION FUNCTIONS
# =========================================================================

def create_rdm(
    config: Configuration,
    list_subj,
    mask_value: Union[int, list[int]],
    set_to_take: str,
    t_test_threshold: float,
    mode="averaged",
    sample_to_pick: int = 0,
    randomization: bool = False,
    augment_shared_set: bool = True,
    randomize_offset: int = 0
):
    """
    Create Representational Dissimilarity Matrices (RDMs) and corresponding MDS outputs.
    
    Args:
        config: Configuration object containing directory paths and analysis parameters
        list_subj: List of subject IDs to process
        mask_value: ROI mask value(s) to analyze (int or list of ints)
        set_to_take: Dataset identifier (e.g., "shared" or subject-specific)
        t_test_threshold: Threshold for t-test based masking
        mode: Analysis mode - "averaged" or "single"
        sample_to_pick: Sample index for single mode
        randomization: Whether to apply randomization
        augment_shared_set: Whether to augment the shared dataset
        randomize_offset: Random seed offset for reproducibility
    """
    assert mode in ["averaged", "single"], f"Mode must be 'averaged' or 'single', got {mode}"
    
    logging.info(f"Creating RDM for mode {mode} - T-Test THRESHOLD: {t_test_threshold}")
    logging.info(f"Using distance-metric={config.pipeline.step_4_rsa_analysis.distance_metric}")
    time.sleep(0.25)
    logging.info(f"Using positive set: {os.path.join(set_to_take, config.dataset_creation.subset_animate_face_final)}")

    # Handle mask_value input formatting
    if isinstance(mask_value, int):
        mask_values = [mask_value]
    elif isinstance(mask_value, list):
        mask_values = mask_value.copy()
    else:
        raise ValueError("mask_value must be int or list of ints")

    logging.info(f"Mask Values for RDM creation: {mask_values}")

    if mode == "single":
        logging.info(f"Picking sample {sample_to_pick}")

    # Process each subject
    for i, sub in enumerate(list_subj):
        logging.info(f"Processing subject {sub} ({i+1}/{len(list_subj)})")
        
        # Determine if using shared set or not
        pick_shared = set_to_take == "shared"

        # Modify and generate temporary t-test mask files
        modify_mask_with_ttest(
            config, t_test_threshold, pick_shared, sub, sub_filename="temporary_mask"
        )

        # Load left and right hemisphere masks
        mask_path_lh = os.path.join(
            config.directories.t_test_roi_dir,
            set_to_take,
            f"lh.subj{sub:02d}.temporary_mask.mgz",
        )
        mask_path_rh = os.path.join(
            config.directories.t_test_roi_dir,
            set_to_take,
            f"rh.subj{sub:02d}.temporary_mask.mgz",
        )
        logging.info(f"Loading mask from\n{mask_path_rh=}\n{mask_path_lh=}")

        mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()
        mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()

        # Load beta values and transpose for further analysis
        betas, image_ids, mds_mapping = retrieve_stacked_betas(
            config,
            sub,
            mode,
            sample_to_pick,
            subj_to_check=set_to_take,
            only_face_set=True,
            randomization=randomization,
            augment_shared_set=augment_shared_set,
            seed_offset=randomize_offset
        )
        
        # Validate beta data
        assert not np.isnan(betas).any(), "NaN values found in beta data"
        betas = np.transpose(betas)

        # Create combined mask and validate dimensions
        combined_mask = np.concatenate((mask_lh, mask_rh)).astype(int)
        assert betas.shape[0] == combined_mask.shape[0], (
            f"Beta shape {betas.shape[0]} does not match combined mask shape "
            f"{combined_mask.shape[0]} for subject {sub}"
        )

        # Split beta values according to hemisphere (left first, then right)
        n_lh = mask_lh.shape[0]
        betas_both = betas  # full concatenated beta matrix
        betas_lh = betas[:n_lh, :]
        betas_rh = betas[n_lh:, :]

        # Create output directories and save metadata
        rdm_dir_subject = os.path.join(
            config.directories.rdm_dir, set_to_take, f"subj_{sub:02d}"
        )
        os.makedirs(rdm_dir_subject, exist_ok=True)
        
        metadata_file = os.path.join(rdm_dir_subject, "metadata.npy")
        logging.info(f"Number of image IDs: {len(image_ids)}")
        np.save(metadata_file, image_ids)

        # Create MDS output directory
        mds_dir_subject = os.path.join(
            config.directories.mds_dir, set_to_take, f"subj_{sub:02d}"
        )
        os.makedirs(mds_dir_subject, exist_ok=True)

        # Process each hemisphere
        _process_hemispheres(
            config, sub, mask_values, combined_mask, mask_lh, mask_rh,
            betas_both, betas_lh, betas_rh, rdm_dir_subject, mds_dir_subject,
            mode, sample_to_pick, randomization, randomize_offset
        )


def _process_hemispheres(
    config, sub, mask_values, combined_mask, mask_lh, mask_rh,
    betas_both, betas_lh, betas_rh, rdm_dir_subject, mds_dir_subject,
    mode, sample_to_pick, randomization, randomize_offset
):
    """
    Process RDM creation for all hemispheres (both, lh, rh).
    
    Args:
        config: Configuration object
        sub: Subject ID
        mask_values: List of mask values to process
        combined_mask, mask_lh, mask_rh: Hemisphere masks
        betas_both, betas_lh, betas_rh: Beta values for each hemisphere
        rdm_dir_subject, mds_dir_subject: Output directories
        mode, sample_to_pick, randomization, randomize_offset: Analysis parameters
    """
    hemispheres = ["both", "lh", "rh"]
    
    for hemisphere in hemispheres:
        # Select appropriate mask and beta data for current hemisphere
        if hemisphere == "both":
            current_mask = combined_mask
            current_betas = betas_both
        elif hemisphere == "lh":
            current_mask = mask_lh.astype(int)
            current_betas = betas_lh
        elif hemisphere == "rh":
            current_mask = mask_rh.astype(int)
            current_betas = betas_rh

        logging.info(f"Processing hemisphere '{hemisphere}' for subject {sub:02d}")

        # Process each mask value
        for mask_value in mask_values:
            _process_single_mask(
                config, sub, mask_value, hemisphere, current_mask, current_betas,
                rdm_dir_subject, mds_dir_subject, mode, sample_to_pick,
                randomization, randomize_offset
            )


def _process_single_mask(
    config, sub, mask_value, hemisphere, current_mask, current_betas,
    rdm_dir_subject, mds_dir_subject, mode, sample_to_pick,
    randomization, randomize_offset
):
    """
    Process RDM creation for a single mask value and hemisphere.
    
    Args:
        config: Configuration object
        sub: Subject ID
        mask_value: Current mask value being processed
        hemisphere: Current hemisphere ("both", "lh", or "rh")
        current_mask: Mask data for current hemisphere
        current_betas: Beta values for current hemisphere
        rdm_dir_subject, mds_dir_subject: Output directories
        mode, sample_to_pick, randomization, randomize_offset: Analysis parameters
    """
    # Create file paths
    base_kwargs = dict(
        mask_value=mask_value,
        mode=mode,
        hemisphere=hemisphere,
        rnd=randomization,
        rnd_offset=randomize_offset,
        sample=sample_to_pick,
    )

    rdm_file = os.path.join(
        rdm_dir_subject,
        _make_fname(**base_kwargs, suffix="rdm")
    )
    mds_file = os.path.join(
        mds_dir_subject,
        _make_fname(**base_kwargs, suffix="mds")
    )

    # Apply mask to beta values
    masked_voxels = current_mask == mask_value
    masked_betas = current_betas[masked_voxels, :]

    # Validate masked data
    if np.isnan(masked_betas).any():
        raise ValueError("Found NaNs in masked betas!")

    # Remove voxels with any NaN values along the feature axis
    good_vox = [np.sum(np.isnan(x)) == 0 for x in masked_betas]
    masked_betas = masked_betas[good_vox, :]

    if masked_betas.shape[0] == 0:
        logging.info(
            f"All voxels have NaN values ... {mask_value=} for hemisphere '{hemisphere}'"
        )
        return  # Skip to the next mask value

    logging.info(
        f"Shape of masked betas for mask value {mask_value}, hemisphere '{hemisphere}': {masked_betas.shape}"
    )

    # Additional NaN check and removal
    if np.isnan(masked_betas).any():
        masked_betas = masked_betas[~np.isnan(masked_betas).any(axis=1), :]

    # Transpose for distance computation (features as rows)
    X = masked_betas.T
    logging.info(f"Masked betas shape for distance computation: {X.shape}")

    # Determine distance metric
    if config.pipeline.step_4_rsa_analysis.distance_metric != "wasserstein":
        metric_to_use = config.pipeline.step_4_rsa_analysis.distance_metric
    else:
        metric_to_use = emd

    # Compute RDM using pdist
    rdm = pdist(X, metric=metric_to_use)
    logging.info(f"Computed RDM shape for hemisphere '{hemisphere}': {rdm.shape}")

    if np.any(np.isnan(rdm)):
        raise ValueError("NaN values found in RDM")

    # Save RDM
    np.save(rdm_file, rdm)
    logging.info(f"Saved RDM to {rdm_file}")

    # Compute and save MDS
    rdm_loaded = np.load(rdm_file, allow_pickle=True).astype(np.float32)
    mds_out = mds(rdm_loaded).astype(np.float32)
    logging.info(f"MDS output shape for hemisphere '{hemisphere}': {mds_out.shape}")
    
    np.save(mds_file, mds_out)
    logging.info(f"Saved MDS output to {mds_file}")


def rdm_mds_from_betas(
    config: Configuration, 
    betas, 
    combined_mask, 
    mask_size_lh, 
    hemis: list[str], 
    sub: int, 
    mask_value: int
):
    """
    Create RDM and MDS directly from beta values without saving to disk.
    
    Args:
        config: Configuration object
        betas: Beta coefficient data
        combined_mask: Combined hemisphere mask
        mask_size_lh: Size of left hemisphere mask
        hemis: List of hemispheres to process
        sub: Subject ID
        mask_value: ROI mask value to analyze
        
    Returns:
        tuple: (rdms, mdss) dictionaries with hemisphere keys
    """
    # Validate inputs
    assert not np.isnan(betas).any(), "NaN values found in beta data"
    assert isinstance(mask_value, int), "mask_value must be an integer"

    mask_values = [mask_value]
    betas = np.transpose(betas)

    # Validate dimensions
    assert betas.shape[0] == combined_mask.shape[0], (
        f"Beta shape {betas.shape[0]} does not match combined mask shape "
        f"{combined_mask.shape[0]} for subject {sub}"
    )

    # Split data by hemisphere
    betas_both = betas  # full concatenated beta matrix
    betas_lh = betas[:mask_size_lh, :]
    betas_rh = betas[mask_size_lh:, :]

    mask_lh = combined_mask[:mask_size_lh]
    mask_rh = combined_mask[mask_size_lh:]

    rdms = {}
    mdss = {}

    # Process each hemisphere
    hemispheres = ["both", "lh", "rh"]
    for hemisphere in hemispheres:
        # Select appropriate data for current hemisphere
        if hemisphere == "both":
            current_mask = combined_mask
            current_betas = betas_both
        elif hemisphere == "lh":
            current_mask = mask_lh.astype(int)
            current_betas = betas_lh
        elif hemisphere == "rh":
            current_mask = mask_rh.astype(int)
            current_betas = betas_rh

        logging.info(f"Processing hemisphere '{hemisphere}' for subject {sub:02d}")

        # Process the single mask value
        assert len(mask_values) == 1, "This function expects exactly one mask value"
        
        for mask_val in mask_values:
            masked_voxels = current_mask == mask_val
            masked_betas = current_betas[masked_voxels, :]

            # Validate and clean data
            if np.isnan(masked_betas).any():
                raise ValueError("Found NaNs in masked betas!")

            # Remove voxels with NaN values
            good_vox = [np.sum(np.isnan(x)) == 0 for x in masked_betas]
            masked_betas = masked_betas[good_vox, :]

            if masked_betas.shape[0] == 0:
                logging.info(
                    f"All voxels have NaN values ... {mask_val=} for hemisphere '{hemisphere}'"
                )
                continue  # Skip to the next hemisphere

            logging.info(
                f"Shape of masked betas for mask value {mask_val}, hemisphere '{hemisphere}': {masked_betas.shape}"
            )

            # Additional NaN cleaning
            if np.isnan(masked_betas).any():
                masked_betas = masked_betas[~np.isnan(masked_betas).any(axis=1), :]

            # Transpose for distance computation (features as rows)
            X = masked_betas.T
            logging.info(f"Masked betas shape for distance computation: {X.shape}")

            # Determine distance metric
            if config.pipeline.step_4_rsa_analysis.distance_metric != "wasserstein":
                metric_to_use = config.pipeline.step_4_rsa_analysis.distance_metric
            else:
                metric_to_use = emd

            # Compute RDM
            rdm = pdist(X, metric=metric_to_use)
            logging.info(f"Computed RDM shape for hemisphere '{hemisphere}': {rdm.shape}")

            if np.any(np.isnan(rdm)):
                raise ValueError("NaN values found in RDM")

            # Compute MDS
            mds_out = mds(rdm).astype(np.float32)

            # Store results
            rdms[hemisphere] = rdm
            mdss[hemisphere] = mds_out

    return rdms, mdss


# =========================================================================
# MAIN EXECUTION
# =========================================================================

if __name__ == "__main__":
    # Example usage (commented out)
    pass
    
    # Example usage:
    # for subj_id in range(1, 9):
    #     set_to_take = f"subj_{subj_id:02d}"
    #     config = load_config("config.yaml")
    #     mask_values = list(config.analysis.rois_to_analyze.values())
    #     
    #     for mask_value in mask_values:
    #         try:
    #             n_samples = 3
    #             for sample_v in range(0, n_samples):
    #                 create_rdm(
    #                     config,
    #                     [subj_id],
    #                     mask_value,
    #                     set_to_take,
    #                     2.0,
    #                     mode="single",
    #                     sample_to_pick=sample_v,
    #                     randomization=True,
    #                 )
    #         except:
    #             n_samples = 2
    #             for sample_v in range(0, n_samples):
    #                 create_rdm(
    #                     config,
    #                     [subj_id],
    #                     mask_value,
    #                     set_to_take,
    #                     2.0,
    #                     mode="single",
    #                     sample_to_pick=sample_v,
    #                     randomization=True,
    #                 )