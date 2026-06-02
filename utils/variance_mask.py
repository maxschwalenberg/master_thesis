"""
Variance Mapping Module

This module creates surface-based variance maps from Gaussian fitting results.
It processes fitted Gaussian parameters to generate visualizable surface maps
for variance explained, spatial positions (angle/radius), sigma values, and slopes.
The module supports coordinate transformations and parameter normalization for
FreeSurfer surface visualization.
"""

# Standard library imports
import os
import math
import logging

# Third-party imports
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# Local imports
from utils.config import Configuration, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# COORDINATE TRANSFORMATION FUNCTIONS
# =========================================================================


def cartesian_to_polar_normalized(x: float, y: float) -> tuple[float, float, float]:
    """
    Convert Cartesian coordinates to polar coordinates with normalization.

    Transforms (x, y) coordinates to polar representation with radius and
    normalized angle suitable for surface visualization.

    Args:
        x: X-coordinate in Cartesian space
        y: Y-coordinate in Cartesian space

    Returns:
        tuple: (radius, theta_degrees, theta_normalized) where:
            - radius: Euclidean distance from origin
            - theta_degrees: Angle in degrees [0, 360)
            - theta_normalized: Angle normalized to [0, 1)
    """
    # Compute radius using Euclidean distance
    r = math.hypot(x, y)

    # Convert to degrees and wrap to [0, 360) range
    # atan2 returns radians in (-π, π]; convert and normalize
    theta_deg = math.degrees(math.atan2(y, x)) % 360

    # Normalize angle to [0, 1) for surface visualization
    theta_norm = theta_deg / 360.0

    return r, theta_deg, theta_norm


# =========================================================================
# DATA LOADING FUNCTIONS
# =========================================================================


def load_excel_files(directory: str) -> pd.DataFrame:
    """
    Load and concatenate all Excel files from a directory into a single DataFrame.

    This function searches for Excel files (.xlsx, .xls) in the specified directory,
    loads each file, and combines them into a unified DataFrame for analysis.

    Args:
        directory: Path to directory containing Excel files

    Returns:
        pd.DataFrame: Combined DataFrame from all Excel files, empty if none found
    """
    logging.info(f"Loading Excel files from directory: {directory}")
    dfs = []

    # Validate directory existence
    if not os.path.exists(directory):
        logging.warning(f"Directory {directory} does not exist")
        return pd.DataFrame()

    # Find all Excel files in directory
    excel_files = [f for f in os.listdir(directory) if f.endswith((".xlsx", ".xls"))]

    if not excel_files:
        logging.warning(f"No Excel files found in {directory}")
        return pd.DataFrame()

    logging.info(f"Found {len(excel_files)} Excel files: {excel_files}")

    # Load each Excel file
    for file in excel_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_excel(file_path)
            dfs.append(df)
            logging.debug(f"Successfully loaded {file_path} with {len(df)} rows")
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    # Concatenate all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logging.info(
            f"Combined {len(dfs)} files into DataFrame with {len(combined_df)} total rows"
        )
        return combined_df
    else:
        logging.warning("No Excel files could be loaded successfully")
        return pd.DataFrame()


# =========================================================================
# VARIANCE MAP CREATION FUNCTIONS
# =========================================================================


def create_variance_mask(
    config: Configuration, subj_to_pick_shared: bool, subj_to_pick: int
):
    """
    Create surface-based variance maps from Gaussian fitting results.

    This function processes fitted Gaussian parameters to generate FreeSurfer-compatible
    surface maps for visualization. It creates maps for variance explained, spatial
    position (angle/radius), sigma values, and slope parameters.

    Args:
        config: Configuration object containing directory paths
        subj_to_pick_shared: Whether to use shared subject data
        subj_to_pick: Subject ID to process
    """
    logging.info(f"Creating variance masks for subject {subj_to_pick}")
    logging.info(f"Using shared data: {subj_to_pick_shared}")

    # Define processing parameters
    hemis = ["lh", "rh"]
    subset = "shared" if subj_to_pick_shared else f"subj_{subj_to_pick:02d}"
    mask_dir = config.directories.t_test_roi_dir

    # Define paths to Gaussian fitting results
    gaussian_results_dir = os.path.join(
        "data/gaussian_results",
        "final_run",
        f"subj_{subj_to_pick:02d}",
        f"subj{subj_to_pick:02d}",
    )

    gaussian_results_dir_slope = os.path.join(
        "data/gaussian_results",
        "final_run",
        f"subj_{subj_to_pick:02d}",
        f"subj{subj_to_pick:02d}",
    )

    # Load Gaussian fitting results
    logging.info("Loading Gaussian fitting results...")
    results_df = load_excel_files(gaussian_results_dir)
    results_df_slope = load_excel_files(gaussian_results_dir_slope)

    if results_df.empty or results_df_slope.empty:
        logging.error("Failed to load Gaussian fitting results")
        return

    # Filter to use only 'both' hemisphere fits for consistency
    results_df = results_df[results_df["mds_hemi"] == "both"]
    results_df_slope = results_df_slope[results_df_slope["mds_hemi"] == "both"]

    logging.info(f"Loaded {len(results_df)} variance records")
    logging.info(f"Loaded {len(results_df_slope)} slope records")

    # Create lookup dictionaries for O(1) parameter access
    logging.info("Creating parameter lookup dictionaries...")
    var_train_lookup = dict(zip(results_df["original_index"], results_df["var_train"]))
    pos_x_lookup = dict(zip(results_df["original_index"], results_df["x0"]))
    pos_y_lookup = dict(zip(results_df["original_index"], results_df["y0"]))
    sigma_lookup = dict(zip(results_df["original_index"], results_df["sigma"]))
    slope_lookup = dict(
        zip(results_df_slope["original_index"], results_df_slope["slope"])
    )

    # Validate consistency between datasets
    assert len(slope_lookup) == len(
        sigma_lookup
    ), f"Mismatch in lookup table sizes: slope={len(slope_lookup)}, sigma={len(sigma_lookup)}"

    logging.info(f"Created lookup tables with {len(var_train_lookup)} entries")

    # Process each hemisphere
    vertex_offset = 0
    for hemi in hemis:
        logging.info(f"Processing hemisphere: {hemi}")

        # Load hemisphere mask
        mask_path = os.path.join(
            mask_dir, subset, f"{hemi}.subj{subj_to_pick:02d}.final.mgz"
        )

        if not os.path.exists(mask_path):
            logging.error(f"Mask file not found: {mask_path}")
            continue

        maskdata = nib.load(mask_path).get_fdata().squeeze()
        logging.info(f"Loaded {hemi} mask with {maskdata.shape[0]} vertices")

        # Initialize output maps
        map_types = {
            "variance": np.zeros_like(maskdata),
            "angle": np.zeros_like(maskdata),
            "radius": np.zeros_like(maskdata),
            "sigma": np.zeros_like(maskdata),
            "slope": np.zeros_like(maskdata),
        }

        # Define output paths for each map type
        out_paths = {
            key: os.path.join(
                config.nsd_data.freesurfer_dir,
                f"subj{subj_to_pick:02d}",
                "label",
                (
                    f"{hemi}.fits_{key}.mgz"
                    if key != "variance"
                    else f"{hemi}.variance_map.mgz"
                ),
            )
            for key in map_types
        }

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(list(out_paths.values())[0])
        os.makedirs(output_dir, exist_ok=True)

        # Process each vertex in the hemisphere
        for i in tqdm(range(maskdata.shape[0]), desc=f"Processing {hemi} vertices"):
            # Calculate global vertex index (accounting for hemisphere offset)
            idx = i + vertex_offset

            # Retrieve parameters for this vertex
            vt = var_train_lookup.get(idx)  # Variance explained
            x0 = pos_x_lookup.get(idx)  # X position
            y0 = pos_y_lookup.get(idx)  # Y position
            sig = sigma_lookup.get(idx)  # Sigma parameter
            slope = slope_lookup.get(idx)  # Slope parameter

            # Validate slope and sigma consistency
            if idx in sigma_lookup:
                assert idx in slope_lookup, f"Missing slope for vertex {idx}"
                assert slope is not None, f"None slope for vertex {idx}"

            # Process variance explained
            if vt is not None:
                map_types["variance"][i] = vt

            # Process spatial position (convert to polar coordinates)
            if x0 is not None and y0 is not None:
                r, theta_deg, theta_norm = cartesian_to_polar_normalized(x0, y0)
                map_types["angle"][i] = theta_norm  # Normalized angle [0,1)
                map_types["radius"][i] = r  # Radius from origin

            # Process sigma parameter (clamp and normalize to [0,1])
            if sig is not None:
                # Clamp sigma to reasonable range [0,2] and normalize
                clamped_sigma = max(0, min(sig, 2))
                map_types["sigma"][i] = clamped_sigma / 2

            # Process slope parameter with robust normalization
            if slope is not None:
                # Clamp extreme slopes to ±5 range
                if np.abs(slope) > 5:
                    slope = np.sign(slope) * 5

                # Transform slope to positive range for visualization
                # Maps [-5, 5] to [0.1, 10.1]
                processed_slope = slope + 5.1
                map_types["slope"][i] = processed_slope

                # Validate transformation bounds
                normalized_slope = max(-10, min(slope, 10)) / 10
                assert (
                    -1 <= normalized_slope <= 1
                ), f"Slope normalization failed: {normalized_slope}"

        # Save all maps to FreeSurfer-compatible format
        for key, arr in map_types.items():
            # Create NIfTI image with proper dimensions for FreeSurfer
            img = nib.Nifti1Image(
                np.expand_dims(arr, axis=(1, 2)),  # Add singleton dimensions
                np.eye(4),  # Identity affine matrix
            )
            nib.save(img, out_paths[key])
            logging.info(f"Saved {key} map to {out_paths[key]}")

        # Update vertex offset for next hemisphere
        if hemi == "lh":
            vertex_offset = maskdata.shape[0]

    logging.info(f"Variance mask creation completed for subject {subj_to_pick}")


# =========================================================================
# MAIN EXECUTION
# =========================================================================


def main():
    """
    Main execution function for variance mapping.

    Loads configuration and creates variance masks for a test subject.
    This can be modified to process multiple subjects or different parameters.
    """
    logging.info("Starting variance mapping pipeline")

    # Load configuration
    config = load_config("config.yaml")

    # Process subject 1 as example
    # Modify these parameters as needed:
    # - subj_to_pick_shared: False for subject-specific data, True for shared data
    # - subj_to_pick: Subject ID to process
    create_variance_mask(config, subj_to_pick_shared=False, subj_to_pick=1)

    logging.info("Variance mapping pipeline completed")


if __name__ == "__main__":
    main()
