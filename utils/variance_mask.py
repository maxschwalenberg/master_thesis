from utils.config import Configuration, load_config
import os
import nibabel as nib
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_variance_mask(
    config: Configuration,
    subj_to_pick_shared: bool,
    subj_to_pick: int,
):
    """
    Creates variance mask files based on Gaussian fit results.

    Args:
        config: Configuration object containing necessary paths
        subj_to_pick_shared: Boolean indicating whether to use shared subject data
        subj_to_pick: Subject ID number to process
    """
    hemis = ["lh", "rh"]
    mask_dir = config.directories.t_test_roi_dir

    # Calculate the directory path once
    subset = "shared" if subj_to_pick_shared else f"subj_{subj_to_pick:02d}"

    # Load all Excel files in one pass
    gaussian_results_dir = os.path.join(
        config.directories.gaussian_fit_results_dir,
        f"subj_{subj_to_pick:02d}",
        f"subj{subj_to_pick:02d}",
    )

    # Load all Excel results into a single DataFrame
    results_df = load_excel_files(gaussian_results_dir)

    # Create a lookup dictionary for faster access
    var_train_lookup = {}
    if not results_df.empty:
        # Create a dictionary mapping original_index to var_train for O(1) lookups
        var_train_lookup = dict(
            zip(results_df["original_index"], results_df["var_train"])
        )

    # Process each hemisphere
    vertex_offset = 0
    for hemi in hemis:
        # Construct file paths
        maskdata_in_file = os.path.join(
            mask_dir, subset, f"{hemi}.subj{subj_to_pick:02d}.testrois.mgz"
        )
        data_out_file = os.path.join(
            config.nsd_data.freesurfer_dir,
            f"subj{subj_to_pick:02d}",
            "label",
            f"{hemi}.variance_map.mgz",
        )

        # Skip if output already exists
        if os.path.exists(data_out_file):
            logging.info(f"File {data_out_file} already exists, skipping")
            # Update vertex offset for next hemisphere
            if hemi == "lh":
                vertex_offset = (
                    nib.load(maskdata_in_file).get_fdata().squeeze().shape[0]
                )
            continue

        # Load mask data
        maskdata = nib.load(maskdata_in_file).get_fdata().squeeze()

        # Pre-allocate output array
        data_out = np.zeros(maskdata.shape)

        # Process vertices
        for i in tqdm(range(maskdata.shape[0]), desc=f"Processing {hemi}"):
            index = i + vertex_offset
            var_train_value = var_train_lookup.get(index)
            if var_train_value is not None:
                data_out[i] = var_train_value

        # Save the output
        img = nib.Nifti1Image(np.expand_dims(data_out, axis=(1, 2)), np.eye(4))
        nib.loadsave.save(img, data_out_file)
        logging.info(f"Saved to {data_out_file}")

        # Update vertex offset for right hemisphere
        if hemi == "lh":
            vertex_offset = maskdata.shape[0]


def load_excel_files(directory):
    """
    Loads all Excel files from a directory into a single DataFrame.

    Args:
        directory: Path to directory containing Excel files

    Returns:
        pandas.DataFrame: Combined DataFrame from all Excel files
    """
    dfs = []

    # Check if directory exists
    if not os.path.exists(directory):
        logging.warning(f"Directory {directory} does not exist")
        return pd.DataFrame()

    # Find all Excel files
    excel_files = [f for f in os.listdir(directory) if f.endswith((".xlsx", ".xls"))]

    if not excel_files:
        logging.warning(f"No Excel files found in {directory}")
        return pd.DataFrame()

    # Load each file
    for file in excel_files:
        file_path = os.path.join(directory, file)
        try:
            df = pd.read_excel(file_path)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")

    # Concatenate all dataframes
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    config = load_config("config.yaml")
    create_variance_mask(config, False, 1)
