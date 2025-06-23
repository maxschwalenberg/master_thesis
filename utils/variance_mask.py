from utils.config import Configuration, load_config
import os
import nibabel as nib
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import math

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def cartesian_to_polar_normalized(x, y):
    # radius
    r = math.hypot(x, y)
    # atan2 returns radians in (-π, π]; convert to degrees and wrap to [0,360)
    theta_deg = math.degrees(math.atan2(y, x)) % 360
    # normalize to [0,1)
    theta_norm = theta_deg / 360.0
    return r, theta_deg, theta_norm


import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def create_variance_mask(config: Configuration, subj_to_pick_shared: bool, subj_to_pick: int):
    hemis = ["lh", "rh"]
    subset = "shared" if subj_to_pick_shared else f"subj_{subj_to_pick:02d}"
    mask_dir = config.directories.t_test_roi_dir
    gaussian_results_dir = os.path.join(
        config.directories.gaussian_fit_results_dir,
        f"subj_{subj_to_pick:02d}",
        f"subj{subj_to_pick:02d}",
    )

    results_df = load_excel_files(gaussian_results_dir)
    # … build var_train_lookup, pos_x_lookup, pos_y_lookup, sigma_lookup as before …
    results_df = results_df[results_df["mds_hemi"] == "both"]

    # Create a dictionary mapping original_index to var_train for O(1) lookups
    var_train_lookup = dict(
        zip(results_df["original_index"], results_df["var_train"])
    )

    pos_x_lookup = dict(zip(results_df["original_index"], results_df["x0"]))
    pos_y_lookup = dict(zip(results_df["original_index"], results_df["y0"]))

    sigma_lookup = dict(zip(results_df["original_index"], results_df["sigma"]))

    vertex_offset = 0
    for hemi in hemis:
        mask_path = os.path.join(
            mask_dir, subset, f"{hemi}.subj{subj_to_pick:02d}.testrois.mgz"
        )
        maskdata = nib.load(mask_path).get_fdata().squeeze()

        # Prepare containers and output paths in one place
        map_types = {
            "variance": np.zeros_like(maskdata),
            "angle":    np.zeros_like(maskdata),
            "radius":   np.zeros_like(maskdata),
            "sigma":    np.zeros_like(maskdata),
        }
        out_paths = {
            key: os.path.join(
                config.nsd_data.freesurfer_dir,
                f"subj{subj_to_pick:02d}",
                "label",
                f"{hemi}.fits_{key}.mgz" if key != "variance" else f"{hemi}.variance_map.mgz"
            )
            for key in map_types
        }

        # Fill them
        for i in tqdm(range(maskdata.shape[0]), desc=f"Processing {hemi}"):
            idx = i + vertex_offset
            vt = var_train_lookup.get(idx)
            x0 = pos_x_lookup.get(idx)
            y0 = pos_y_lookup.get(idx)
            sig = sigma_lookup.get(idx)

            if vt is not None:
                map_types["variance"][i] = vt
            if x0 is not None and y0 is not None and sig is not None:
                r, theta, tnorm = cartesian_to_polar_normalized(x0, y0)
                map_types["angle"][i]  = tnorm
                map_types["radius"][i] = r
                # clamp sigma into [0,2], then normalize
                map_types["sigma"][i]  = max(0, min(sig, 2)) / 2

        # Save all four in a loop
        for key, arr in map_types.items():
            img = nib.Nifti1Image(
                np.expand_dims(arr, axis=(1, 2)), 
                np.eye(4)
            )
            nib.save(img, out_paths[key])
            logging.info(f"Saved {key} map to {out_paths[key]}")

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
