import glob
import os

import pandas as pd
import numpy as np
import nibabel as nib

from utils.config import Configuration

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def retrieve_roi_mask(config: Configuration, subject: int, subj_to_check: str):
    mask_path_lh = os.path.join(
        config.t_test_roi_dir, subj_to_check, f"lh.subj{subject:02d}.cleanedrois.mgz"
    )
    mask_path_rh = os.path.join(
        config.t_test_roi_dir, subj_to_check, f"rh.subj{subject:02d}.cleanedrois.mgz"
    )
    logging.info(f"Loading mask from\n{mask_path_lh}\n{mask_path_rh}")

    mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()

    mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()

    mask = np.concatenate((mask_lh, mask_rh)).astype(int)

    return mask


def filter_roi_mask(roi_value: int, mask: np.ndarray):
    indices = np.where(mask == roi_value)
    return indices


def retrieve_stacked_betas(
    config: Configuration,
    subj: int,
    mode: str,
    sample: int,
    label_subset_name: str = None,
    subj_to_check="shared",
):
    assert mode in ["averaged", "single"]

    if label_subset_name is None:
        label_subset_name = config.subset_animate_face_final

    logging.info(f"Loading from {label_subset_name}")

    betas_dir = config.image_betas_dir

    data = []
    image_ids = []

    set_excel_path = os.path.join(
        config.excel_files_target_dir, subj_to_check, label_subset_name
    )

    subset = pd.read_excel(set_excel_path)
    filenames = subset["cocoId"].to_list()
    filenames = [f"{e:012d}" for e in filenames]

    for entry in filenames:
        image_ids.append(entry)

        npy_files = sorted(
            glob.glob(
                os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
            )
        )

        if len(npy_files) == 0:
            raise ValueError(f"No data found for {entry}!")

        if mode == "single":
            npy_file = npy_files[sample]
            extracted_sample = np.load(npy_file)

        elif mode == "averaged":
            # leave one out for test set
            samples = [np.load(npy_file) for npy_file in npy_files[:-1]]
            samples = np.stack(samples)
            extracted_sample = np.mean(samples, axis=0)

        data.append(extracted_sample)

    return np.stack(data), image_ids


def retrieve_stacked_betas_test(
    config: Configuration,
    subj: int,
    subj_to_check="shared",
    label_subset_name: str = None,
):
    betas_dir = config.image_betas_dir

    data = []

    if label_subset_name is None:
        label_subset_name = config.subset_animate_face_final

    positive_set_excel_path = os.path.join(
        config.excel_files_target_dir, subj_to_check, label_subset_name
    )

    positive_subset = pd.read_excel(positive_set_excel_path)
    positive_filenames = positive_subset["cocoId"].to_list()
    positive_filenames = [f"{e:012d}" for e in positive_filenames]

    for entry in positive_filenames:

        npy_files = sorted(
            glob.glob(os.path.join(betas_dir, entry, f"subj_{subj:02d}", "*.npy"))
        )

        if len(npy_files) == 0:
            raise ValueError(f"No data found for {entry}!")

        if len(npy_files) <= 1:
            logging.warning(
                f"Leave-one-out validation not possible! Image {entry} has {len(npy_files)} samples"
            )

        # leave one out for test set
        extracted_sample = np.load(npy_files[-1])
        data.append(extracted_sample)

    return np.stack(data)


# from utils.config import load_config, Configuration

# config = load_config("config.yaml")
# retrieve_roi_mask(config, 1)
