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


def logging_message(step: int, message: str):
    return f"Pipeline (S={step}): {message}"


def subjects_list_unifier(subjects_list: list, reduce_shared: bool):
    # check validity
    assert not (
        set(subjects_list) - {"shared", 1, 2, 3, 4, 5, 6, 7, 8}
    ), f"Invalid subjects specification: {subjects_list}"

    if reduce_shared:
        if "shared" in subjects_list:
            return list(range(1, 8 + 1))
        else:
            return subjects_list
    else:
        return subjects_list


def retrieve_roi_mask(
    config: Configuration,
    subject: int,
    subj_to_check: str,
    take_cleaned_roi: bool,
    sub_filename: str = "cleanedrois",
):
    logging.info(f"Loading ROI mask {take_cleaned_roi=}")
    if take_cleaned_roi:
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
        mask_path_lh = os.path.join(
            config.directories.t_test_roi_dir,
            subj_to_check,
            f"lh.subj{subject:02d}.testrois.mgz",
        )
        mask_path_rh = os.path.join(
            config.directories.t_test_roi_dir,
            subj_to_check,
            f"rh.subj{subject:02d}.testrois.mgz",
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
    only_face_set=True,
    randomization: bool = False,
    augment_shared_set: bool = False,
):
    assert mode in ["averaged", "single", "multiple"]

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

    # having all of the array loaded at once results in memory pressure
    # reduce to needed subset!

    tsv_data = pd.read_csv(tsv_path, sep="\t")

    if label_subset_name is None:
        label_subset_name = config.dataset_creation.subset_animate_face_final

    logging.info(f"Loading from {label_subset_name}")

    betas_dir = config.directories.image_betas_dir

    data = []
    image_ids = []

    set_excel_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_check, label_subset_name
    )

    if augment_shared_set:
        logging.info(f"Augmenting with shared set")
        shared_path = os.path.join(config.directories.excel_files_target_dir, "shared", label_subset_name)
        subset = pd.concat([pd.read_excel(set_excel_path), pd.read_excel(shared_path)], ignore_index=True)

    else:
        logging.info(f"Not augmenting with shared set")
        subset = pd.read_excel(set_excel_path)


    if only_face_set:
        pass
    else:
        subset = pd.concat(
            [
                subset,
                pd.read_excel(
                    os.path.join(
                        config.directories.excel_files_target_dir,
                        subj_to_check,
                        config.dataset_creation.subset_animate_non_face_final,
                    )
                ),
            ]
        )

    filenames_coco_number = subset["cocoId"].to_list()
    filenames_nsd_number = subset["nsdId"].to_list()
    filenames = [f"{e:012d}" for e in filenames_coco_number]

    mds_mapping = []

    for mds_index, entry in enumerate(filenames):
        image_ids.append(entry)

        tsv_matches = tsv_data.loc[
            tsv_data["73KID"] == filenames_nsd_number[mds_index] + 1
        ].index.tolist()

        npy_files = sorted(
            glob.glob(
                os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
            )
        )

        if randomization:
            seed = subj * 10000 + int(entry)
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(npy_files))
            npy_files = [npy_files[i] for i in perm]

        for npy_file in npy_files:

            base_name = os.path.basename(npy_file)
            # full.betas_*.npy
            extracted_trial_id = int(base_name[11:-4])
            assert (
                extracted_trial_id in tsv_matches
            ), "Something is wrong... the indices are not matching!"

        if len(npy_files) == 0:
            raise ValueError(f"No data found for {entry}!")

        if mode == "single":
            npy_file = npy_files[sample]
            extracted_sample = np.load(npy_file)

            mds_mapping.append(mds_index)

        elif mode == "multiple":
            npy_files = npy_files[:-1]

            samples = []
            for i in range(len(npy_files)):
                sample = np.load(npy_files[i])
                data.append(sample)
                mds_mapping.append(mds_index)

        elif mode == "averaged":
            # leave one out for test set
            samples = [np.load(npy_file) for npy_file in npy_files[:-1]]
            samples = np.stack(samples)
            extracted_sample = np.mean(samples, axis=0)

            mds_mapping.append(mds_index)

        data.append(extracted_sample)

    return np.stack(data), image_ids, mds_mapping


def retrieve_stacked_betas_test(
    config: Configuration,
    subj: int,
    subj_to_check="shared",
    label_subset_name: str = None,
    augment_shared_set: bool = False,
):
    betas_dir = config.directories.image_betas_dir

    data = []

    if label_subset_name is None:
        label_subset_name = config.dataset_creation.subset_animate_face_final

    positive_set_excel_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_check, label_subset_name
    )

    

    positive_subset = pd.read_excel(positive_set_excel_path)

    if augment_shared_set:
        logging.info(f"Augmenting with shared set")
        shared_path = os.path.join(config.directories.excel_files_target_dir, "shared", label_subset_name)
        positive_subset = pd.concat([positive_subset, pd.read_excel(shared_path)], ignore_index=True)

    else:
        logging.info(f"Not augmenting with shared set")

    positive_filenames = positive_subset["cocoId"].to_list()
    filenames_nsd_number = positive_subset["nsdId"].to_list()

    positive_filenames = [f"{e:012d}" for e in positive_filenames]

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

    # having all of the array loaded at once results in memory pressure
    # reduce to needed subset!

    tsv_data = pd.read_csv(tsv_path, sep="\t")

    for entry_index, entry in enumerate(positive_filenames):
        tsv_matches = tsv_data.loc[
            tsv_data["73KID"] == filenames_nsd_number[entry_index] + 1
        ].index.tolist()

        npy_files = sorted(
            glob.glob(
                os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
            )
        )

        for npy_file in npy_files:

            base_name = os.path.basename(npy_file)
            # full.betas_*.npy
            extracted_trial_id = int(base_name[11:-4])
            assert (
                extracted_trial_id in tsv_matches
            ), "Something is wrong... the indices are not matching!"

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


import os
import pandas as pd
from utils.config import Configuration

def load_negative_set(
    config: Configuration,
    subj_to_check: str,
    label_subset_name: str = None,
    augment_shared_set: bool = False,
    remove_animate_face: bool = False,
):
    """
    Load the negative set for a subject, with optional augmentation and filtering.

    Parameters
    ----------
    config : Configuration
        Your config object (must have .directories.excel_files_target_dir and
        .dataset_creation.subset_animate_non_face_final, .dataset_creation.shared_animate_non_face_final).
    subj : int
        Subject ID to load.
    subj_to_check : str
        (unused here, but kept for compatibility).
    label_subset_name : str, optional
        Override the per‐subject filename for the negative‐set sheet.
    augment_shared_set : bool
        If True, also load and concatenate the shared negative‐set sheet.
    remove_animate_face : bool
        If True, drop rows where label == "animate_face".
    """
    # 1) Load per‐subject sheet
    subset_fname = config.dataset_creation.subset_animate_non_face_final
    subj_dir = os.path.join(
        config.directories.excel_files_target_dir,
        subj_to_check
    )
    personal_path = os.path.join(subj_dir, subset_fname)
    neg_df = pd.read_excel(personal_path)

    # 2) Optionally augment with shared sheet
    if augment_shared_set:
        shared_fname = "shared"
        shared_path = os.path.join(config.directories.excel_files_target_dir, shared_fname, subset_fname)
        shared_df = pd.read_excel(shared_path)
        neg_df = pd.concat([neg_df, shared_df], ignore_index=True)

    # 3) Ensure there's a 'label' column
    if 'label' not in neg_df.columns:
        raise ValueError(f"No 'label' column in {personal_path!r}; found columns {list(neg_df.columns)}")

    # 4) Drop generic 'animate' rows
    neg_df = neg_df[neg_df['label'] != 'animate']

    # 5) Check for ONLY the allowed labels
    allowed = {'animate_persons', 'animate_face', 'animate_animal'}
    found = set(neg_df['label'].unique())
    bad = found - allowed
    if bad:
        raise ValueError(
            f"Found unsupported labels {bad}. "
            f"Labels must be one of {allowed} (or 'animate', which is auto‐dropped)."
        )

    # 6) Optionally remove animate_face
    if remove_animate_face:
        neg_df = neg_df[neg_df['label'] != 'animate_face']

    # 7) Ensure we still have samples
    if neg_df.empty:
        raise ValueError("After filtering, no negative examples remain! Aborting.")

    # 8) Extract bare filenames (no path, no extension)
    filenames = (
        neg_df['file_name']
          .apply(lambda p: os.path.splitext(os.path.basename(p))[0])
          .tolist()
    )

    # 9) Return a single‐element list of (name, file_list)
    name = label_subset_name or 'animate'
    return [(name, filenames)]
