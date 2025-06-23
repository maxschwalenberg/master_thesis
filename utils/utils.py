import glob
import os

import pandas as pd
import numpy as np
import nibabel as nib

from utils.config import Configuration
from t_testing.clean_roi_mask import modify_mask_with_ttest

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



def retrieve_roi_mask_extended(
    config: Configuration,
    subject: int,
    subj_to_check: str,
    take_cleaned_roi: bool,
    sub_filename: str = "cleanedrois",
    t_test_threshold: float = 2.0

):
    logging.info(f"Loading ROI mask {take_cleaned_roi=}")
    if take_cleaned_roi:
        modify_mask_with_ttest(config, t_test_threshold, False, subject, sub_filename=sub_filename)

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

    return mask, mask_lh.size


def filter_roi_mask(roi_value: int, mask: np.ndarray):
    indices = np.where(mask == roi_value)
    return indices


# def retrieve_stacked_betas(
#     config: Configuration,
#     subj: int,
#     mode: str,
#     sample: int,
#     label_subset_name: str = None,
#     subj_to_check="shared",
#     only_face_set=True,
#     randomization: bool = False,
#     augment_shared_set: bool = False,
# ):
#     assert mode in ["averaged", "single", "multiple"]

#     nsd_dir = os.path.join(
#         config.nsd_project.nsd_data_dir, config.nsd_project.nsd_subdir
#     )
#     tsv_path = os.path.join(
#         nsd_dir,
#         config.nsd_project.label_subdir,
#         "ppdata",
#         f"subj{subj:02d}",
#         "behav",
#         "responses.tsv",
#     )

#     # having all of the array loaded at once results in memory pressure
#     # reduce to needed subset!

#     tsv_data = pd.read_csv(tsv_path, sep="\t")

#     if label_subset_name is None:
#         label_subset_name = config.dataset_creation.subset_animate_face_final

#     logging.info(f"Loading from {label_subset_name}")

#     betas_dir = config.directories.image_betas_dir

#     data = []
#     image_ids = []

#     set_excel_path = os.path.join(
#         config.directories.excel_files_target_dir, subj_to_check, label_subset_name
#     )

#     if augment_shared_set:
#         logging.info(f"Augmenting with shared set")
#         shared_path = os.path.join(config.directories.excel_files_target_dir, "shared", label_subset_name)
#         shared_df = pd.read_excel(shared_path)


#         subset = pd.concat([pd.read_excel(set_excel_path), shared_df], ignore_index=True)

#         print(f"Label distribution\nShared: {len(shared_df)} - Total: {len(subset)}")

#     else:
#         logging.info(f"Not augmenting with shared set")
#         subset = pd.read_excel(set_excel_path)


#     if only_face_set:
#         pass
#     else:
#         subset = pd.concat(
#             [
#                 subset,
#                 pd.read_excel(
#                     os.path.join(
#                         config.directories.excel_files_target_dir,
#                         subj_to_check,
#                         config.dataset_creation.subset_animate_non_face_final,
#                     )
#                 ),
#             ]
#         )

#     filenames_coco_number = subset["cocoId"].to_list()
#     filenames_nsd_number = subset["nsdId"].to_list()
#     filenames = [f"{e:012d}" for e in filenames_coco_number]

#     mds_mapping = []

#     for mds_index, entry in enumerate(filenames):
#         cocoId = int(entry.lstrip())
#         image_ids.append(entry)

#         tsv_matches = tsv_data.loc[
#             tsv_data["73KID"] == filenames_nsd_number[mds_index] + 1
#         ].index.tolist()

#         npy_files = sorted(
#             glob.glob(
#                 os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
#             )
#         )

#         npy_files_cleaned = npy_files.copy()

#         if randomization:
#             seed = subj * 10000 + int(entry)
#             rng = np.random.RandomState(seed)
#             perm = rng.permutation(len(npy_files))
#             npy_files = [npy_files[i] for i in perm]

#         for npy_file in npy_files:

#             base_name = os.path.basename(npy_file)
#             # full.betas_*.npy
#             extracted_trial_id = int(base_name[11:-4])
#             assert (
#                 extracted_trial_id in tsv_matches
#             ), "Something is wrong... the indices are not matching!"


#         n_usable_files = len(npy_files)
#         # faulty samples check!
#         faulty = 0 
#         for npy_index, npy_file_path in reversed(list(enumerate(npy_files))):
#             sample = np.load(npy_file_path)

#             if np.isnan(sample).any():
#                 print(f"{npy_file_path}: removing due to NaN")
#                 del npy_files_cleaned[npy_index]
#                 faulty += 1
        
#         n_usable_files -= faulty

#         if n_usable_files == 0:
#             if cocoId in shared_df["cocoId"].tolist():
#                 continue
#             else:
#                 raise ValueError(f"No data found for {entry}!")
        
#         if n_usable_files<2:
#             if cocoId in shared_df["cocoId"].tolist():
#                 continue
#             else:
#                 raise ValueError(f"Not 3 correct samples for {entry}!")
            
        



#         if mode == "single":
#             npy_file = npy_files_cleaned[sample]
#             extracted_sample = np.load(npy_file)

#             mds_mapping.append(mds_index)

#         elif mode == "multiple":
#             npy_files_cleaned = npy_files_cleaned[:-1]

#             samples = []
#             for i in range(len(npy_files_cleaned)):
#                 sample = np.load(npy_files_cleaned[i])
#                 data.append(sample)
#                 mds_mapping.append(mds_index)

#         elif mode == "averaged":
#             # leave one out for test set
#             samples = [np.load(npy_file) for npy_file in npy_files_cleaned[:-1]]
#             samples = np.stack(samples)
#             extracted_sample = np.mean(samples, axis=0)

#             mds_mapping.append(mds_index)

#         data.append(extracted_sample)

#     print(f"Final data shape = {np.array(data).shape}")

#     return np.stack(data), image_ids, mds_mapping


# def retrieve_stacked_betas_test(
#     config: Configuration,
#     subj: int,
#     subj_to_check="shared",
#     label_subset_name: str = None,
#     augment_shared_set: bool = False,
# ):
#     betas_dir = config.directories.image_betas_dir

#     data = []

#     if label_subset_name is None:
#         label_subset_name = config.dataset_creation.subset_animate_face_final

#     positive_set_excel_path = os.path.join(
#         config.directories.excel_files_target_dir, subj_to_check, label_subset_name
#     )

    

#     positive_subset = pd.read_excel(positive_set_excel_path)

#     if augment_shared_set:
#         logging.info(f"Augmenting with shared set")
#         shared_path = os.path.join(config.directories.excel_files_target_dir, "shared", label_subset_name)
#         shared_df = pd.read_excel(shared_path)
#         # shared_path = os.path.join(config.directories.excel_files_target_dir, "shared", label_subset_name)
#         positive_subset = pd.concat([positive_subset, shared_df], ignore_index=True)

#     else:
#         logging.info(f"Not augmenting with shared set")

#     positive_filenames = positive_subset["cocoId"].to_list()
#     filenames_nsd_number = positive_subset["nsdId"].to_list()

#     positive_filenames = [f"{e:012d}" for e in positive_filenames]

#     nsd_dir = os.path.join(
#         config.nsd_project.nsd_data_dir, config.nsd_project.nsd_subdir
#     )
#     tsv_path = os.path.join(
#         nsd_dir,
#         config.nsd_project.label_subdir,
#         "ppdata",
#         f"subj{subj:02d}",
#         "behav",
#         "responses.tsv",
#     )

#     # having all of the array loaded at once results in memory pressure
#     # reduce to needed subset!

#     tsv_data = pd.read_csv(tsv_path, sep="\t")

#     for entry_index, entry in enumerate(positive_filenames):
#         cocoId = int(entry.lstrip())


#         tsv_matches = tsv_data.loc[
#             tsv_data["73KID"] == filenames_nsd_number[entry_index] + 1
#         ].index.tolist()

#         npy_files = sorted(
#             glob.glob(
#                 os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
#             )
#         )

#         for npy_file in npy_files:

#             base_name = os.path.basename(npy_file)
#             # full.betas_*.npy
#             extracted_trial_id = int(base_name[11:-4])
#             assert (
#                 extracted_trial_id in tsv_matches
#             ), "Something is wrong... the indices are not matching!"

#         npy_files = sorted(
#             glob.glob(os.path.join(betas_dir, entry, f"subj_{subj:02d}", "*.npy"))
#         )

#         if len(npy_files) == 0:
#             if cocoId in shared_df["cocoId"].tolist():
#                 continue
#             else:
#                 raise ValueError(f"No data found for {entry}!")
        
#         if len(npy_files)<2:
#             if cocoId in shared_df["cocoId"].tolist():
#                 continue
#             else:
#                 raise ValueError(f"Not 3 correct samples for {entry}!")

#         # if len(npy_files) == 0:
#         #     raise ValueError(f"No data found for {entry}!")

#         # if len(npy_files) <= 1:
#         #     logging.warning(
#         #         f"Leave-one-out validation not possible! Image {entry} has {len(npy_files)} samples"
#         #     )

#         # leave one out for test set
#         extracted_sample = np.load(npy_files[-1])
#         data.append(extracted_sample)

#     return np.stack(data)


def retrieve_stacked_betas(
    config: Configuration,
    subj: int,
    train_mode: str,
    sample: int,
    test: bool = False,                # if True, behaves like your old retrieve_stacked_betas_test
    subj_to_check="shared",
    label_subset_name: str = None,
    only_face_set=True,
    randomization: bool = False,
    augment_shared_set: bool = False,
    seed_offset: int = 0
):
    """
    Combined train/test loader:
      - test=True: loads the last clean sample per image (like retrieve_stacked_betas_test).
      - test=False: train mode, with train_mode in {"single","multiple","averaged"}.
    Preserves all prints/logging from your originals.
    """
    assert train_mode in ("single", "multiple", "averaged")
    
    # 1) behavioral TSV for trial matching
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
    
    # 2) which labels to load?
    if label_subset_name is None:
        label_subset_name = config.dataset_creation.subset_animate_face_final
    logging.info(f"Loading from {label_subset_name}")
    
    set_excel_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_check, label_subset_name
    )
    
    if augment_shared_set:
        logging.info(f"Augmenting with shared set")
        shared_path = os.path.join(
            config.directories.excel_files_target_dir, "shared", label_subset_name
        )
        shared_df = pd.read_excel(shared_path)
        base_df = pd.concat(
            [pd.read_excel(set_excel_path), shared_df], ignore_index=True
        )
        logging.info(f"Label distribution\nShared: {len(shared_df)} - Total: {len(base_df)}")
    else:
        logging.info(f"Not augmenting with shared set")
        # create empty shared_df so we can still reference it later
        temp = pd.read_excel(set_excel_path)
        shared_df = pd.DataFrame(columns=temp.columns)
        base_df = temp
    
    # 3) optionally add non-face
    if not only_face_set:
        nonface_path = os.path.join(
            config.directories.excel_files_target_dir,
            subj_to_check,
            config.dataset_creation.subset_animate_non_face_final,
        )
        base_df = pd.concat([base_df, pd.read_excel(nonface_path)], ignore_index=True)
    
    # build ID lists
    coco_ids = [f"{int(x):012d}" for x in base_df["cocoId"].tolist()]
    nsd_ids  = base_df["nsdId"].tolist()
    
    data = []
    image_ids   = []
    mds_mapping = []
    betas_dir   = config.directories.image_betas_dir
    
    # 4) loop over each image
    for mds_index, entry in enumerate(coco_ids):
        cocoId = int(entry)
        
        # find all trial‐matched betas
        tsv_matches = tsv_data.loc[
            tsv_data["73KID"] == nsd_ids[mds_index] + 1
        ].index.tolist()
        
        npy_files = sorted(
            glob.glob(
                os.path.join(betas_dir, entry, f"subj_{subj:02d}", "full.betas_*.npy")
            )
        )
        
        # randomize order (train only)
        if randomization:
            seed = subj * 10 + int(entry) + seed_offset
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(npy_files))
            npy_files = [npy_files[i] for i in perm]
        

        npy_files_cleaned = npy_files.copy()


        # sanity‐check trial indices
        for npy_path in npy_files:
            trial_id = int(os.path.basename(npy_path)[11:-4])
            assert (
                trial_id in tsv_matches
            ), "Something is wrong... the indices are not matching!"
        
        # NaN‐cleanup
        n_usable = len(npy_files)
        faulty = 0

        # for i, npy_path in reversed(list(enumerate(npy_files))):
        for i in range(len(npy_files)-1, 0-1, -1):
            npy_path = npy_files[i]
            arr = np.load(npy_path)
            if np.isnan(arr).any():
                logging.info(f"{npy_path}: removing due to NaN")
                # logging.info(f"Removing NaN... Has NaN={np.isnan(np.load(npy_files_cleaned[i])).any()};{i=};{npy_files_cleaned[i]=}")
                del npy_files_cleaned[i]
                faulty += 1
        n_usable -= faulty
        
        # print(f"{n_usable=}")
        # no usable?
        if n_usable == 0:
            if cocoId in shared_df["cocoId"].tolist():
                continue
            raise ValueError(f"No data found for {entry}!")
        
        # for train we need at least 2 samples (so we can leave one out)
        if n_usable < 2:
            if cocoId in shared_df["cocoId"].tolist():
                continue
            raise ValueError(f"Not enough correct samples for {entry}!")
        
        # record picks
        if test:
            # always take last cleaned sample
            picks = [npy_files_cleaned[-1]]
        else:
            if train_mode == "single":
                picks = [npy_files_cleaned[sample]]
            elif train_mode == "multiple":
                picks = npy_files_cleaned[:-1]
            else:  # averaged
                picks = npy_files_cleaned[:-1]
        
        # load them
        if test or train_mode == "single":
            arr = np.load(picks[-1])
            data.append(arr)
            mds_mapping.append(mds_index)
            image_ids.append(entry)
        elif train_mode == "multiple":
            for p in picks:
                arr = np.load(p)
                data.append(arr)
                mds_mapping.append(mds_index)
                image_ids.append(entry)
        else:  # averaged
            stacked = np.stack([np.load(p) for p in picks])
            mean_arr = np.mean(stacked, axis=0)
            data.append(mean_arr)
            mds_mapping.append(mds_index)
            image_ids.append(entry)

    
    assert not np.isnan(np.stack(data)).any()
    
    print(f"Final data shape = {np.array(data).shape}")
    assert np.stack(data).shape[0] == len(image_ids)
    return np.stack(data), image_ids, mds_mapping




# def retrieve_stacked_betas(
#     config: Configuration,
#     subj: int,
#     mode: str = "train",            # "train" or "test"
#     train_mode: str = "averaged",   # only used if mode=="train": "single","multiple","averaged"
#     sample_idx: int = 0,            # index for train_mode="single"
#     subj_to_check="shared",
#     label_subset_name: str = None,
#     only_face_set: bool = True,
#     augment_shared_set: bool = False,
#     randomization: bool = False,    # NEW FLAG
# ):
#     """
#     - mode="train": like your original retrieve_stacked_betas()
#       * train_mode controls "single"/"multiple"/"averaged"
#       * sample_idx only for train_mode="single"
#     - mode="test": like retrieve_stacked_betas_test()
#       * ignores train_mode & sample_idx, always picks last cleaned sample
#     - randomization: if True, shuffle the order of cleaned samples per image
#       (seeded by subj and image ID for reproducibility)
#     """

#     assert mode in ("train", "test"), "mode must be 'train' or 'test'"
#     if mode == "train":
#         assert train_mode in ("single", "multiple", "averaged"), \
#                "train_mode must be 'single','multiple',or 'averaged'"

#     # --- 1) Load and optionally augment label set ---
#     if label_subset_name is None:
#         label_subset_name = config.dataset_creation.subset_animate_face_final

#     base_df = pd.read_excel(
#         os.path.join(config.directories.excel_files_target_dir,
#                      subj_to_check, label_subset_name)
#     )
#     if augment_shared_set:
#         shared_df = pd.read_excel(
#             os.path.join(config.directories.excel_files_target_dir,
#                          "shared", label_subset_name)
#         )
#         base_df = pd.concat([base_df, shared_df], ignore_index=True)
#     else:
#         shared_df = pd.DataFrame(columns=base_df.columns)

#     if mode == "train" and not only_face_set:
#         nonface = pd.read_excel(
#             os.path.join(config.directories.excel_files_target_dir,
#                          subj_to_check,
#                          config.dataset_creation.subset_animate_non_face_final)
#         )
#         base_df = pd.concat([base_df, nonface], ignore_index=True)

#     coco_ids = [f"{int(x):012d}" for x in base_df["cocoId"]]
#     nsd_ids = base_df["nsdId"].tolist()

#     # --- 2) Load behavioral responses for trial matching ---
#     nsd_dir = os.path.join(config.nsd_project.nsd_data_dir,
#                            config.nsd_project.nsd_subdir)
#     tsv = pd.read_csv(
#         os.path.join(nsd_dir, config.nsd_project.label_subdir,
#                      "ppdata", f"subj{subj:02d}", "behav", "responses.tsv"),
#         sep="\t"
#     )

#     betas_dir = config.directories.image_betas_dir
#     data, image_ids, mapping = [], [], []

#     # --- 3) Iterate images ---
#     for idx, entry in enumerate(coco_ids):
#         # find which trials correspond
#         trials = tsv.loc[tsv["73KID"] == nsd_ids[idx] + 1].index.tolist()

#         # grab all .npy and filter by trial‐ID
#         all_npys = sorted(glob.glob(
#             os.path.join(betas_dir, entry, f"subj_{subj:02d}", "*.npy")
#         ))
#         valid = [p for p in all_npys
#                  if int(os.path.basename(p)[11:-4]) in trials]

#         if not valid:
#             if int(entry) in shared_df["cocoId"].tolist():
#                 continue
#             raise ValueError(f"No data found for {entry}")

#         # NaN‐cleanup
#         cleaned = []
#         for p in valid:
#             arr = np.load(p)
#             if np.isnan(arr).any():
#                 print(f"Removing {p} (contains NaN)")
#             else:
#                 cleaned.append(p)

#         if not cleaned:
#             if int(entry) in shared_df["cocoId"].tolist():
#                 continue
#             raise ValueError(f"All samples invalid for {entry}")

#         # *** RANDOMIZATION ***
#         if randomization and mode == "train":
#             seed = subj * 10000 + int(entry)
#             rng  = np.random.RandomState(seed)
#             cleaned = [cleaned[i] for i in rng.permutation(len(cleaned))]

#         # --- 4) Select according to mode ---
#         if mode == "test":
#             picks = [cleaned[-1]]
#         else:  # train
#             if train_mode == "single":
#                 picks = [cleaned[sample_idx]]
#             elif train_mode == "multiple":
#                 # use *all but last* as training
#                 picks = cleaned[:-1]
#             else:  # averaged
#                 picks = cleaned[:-1]

#         # --- 5) Load into arrays & record mapping/image IDs ---
#         if mode == "train" and train_mode == "averaged":
#             arrs = np.stack([np.load(p) for p in picks])
#             data.append(arrs.mean(axis=0))
#             mapping.append(idx)
#             image_ids.append(entry)
#         else:
#             for p in picks:
#                 data.append(np.load(p))
#                 mapping.append(idx)
#                 image_ids.append(entry)

#     print(f"Final data shape = {np.array(data).shape}")
#     return np.stack(data), image_ids, mapping



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
