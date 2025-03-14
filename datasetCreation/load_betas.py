import os
import numpy as np
import pandas as pd


from utils.config import Configuration, load_config
import logging
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


"""
Fetch  betas from all participants, and average them based on conditions
The conditions needs to be stored in the nsd_dir, then under ppdata
The responses.tsv files for each participants can be found in the original dataset, under 
their amazon webservice database 

I will try to provide a linux command to download it (and everything else) 

The betas are then stored in the appropriately named directory, and will 
be used for the RDM and fitting later on. 

I also allow for train-test spliiting
Default is None for simplicity sake

For loading the whole brain, refer to the load_betas_full.py; splitting the data was needed to not overload the RAM 
"""


targetspace = "nativesurface"


def save_betas_single_image_improved(
    nsd_coco_df_positive: pd.DataFrame,
    nsd_coco_df_negative: pd.DataFrame,
    subject_id: int,
    config: Configuration,
    target_dir_positive: str,
    target_dir_negative: str,
    overwrite: bool = False,
    mode: str = "averaged",
):
    logging.info(f"Processing subject {subject_id:02d}")
    nsd_dir = os.path.join(
        config.nsd_project.nsd_data_dir, config.nsd_project.nsd_subdir
    )

    # STEPS:
    # - load tsv file for subject
    # - load full brain data (stans files)
    # - for each sample in set:
    # - filter for 73K ID
    # - extract trial -> save the data

    if not os.path.exists(target_dir_positive):
        os.makedirs(target_dir_positive)

    tsv_path = os.path.join(
        nsd_dir,
        config.nsd_project.label_subdir,
        "ppdata",
        f"subj{subject_id:02d}",
        "behav",
        "responses.tsv",
    )

    # having all of the array loaded at once results in memory pressure
    # reduce to needed subset!

    tsv_data = pd.read_csv(tsv_path, sep="\t")

    # get all nsd ids in given set
    nsd_ids_pandas_positive = nsd_coco_df_positive["nsdId"]
    nsd_ids_pandas_negative = nsd_coco_df_negative["nsdId"]

    # turn into array of nsd ids
    nsd_ids_positive = nsd_ids_pandas_positive.tolist()
    nsd_ids_negative = nsd_ids_pandas_negative.tolist()

    nsd_ids = nsd_ids_positive + nsd_ids_negative
    # increase by one to conform with responses.tsv
    ids_73k = [id_ + 1 for id_ in nsd_ids]

    # extract all recordings matching nsdid / 73kid
    matching_trials = tsv_data[tsv_data["73KID"].isin(ids_73k)]

    # extract indices
    matching_indices = list(matching_trials.index)

    if overwrite:
        need_to_process = True
    else:
        need_to_process = False

    counter = 0
    for i, row in tqdm(matching_trials.iterrows(), total=matching_trials.shape[0]):

        nsd = int(row["73KID"]) - 1

        if nsd in nsd_ids_positive:
            coco_id = nsd_coco_df_positive[nsd_coco_df_positive["nsdId"] == nsd].iloc[
                0
            ]["cocoId"]
            final_target_dir = os.path.join(
                target_dir_positive,
                f"{coco_id:012d}",
                f"subj_{subject_id:02d}",
            )
        else:
            coco_id = nsd_coco_df_negative[nsd_coco_df_negative["nsdId"] == nsd].iloc[
                0
            ]["cocoId"]
            final_target_dir = os.path.join(
                target_dir_negative,
                f"{coco_id:012d}",
                f"subj_{subject_id:02d}",
            )

        mt_index = matching_indices[counter]
        numpy_file_path = os.path.join(final_target_dir, f"full.betas_{mt_index}.npy")
        if not os.path.exists(numpy_file_path):
            need_to_process = True
            logging.info(f"need to process {numpy_file_path}")
            break

        counter += 1

    if not need_to_process:
        logging.info("Required files already existent - returning!")
        return

    # path = os.path.join("/media/harveylab/STORAGE1_NA/NSD/full_brain/subj01", "full_betas_subj01.npy")
    full_brain_data_path = os.path.join(
        config.nsd_data.full_brain_data_dir,
        f"subj{subject_id:02d}",
        f"full_betas_subj{subject_id:02d}.npy",
    )

    logging.info(f"Load full brain data for subject {subject_id} - {full_brain_data_path}")

    full_brain_data = np.load(full_brain_data_path, allow_pickle=False)


    logging.info("Loading finished")
    #full_brain_data = full_brain_data.T
    logging.info("Transposing finished")

    # testing
    # full_brain_data = np.zeros((tsv_data.shape[0], 1))
    print(tsv_data.shape)
    print(full_brain_data.shape)

    assert tsv_data.shape[0] == full_brain_data.shape[0]

    logging.info("Extract only needed trials")
    # reduce full brain array of subject to extracted recordings
    full_brain_data = full_brain_data[matching_indices]
    logging.info("Extraction finished")

    counter = 0
    for i, row in tqdm(matching_trials.iterrows(), total=matching_trials.shape[0]):
        nsd_id = int(row["73KID"]) - 1
        nsd = int(row["73KID"]) - 1
        if nsd in nsd_ids_positive:
            coco_id = nsd_coco_df_positive[
                nsd_coco_df_positive["nsdId"] == nsd_id
            ].iloc[0]["cocoId"]

            final_target_dir = os.path.join(
                target_dir_positive,
                f"{coco_id:012d}",
                f"subj_{subject_id:02d}",
            )

        else:
            coco_id = nsd_coco_df_negative[
                nsd_coco_df_negative["nsdId"] == nsd_id
            ].iloc[0]["cocoId"]

            final_target_dir = os.path.join(
                target_dir_positive,
                f"{coco_id:012d}",
                f"subj_{subject_id:02d}",
            )

        os.makedirs(final_target_dir, exist_ok=True)

        # nsd_coco ids and 73K_ID dont align, so add + 1
        id_73k = nsd_id + 1

        # logging.info(f"Loading betas for {nsd_id=} -> {id_73k=}\t{subject_id=}")

        # find trials matching nsd_id
        matching_trials_subset = matching_trials[matching_trials["73KID"] == id_73k]
        matching_trials_subset.to_csv(
            os.path.join(final_target_dir, "matching_responses.tsv"), sep="\t"
        )

        mt_index = matching_indices[counter]
        # logging.info(f"Extracting for {mt_index=}")

        extract = full_brain_data[counter]

        # logging.info(f"Extracting finished")

        numpy_file_path = os.path.join(final_target_dir, f"full.betas_{mt_index}.npy")
        if os.path.exists(numpy_file_path):
            # logging.info(f"Removing existent beta for {mt_index=}")
            os.remove(numpy_file_path)

        # logging.info(f"Saving beta for {mt_index=}")

        np.save(
            numpy_file_path,
            extract,
        )
        counter += 1

    del full_brain_data


def load_betas_subset(
    config: Configuration, overwrite: bool = False, subj_to_pick="shared"
):
    target_image_betas_dir_positive = os.path.join(config.directories.image_betas_dir)
    target_image_betas_dir_negative = os.path.join(config.directories.image_betas_dir)

    positive_subset_excel_path = os.path.join(
        config.directories.excel_files_target_dir,
        subj_to_pick,
        config.dataset_creation.subset_animate_face_unchecked,
    )
    negative_subset_excel_path = os.path.join(
        config.directories.excel_files_target_dir,
        subj_to_pick,
        config.dataset_creation.subset_animate_non_face_unchecked,
    )

    # positive_subset = pd.read_excel(positive_subset_excel_path)
    negative_subset = pd.read_excel(negative_subset_excel_path)
    positive_subset = pd.read_excel(positive_subset_excel_path)

    logging.info(f"Positive Set: {len(positive_subset)}")
    logging.info(f"Negative Set: {len(negative_subset)}")

    os.makedirs(target_image_betas_dir_negative, exist_ok=True)
    os.makedirs(target_image_betas_dir_positive, exist_ok=True)

    logging.info(f"Extract betas for positive&negative subset")
    subjects = list(range(1, 8 + 1))
    for subject in subjects:
        save_betas_single_image_improved(
            positive_subset,
            negative_subset,
            subject,
            config,
            target_image_betas_dir_positive,
            target_image_betas_dir_negative,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    config = load_config("config.yaml")

    assert len(config.dataset_validation.nsd_samples_subjects_to_check) == 1

    subj_to_pick = (
        "shared"
        if config.dataset_validation.nsd_samples_subjects_to_check[0] == "shared"
        else f"subj_{int(config.dataset_validation.nsd_samples_subjects_to_check[0]):02d}"
    )

    load_betas_subset(config, overwrite=False, subj_to_pick=subj_to_pick)
