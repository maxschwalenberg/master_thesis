import os
import json
from tqdm import tqdm
import logging
import glob
from collections import defaultdict

from utils.config import load_config, Configuration

import pandas as pd
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def correct_sets_for_nans_and_missing_samples(
    config: Configuration, sets: list[str], subject_to_filter: list
):
    subject_stats = {}
    for i in range(1, 8 + 1):
        subject_stats[i] = []

    missing_nan_information_path = os.path.join(
        config.excel_files_target_dir, "missing_nan_info.json"
    )
    with open(missing_nan_information_path, "r") as f:
        missing_nan_information = json.load(f)

    missing_subjects_json_path = os.path.join(
        config.excel_files_target_dir, "missing_subjects.json"
    )
    with open(missing_subjects_json_path, "r") as f:
        missing_subjects = json.load(f)

    # valid_filenames = []
    # for file_path in sets:
    #     logging.info(f"Processing {os.path.basename(file_path)}")
    #     subset = pd.read_excel(file_path)

    #     coco_id = subset["cocoId"].to_list()
    #     coco_id = [f"{e:012d}" for e in coco_id]

    #     coco_ids_per_set.append(coco_id)

    #     duplicates = len(coco_id) != len(set(coco_id))
    #     logging.info(f"Duplicates: {duplicates}")

    #     valid_filenames += coco_id

    # duplicates = len(valid_filenames) != len(set(valid_filenames))
    # logging.info(f"Duplicates for joined sets?: {duplicates}")
    # print("-" * 20)

    for coco_id, missing_subjects in missing_subjects.items():
        for missing_subject in missing_subjects:
            subject_stats[missing_subject].append(coco_id)

    for coco_id, missing_subjects_dict in missing_nan_information.items():
        missing_subjects = list(missing_subjects_dict.keys())
        for missing_subject in missing_subjects:
            subject_stats[int(missing_subject)].append(coco_id)

    set_coco_ids = []
    for set_path in sets:
        set_df = pd.read_excel(set_path)
        coco_ids = set_df["cocoId"].to_list()
        coco_ids = [f"{e:012d}" for e in coco_ids]

        set_coco_ids.append(coco_ids)

    all_faulty = []
    for subject in range(1, 8 + 1):
        print(f"{subject=}")
        subject_stats[subject] = list(set(subject_stats[subject]))

        if subject in subject_to_filter:
            all_faulty += subject_stats[subject]

        # print(f"Subject {subject}: {len(subject_stats[subject])}")
        for i, set_coco_id in enumerate(set_coco_ids):

            print(
                f"{os.path.basename(sets[i])}\t{len(list(set(subject_stats[subject]) & set(set_coco_id)))} / {len(subject_stats[subject])}"
            )

        print("-" * 25)

    all_faulty = list(set(all_faulty))
    for i, set_coco_id in enumerate(set_coco_ids):

        print(
            f"{os.path.basename(sets[i])}\t{len(list(set(all_faulty) & set(set_coco_id)))} / {len(all_faulty)}"
        )

    # subset the datasets
    for i, set_path in enumerate(sets):
        set_df = pd.read_excel(set_path)

        path_basename = os.path.basename(set_path)
        new_path = os.path.join(
            config.excel_files_target_dir,
            os.path.splitext(path_basename)[0]
            + "_filtered"
            + os.path.splitext(path_basename)[1],
        )

        set_coco_id = set_coco_ids[i]
        intersect = list(set(all_faulty) & set(set_coco_id))
        intersect = [int(e.lstrip("0")) for e in intersect]

        set_df = set_df[~set_df["cocoId"].isin(intersect)]
        set_df.to_excel(new_path, index=False)


def adjust_labelled_data(
    config: Configuration, fullset: bool = False, strict: bool = True
):
    missing_subjects_json_path = os.path.join(
        config.excel_files_target_dir, "missing_subjects.json"
    )
    nsd_all_coco_path = os.path.join(
        config.excel_files_target_dir, config.nsd_coco_file_path
    )

    nsd_coco = pd.read_excel(nsd_all_coco_path)
    nsd_coco_filtered = nsd_coco[nsd_coco["amount_participants"] == 8]

    with open(missing_subjects_json_path, "r") as f:
        missing_subjects = json.load(f)

    if fullset:
        # handle nsd_coco_filtered
        raise NotImplementedError()

    else:
        file_paths = [
            os.path.join(config.excel_files_target_dir, "non_animate_subset.xlsx"),
            os.path.join(config.excel_files_target_dir, "faces_subset.xlsx"),
            os.path.join(config.excel_files_target_dir, "animate_non_face.xlsx"),
        ]

        coco_ids_per_set = []

        valid_filenames = []
        for file_path in file_paths:
            logging.info(f"Processing {os.path.basename(file_path)}")
            subset = pd.read_excel(file_path)

            coco_id = subset["cocoId"].to_list()
            coco_id = [f"{e:012d}" for e in coco_id]

            coco_ids_per_set.append(coco_id)

            duplicates = len(coco_id) != len(set(coco_id))
            logging.info(f"Duplicates: {duplicates}")

            valid_filenames += coco_id

        duplicates = len(valid_filenames) != len(set(valid_filenames))
        logging.info(f"Duplicates for joined sets?: {duplicates}")
        print("-" * 20)

    missing_subjects_coco_ids = list(missing_subjects.keys())
    duplicates = len(missing_subjects_coco_ids) != len(set(missing_subjects_coco_ids))
    logging.info(f"Duplicates for missing subjects?: {duplicates}")

    logging.info(
        f"Removing {len(missing_subjects_coco_ids)} because of missing subjects...."
    )
    valid_filenames = [x for x in valid_filenames if x not in missing_subjects_coco_ids]

    remove_nan_filenames = []

    remove_by_subject = {}

    missing_nan_information = defaultdict(lambda: defaultdict(list))

    for subj in range(1, 8 + 1):
        remove_by_subject[subj] = []

    images_betas_dir = config.image_betas_dir
    for valid_filename in tqdm(valid_filenames):
        for subj in range(1, 8 + 1):
            npy_files = sorted(
                glob.glob(
                    os.path.join(
                        images_betas_dir, valid_filename, f"subj_{subj:02d}", "*.npy"
                    )
                )
            )

            all_nan = []
            for npy_file in npy_files:
                sample = np.load(npy_file)
                is_nan = np.isnan(np.sum(sample))
                if is_nan:

                    missing_nan_information[valid_filename][subj].append(
                        os.path.basename(npy_file)
                    )

                    all_nan.append(True)
                    nan_count = np.sum(np.isnan(sample))
                    # print(
                    #     f"{valid_filename}\t{npy_file}\tNumber of NaN values: {nan_count}",
                    # )

                else:
                    all_nan.append(False)

            # collect number of instance where NO NaN
            # only remove if less than 2 samples are not NaN (because we dont have a test set anymore)
            if strict:
                if True in all_nan:
                    remove_nan_filenames.append(valid_filename)
                    remove_by_subject[subj].append(valid_filename)

                    remove_by_subject[subj] = list(set(remove_by_subject[subj]))

            else:
                if len([e for e in all_nan if not e]) < 2:
                    remove_nan_filenames.append(valid_filename)
                    remove_by_subject[subj].append(valid_filename)

                    remove_by_subject[subj] = list(set(remove_by_subject[subj]))

    remove_nan_filenames = list(set(remove_nan_filenames))
    logging.info(f"Removing {len(remove_nan_filenames)} because of NaNs....")
    if not fullset:
        for set_i, file_path in enumerate(file_paths):
            logging.info(
                f"{os.path.basename(file_path)}: {len(list(set(remove_nan_filenames) & set(coco_ids_per_set[set_i])))} / {len(remove_nan_filenames)}"
            )

    print("-" * 20)

    valid_filenames = [x for x in valid_filenames if x not in remove_nan_filenames]

    for subj in range(1, 8 + 1):
        logging.info(
            f"{subj:02d}: Removing {len(remove_by_subject[subj])} because of NaNs...."
        )
        if not fullset:
            for set_i, file_path in enumerate(file_paths):
                logging.info(
                    f"{os.path.basename(file_path)}: {len(list(set(remove_by_subject[subj]) & set(coco_ids_per_set[set_i])))} / {len(remove_by_subject[subj])}"
                )

        print()

    missing_nan_information_path = os.path.join(
        config.excel_files_target_dir, "missing_nan_info.json"
    )
    with open(missing_nan_information_path, "w") as f:
        json.dump(dict(missing_nan_information), f, indent=4)


def check_labels(config: Configuration):
    nsd_dir = os.path.join(config.nsd_data_dir, config.nsd_subdir)

    nsd_coco_path = os.path.join(
        config.excel_files_target_dir, config.nsd_coco_file_path
    )

    nsd_coco = pd.read_excel(nsd_coco_path)
    nsd_coco_filtered = nsd_coco[nsd_coco["amount_participants"] == 8]

    tsv_datas = []
    for subject_id in range(1, 8 + 1):
        tsv_path = os.path.join(
            nsd_dir,
            config.label_subdir,
            "ppdata",
            f"subj{subject_id:02d}",
            "behav",
            "responses.tsv",
        )

        tsv_data = pd.read_csv(tsv_path, sep="\t")
        tsv_datas.append(tsv_data)

    counter = 0
    result_dict = {}

    for i, entry in nsd_coco_filtered.iterrows():
        entry_nsd = entry["nsdId"]

        missing_subjects = []
        for subj in range(8):
            tsv_data = tsv_datas[subj]

            matching_trials = tsv_data[tsv_data["73KID"].isin([entry_nsd + 1])]

            if matching_trials.shape[0] == 0:
                missing_subjects.append(subj + 1)

        if len(missing_subjects) != 0:
            counter += 1

            result_dict[f'{entry["cocoId"]:012d}'] = missing_subjects

            print(f"{entry['cocoId']}\t{missing_subjects}")

    with open(
        os.path.join(config.excel_files_target_dir, "missing_subjects.json"), "w"
    ) as f:
        json.dump(result_dict, f)

    print(f"In total: {counter} missing / wrongly labelled")


if __name__ == "__main__":
    config = load_config("config.yaml")

    correct_sets_for_nans_and_missing_samples(
        config,
        [
            "data/labels/animate_nonface/animate_non_face.xlsx",
            "data/labels/faces/faces_subset.xlsx",
            "data/labels/nonanimate/non_animate_subset.xlsx",
        ],
        [1, 3, 7, 2],
    )
    #  adjust_labelled_data(config)
    # check_labels(config)
