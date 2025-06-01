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


def correct_sets_for_nans_and_missing_samples(config: Configuration, subj_to_pick: str):

    subject_stats = {}

    if subj_to_pick == "shared":
        for i in range(1, 8 + 1):
            subject_stats[i] = []
    else:
        subject_stats[int(subj_to_pick[-1])] = []

    missing_nan_information_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_pick, "missing_nan_info.json"
    )
    with open(missing_nan_information_path, "r") as f:
        missing_nan_information = json.load(f)

    missing_subjects_json_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_pick, "missing_subjects.json"
    )

    if subj_to_pick == "shared":
        with open(missing_subjects_json_path, "r") as f:
            missing_subjects = json.load(f)

    else:
        missing_subjects = {}

    for coco_id, missing_subjects in missing_subjects.items():
        for missing_subject in missing_subjects:
            subject_stats[missing_subject].append(coco_id)

    for coco_id, missing_subjects_dict in missing_nan_information.items():
        missing_subjects = list(missing_subjects_dict.keys())
        for missing_subject in missing_subjects:
            subject_stats[int(missing_subject)].append(coco_id)

    set_coco_ids = []
    #
    sets = [
        os.path.join(
            config.directories.excel_files_target_dir,
            subj_to_pick,
            config.dataset_creation.subset_animate_face_unchecked,
        ),
        os.path.join(
            config.directories.excel_files_target_dir,
            subj_to_pick,
            config.dataset_creation.subset_animate_non_face_unchecked,
        ),
    ]

    target_paths = [
        os.path.join(
            config.directories.excel_files_target_dir,
            subj_to_pick,
            config.dataset_creation.subset_animate_face_nans_removed,
        ),
        os.path.join(
            config.directories.excel_files_target_dir,
            subj_to_pick,
            config.dataset_creation.subset_animate_non_face_nans_removed,
        ),
    ]

    for set_path in sets:
        set_df = pd.read_excel(set_path)
        coco_ids = set_df["cocoId"].to_list()
        coco_ids = [f"{e:012d}" for e in coco_ids]

        set_coco_ids.append(coco_ids)

    all_faulty = []
    for subject in list(subject_stats.keys()):
        print(f"{subject=}")
        subject_stats[subject] = list(set(subject_stats[subject]))

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
        new_path = target_paths[i]

        set_coco_id = set_coco_ids[i]
        intersect = list(set(all_faulty) & set(set_coco_id))
        intersect = [int(e.lstrip("0")) for e in intersect]

        set_df = set_df[~set_df["cocoId"].isin(intersect)]
        logging.info(f"Saving to {new_path}")
        set_df.to_excel(new_path, index=False)


def adjust_labelled_data(
    config: Configuration, subj_to_pick: str, fullset: bool = False, strict: bool = True
):

    if subj_to_pick == "shared":
        check_labels(config)
        missing_subjects_json_path = os.path.join(
            config.directories.excel_files_target_dir, "missing_subjects.json"
        )
        with open(missing_subjects_json_path, "r") as f:
            missing_subjects = json.load(f)

    else:
        missing_subjects = {}

    nsd_all_coco_path = os.path.join(
        config.directories.excel_files_target_dir, config.coco_data.nsd_coco_file_path
    )

    nsd_coco = pd.read_excel(nsd_all_coco_path)
    nsd_coco_filtered = nsd_coco[nsd_coco["amount_participants"] == 8]

    if fullset:
        # handle nsd_coco_filtered
        raise NotImplementedError()

    else:
        file_paths = [
            # os.path.join(
            #     config.excel_files_target_dir, subj_to_pick, "non_animate_subset.xlsx"
            # ),
            os.path.join(
                config.directories.excel_files_target_dir,
                subj_to_pick,
                config.dataset_creation.subset_animate_face_unchecked,
            ),
            os.path.join(
                config.directories.excel_files_target_dir,
                subj_to_pick,
                config.dataset_creation.subset_animate_non_face_unchecked,
            ),
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

    if subj_to_pick == "shared":
        subjects_to_loop = list(range(1, 8 + 1))
    else:
        subjects_to_loop = [int(subj_to_pick[-1])]

    for subj in subjects_to_loop:
        remove_by_subject[subj] = []

    images_betas_dir = config.directories.image_betas_dir
    for valid_filename in tqdm(valid_filenames):
        for subj in subjects_to_loop:
            npy_files = sorted(
                glob.glob(
                    os.path.join(
                        images_betas_dir, valid_filename, f"subj_{subj:02d}", "*.npy"
                    )
                )
            )

            if len(npy_files) < 2:
                remove_nan_filenames.append(valid_filename)
                remove_by_subject[subj].append(valid_filename)
                missing_nan_information[valid_filename][subj].append(
                    os.path.basename(npy_file)
                )
                logging.info(
                    f"Only {len(npy_files)} found for {valid_filename}- skipping..."
                )
                continue

            assert len(npy_files) != 0

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
                    print(
                        f"{valid_filename}\t{npy_file}\tNumber of NaN values: {nan_count} / {np.shape(sample)}",
                    )

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

    for subj in subjects_to_loop:
        logging.info(
            f"{subj:02d}: Removing {len(remove_by_subject[subj])} because of NaNs...."
        )
        if not fullset:
            for set_i, file_path in enumerate(file_paths):
                logging.info(
                    f"{os.path.basename(file_path)}: {len(list(set(remove_by_subject[subj]) & set(coco_ids_per_set[set_i])))} / {len(remove_by_subject[subj])}"
                )

        print()


    quit()
    missing_nan_information_path = os.path.join(
        config.directories.excel_files_target_dir, subj_to_pick, "missing_nan_info.json"
    )
    with open(missing_nan_information_path, "w") as f:
        json.dump(dict(missing_nan_information), f, indent=4)


def check_labels(config: Configuration):
    nsd_dir = os.path.join(
        config.nsd_project.nsd_data_dir, config.nsd_project.nsd_subdir
    )

    nsd_coco_path = os.path.join(
        config.directories.excel_files_target_dir, config.coco_data.nsd_coco_file_path
    )

    nsd_coco = pd.read_excel(nsd_coco_path)
    nsd_coco_filtered = nsd_coco[nsd_coco["amount_participants"] == 8]

    tsv_datas = []
    for subject_id in range(1, 8 + 1):
        tsv_path = os.path.join(
            nsd_dir,
            config.nsd_project.label_subdir,
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
        os.path.join(
            config.directories.excel_files_target_dir, "missing_subjects.json"
        ),
        "w",
    ) as f:
        json.dump(result_dict, f)

    print(f"In total: {counter} missing / wrongly labelled")


import os
import json
import logging
from collections import defaultdict
import pandas as pd


def load_existing_missing_data_stats(config: Configuration, subj_to_pick: str):

    missing_nan_information_path = os.path.join(
        config.excel_files_target_dir, subj_to_pick, "missing_nan_info.json"
    )

    file_paths = [
        # os.path.join(
        #     config.excel_files_target_dir, subj_to_pick,"nonanimate", "non_animate_subset.xlsx"
        # ),
        os.path.join(
            config.excel_files_target_dir,
            subj_to_pick,
            config.subset_animate_face_final,
        ),
        os.path.join(
            config.excel_files_target_dir,
            subj_to_pick,
            config.subset_animate_non_face_final,
        ),
    ]

    if subj_to_pick == "shared":
        missing_subjects_json_path = os.path.join(
            config.excel_files_target_dir, subj_to_pick, "missing_subjects.json"
        )

        with open(missing_subjects_json_path, "r") as f:
            missing_subjects = json.load(f)
    else:
        missing_subjects = {}

    with open(missing_nan_information_path, "r") as f:
        missing_nan_information = json.load(f)

    coco_ids_per_set = []
    valid_filenames = []
    for file_path in file_paths:
        logging.info(f"Processing {os.path.basename(file_path)}")
        subset = pd.read_excel(file_path)

        coco_id = subset["cocoId"].to_list()
        coco_id = [f"{e:012d}" for e in coco_id]

        coco_ids_per_set.append(coco_id)
        valid_filenames += coco_id

    missing_subjects_coco_ids = list(missing_subjects.keys())
    logging.info(
        f"Removing {len(missing_subjects_coco_ids)} because of missing subjects...."
    )
    valid_filenames = [x for x in valid_filenames if x not in missing_subjects_coco_ids]

    remove_nan_filenames = list(missing_nan_information.keys())
    logging.info(f"Removing {len(remove_nan_filenames)} because of NaNs....")

    for set_i, file_path in enumerate(file_paths):
        logging.info(
            f"{os.path.basename(file_path)}: {len(list(set(remove_nan_filenames) & set(coco_ids_per_set[set_i])))} / {len(remove_nan_filenames)}"
        )

    for subj in range(1, 9):
        remove_by_subject = [
            img
            for img, subjects in missing_nan_information.items()
            if str(subj) in subjects
        ]
        print()
        logging.info(
            f"{subj:02d}: Removing {len(remove_by_subject)} because of NaNs...."
        )
        for set_i, file_path in enumerate(file_paths):
            logging.info(
                f"{os.path.basename(file_path)}: {len(list(set(remove_by_subject) & set(coco_ids_per_set[set_i])))} / {len(remove_by_subject)}"
            )

    print("-" * 20)


if __name__ == "__main__":
    config = load_config("config.yaml")

    assert len(config.nsd_samples_subjects_to_check) == 1

    subj_to_pick = (
        "shared"
        if config.nsd_samples_subjects_to_check[0] == "shared"
        else f"subj_{int(config.nsd_samples_subjects_to_check[0]):02d}"
    )

    # subj_to_pick = "subj_01"

    adjust_labelled_data(config, subj_to_pick)

    # load_existing_missing_data_stats(config, subj_to_pick)

    correct_sets_for_nans_and_missing_samples(
        config,
        subj_to_pick,
    )
    #  adjust_labelled_data(config)
    # check_labels(config)
