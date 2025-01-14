import os
import json

from utils.config import load_config, Configuration

import pandas as pd


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
    check_labels(config)
