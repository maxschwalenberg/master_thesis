import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import numpy as np

from utils.config import load_config, Configuration

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate_positive_set(config: Configuration):
    for nsd_subj_subset in config.nsd_samples_subjects_to_check:
        # Create subject folder name if necessary
        if nsd_subj_subset != "shared":
            nsd_subj_subset = f"subj_{int(nsd_subj_subset):02d}"
        logging.info(f"Generating face set for {nsd_subj_subset=}")

        # Create the subdirectory path and make sure it exists
        subdir_path = os.path.join(config.excel_files_target_dir, nsd_subj_subset)

        os.makedirs(os.path.join(subdir_path, "faces"), exist_ok=True)

        face_detection_results_path = os.path.join(
            subdir_path, config.face_detection_results_path
        )
        with open(face_detection_results_path, "r") as f:
            detection_results = json.load(f)

        humans_subset_excel = pd.read_excel(
            os.path.join(subdir_path, config.nsd_labeled_subset_animals_humans)
        )

        total_image_area = 640 * 640

        bboxes = []
        bbox_image_areas = []

        for det in detection_results:
            bboxes += [
                (
                    det["detection"],
                    det["file_name"],
                    det["n_persons_label"],
                    len(det["detection"]),
                )
            ]

        logging.info(f"Filtering for images with 1 person and 1 face of large area!")
        for i, r in enumerate(bboxes):
            for b in r[0]:
                area = (b[2] - b[0]) * (b[3] - b[1])
                area_fraction = round((area / total_image_area), 2)

                if area_fraction > 0.03 and r[2] == 1 and r[3] == 1:
                    bbox_image_areas.append((area_fraction, r[1], r[2], r[3]))

        sorted_areas = sorted(bbox_image_areas, key=lambda x: x[0], reverse=True)

        # unique sorted areas - each file only present once
        seen = set()
        unique = [
            obj for obj in sorted_areas if obj[1] not in seen and not seen.add(obj[1])
        ]

        print(f"Uniques: {len(unique)}")

        result_df = pd.DataFrame(columns=humans_subset_excel.columns)
        for e in unique:
            img_path = os.path.join(config.images_target_dir, e[1])

            # print(e)
            # n face-detections == 1
            if e[3] == 1:
                file_name = e[1]
                excel_file_name_filter = file_name

                row = humans_subset_excel[
                    humans_subset_excel["file_name"].str.contains(
                        excel_file_name_filter, na=False
                    )
                ].iloc[0]
                result_df = pd.concat(
                    [result_df, pd.DataFrame([row])], ignore_index=True
                )

        result_df = result_df.loc[
            :, ~result_df.columns.str.contains("Unnamed", case=False, na=False)
        ]
        result_df.to_excel(
            os.path.join(
                subdir_path,
                "faces",
                "faces.xlsx".split(".")[0] + "_unchecked.xlsx",
            )
        )


def generate_animate_non_face(config: Configuration):
    for nsd_subj_subset in config.nsd_samples_subjects_to_check:
        # Create subject folder name if necessary
        if nsd_subj_subset != "shared":
            nsd_subj_subset = f"subj_{int(nsd_subj_subset):02d}"
        logging.info(f"Generating animate-non-face set for {nsd_subj_subset=}")

        # Create the subdirectory path and make sure it exists
        subdir_path = os.path.join(config.excel_files_target_dir, nsd_subj_subset)

        os.makedirs(os.path.join(subdir_path, "animate_nonface"), exist_ok=True)

        face_detection_results_path = os.path.join(
            subdir_path, config.face_detection_results_path
        )
        with open(face_detection_results_path, "r") as f:
            detection_results = json.load(f)

        humans_subset_excel = pd.read_excel(
            os.path.join(subdir_path, config.nsd_labeled_subset_animals_humans)
        )

        total_image_area = 640 * 640

        bboxes = []
        bbox_image_areas = []

        for det in detection_results:
            # bboxes += [
            #     det["detection"],
            #     det["file_name"],
            #     det["n_persons_label"],
            #     len(det["detection"]),
            # ]
            if len(det["detection"]) > 0:
                bboxes.append(det["file_name"])

        face_file_names = set(bboxes)

        all_file_names = humans_subset_excel["file_name"].to_list()
        for i in range(len(all_file_names)):
            all_file_names[i] = all_file_names[i].split("/")[1]

        all_file_names = set(all_file_names)

        non_face_set = list(all_file_names - face_file_names)

        logging.info(f"All Files: {len(list(all_file_names))}")
        logging.info(f"Face Files: {len(list(face_file_names))}")
        logging.info(f"Non Face Files: {len(non_face_set)}")

        result_df = pd.DataFrame(columns=humans_subset_excel.columns)

        for e in non_face_set:
            file_name = e
            excel_file_name_filter = file_name

            row = humans_subset_excel[
                humans_subset_excel["file_name"].str.contains(
                    excel_file_name_filter, na=False
                )
            ].iloc[0]
            result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)

        result_df = result_df.loc[
            :, ~result_df.columns.str.contains("Unnamed", case=False, na=False)
        ]

        final_path = os.path.join(
            subdir_path,
            "animate_nonface",
            "animate_non_face" + "_unchecked.xlsx",
        )
        logging.info(f"Saving to {final_path}")
        result_df.to_excel(final_path)


if __name__ == "__main__":
    config = load_config("config.yaml")
    generate_animate_non_face(config)
    generate_positive_set(config)
