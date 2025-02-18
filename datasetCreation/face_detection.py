import os
import json

from pprint import pprint

import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import model_zoo

import insightface
from insightface.data import get_image as ins_get_image
import pandas as pd
from tqdm import tqdm

from utils.config import Configuration, load_config
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def generate_face_detection_results(config: Configuration):
    app = FaceAnalysis(
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        name="buffalo_l",
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    for nsd_subj_subset in config.nsd_samples_subjects_to_check:
        # Create subject folder name if necessary
        if nsd_subj_subset != "shared":
            nsd_subj_subset = f"subj_{int(nsd_subj_subset):02d}"

        logging.info(f"Distribution for {nsd_subj_subset}...")
        subdir_path = os.path.join(config.excel_files_target_dir, nsd_subj_subset)

        faces_dataset_split = pd.read_excel(
            os.path.join(subdir_path, config.nsd_labeled_subset_animals_humans)
        )

        detection_results = []

        face_det_results_path = os.path.join(
            subdir_path, config.face_detection_results_path
        )

        if os.path.exists(face_det_results_path):
            with open(face_det_results_path, "r") as f:
                existing_results = json.load(f)
                checked_files = [
                    os.path.basename(e["img_path"]) for e in existing_results
                ]
        else:
            existing_results = []
            checked_files = []

        detection_results = existing_results

        counter = 0
        for i, row in tqdm(
            faces_dataset_split.iterrows(), total=faces_dataset_split.shape[0]
        ):
            res = {}

            filename = os.path.basename(row["file_name"])
            img_path = os.path.join(config.images_target_dir, filename)

            if filename in checked_files:
                continue

            # only if person present
            if "person" in row["labels"]:
                res["img_path"] = img_path
                res["file_name"] = filename
                res["n_persons_label"] = row["labels"].count("person")

                img = cv2.imread(img_path)

                faces = app.get(img)
                detections = []

                for face in faces:
                    bbox = face.bbox.astype(int).tolist()
                    detections.append(bbox)

                res["detection"] = detections
                detection_results.append(res)

            else:
                continue

            if i % 250 == 0:
                with open(face_det_results_path, "w") as f:
                    json.dump(detection_results, f, indent=4)

        with open(face_det_results_path, "w") as f:
            json.dump(detection_results, f, indent=4)


if __name__ == "__main__":
    config = load_config("config.yaml")
    generate_face_detection_results(config)
