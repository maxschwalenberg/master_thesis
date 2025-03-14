import pandas as pd
import json
import numpy as np
import wget
import os


from utils.config import Configuration, load_config
from utils.utils import logging_message, subjects_list_unifier

import logging
from tqdm import tqdm
from urllib.parse import urlparse
from pycocotools.coco import COCO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_coco_image_labels(image_id, coco_instance: COCO):
    """
    Retrieve the labels for a given image in the COCO dataset.

    Parameters:
    image_id (int): The ID of the image to get labels for.
    coco_data_dir (str): The directory containing the COCO dataset files.

    Returns:
    list: A list of label names for the image.
    """
    # Load the COCO annotations

    # Get the annotations for the given image
    annot_ids = coco_instance.getAnnIds(imgIds=image_id)
    annotations = coco_instance.loadAnns(annot_ids)

    # Extract the label names
    label_names = [
        coco_instance.loadCats(ann["category_id"])[0]["name"] for ann in annotations
    ]

    return label_names


def filter_full_nsd_df(df: pd.DataFrame, config: Configuration):
    conditions = []

    if "shared" in config.nsd_samples_subjects_to_check:
        conditions.append(("amount_participants", {8}))

        # Erstelle eine neue Liste ohne "shared"
        subjects = {
            int(s) for s in config.nsd_samples_subjects_to_check if s != "shared"
        }
        conditions.append(("subject", subjects))
    else:
        conditions.append(
            ("subject", {int(s) for s in config.nsd_samples_subjects_to_check})
        )

    # Filtere die DataFrame-Zeilen basierend auf den Bedingungen
    df = df[
        df.apply(
            lambda row: any(row[col] in values for col, values in conditions), axis=1
        )
    ]

    return df


def extract_subj_nsd_df(df: pd.DataFrame, subj: str):
    if subj == "shared":
        df = df[(df["amount_participants"] == 8)]
    else:
        df = df[(df["subject"] == int(subj))]

    return df


def download_data(config: Configuration):

    if config.pipeline.step_2_dataset_creation.download_data:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step, "Starting image download"
            )
        )
    else:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step, "Skipping image download"
            )
        )
        return

    logging.info("Processing coco train...")

    with open(config.coco_data.coco_train_json_path, "r") as f:
        info = json.load(f)

    coco_train = COCO(config.coco_data.coco_train_json_path)

    info = info["images"]

    logging.info("Processing coco val...")
    with open(config.coco_data.coco_val_json_path, "r") as f:
        info = json.load(f)

    coco_val = COCO(config.coco_data.coco_val_json_path)

    # --> find jpgs --> get corresponding url
    df = pd.read_excel(
        os.path.join(
            config.directories.excel_files_target_dir,
            config.coco_data.nsd_coco_file_path,
        )
    )
    df = filter_full_nsd_df(df, config)

    urls = []
    labels = []

    # if os.path.exists(
    #     os.path.join(config.excel_files_target_dir, config.nsd_labeled_subset_path)
    # ):
    #     need_to_label = False
    # else:
    #     need_to_label = True
    #     logging.info("Getting Labels")

    need_to_label = True
    logging.info("Getting Labels")

    # 'http://images.cocodataset.org/train2017/000000391895.jpg'
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = os.path.join("http://images.cocodataset.org", row["file_name"])
        urls.append(url)

        if need_to_label:
            image_id = row["cocoId"]
            coco_instance = coco_train if row["cocoSplit"] == "train2017" else coco_val

            labels.append(get_coco_image_labels(image_id, coco_instance))

    if need_to_label:
        df["labels"] = labels
        df.to_excel(
            os.path.join(
                config.directories.excel_files_target_dir,
                config.coco_data.nsd_labeled_subset_path,
            ),
            index=False,
        )

    logging.info("Downloading...")
    for i in tqdm(range(len(urls))):
        parsed_url = urlparse(urls[i])
        filename = os.path.basename(parsed_url.path)

        if os.path.exists(os.path.join(config.directories.images_target_dir, filename)):
            continue
        else:
            wget.download(urls[i], out=config.directories.images_target_dir)


def create_data_split(config: Configuration):
    if config.pipeline.step_2_dataset_creation.create_data_split:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step,
                "Starting data split generation",
            )
        )
    else:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step,
                "Skipping data split generation",
            )
        )
        return

    subjects = subjects_list_unifier(
        config.pipeline.step_2_dataset_creation.subjects, False
    )

    full_dataset = pd.read_excel(
        os.path.join(
            config.directories.excel_files_target_dir,
            config.coco_data.nsd_labeled_subset_path,
        )
    )

    full_dataset = filter_full_nsd_df(full_dataset, config)

    for nsd_subj_subset in subjects:
        filtered_df = extract_subj_nsd_df(full_dataset, nsd_subj_subset)

        # Create subject folder name if necessary
        if nsd_subj_subset != "shared":
            nsd_subj_subset = f"subj_{nsd_subj_subset:02d}"

        logging.info(f"Distribution for {nsd_subj_subset}...")

        # Create the subdirectory path and make sure it exists
        subdir_path = os.path.join(
            config.directories.excel_files_target_dir, nsd_subj_subset
        )
        os.makedirs(subdir_path, exist_ok=True)

        # Create file paths for the two Excel files
        animals_path = os.path.join(subdir_path, config.dataset_creation.subset_animate)
        non_animals_path = os.path.join(
            subdir_path, config.dataset_creation.subset_non_animate
        )

        # Assume that if the animals file exists, both splits exist.
        if os.path.exists(animals_path):
            logging.info("Splits already exist.")
            non_animals_df = pd.read_excel(non_animals_path)
            animals_df = pd.read_excel(animals_path)

            logging.info(f"n_samples Non-Animals: {len(non_animals_df)}")
            logging.info(f"n_samples Person & Animals: {len(animals_df)}")
            continue

        # Create empty DataFrames with the same columns as in full_dataset
        non_animals_df = pd.DataFrame(columns=filtered_df.columns)
        animals_df = pd.DataFrame(columns=filtered_df.columns)

        # Define labels that indicate an animal or a person is present
        animal_or_person_labels = [
            "person",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
        ]

        # Iterate over each row and append it to the correct DataFrame
        for i, row in filtered_df.iterrows():
            labels = row["labels"]
            has_animal_or_person_label = any(
                label in labels for label in animal_or_person_labels
            )
            if not has_animal_or_person_label:
                non_animals_df = pd.concat(
                    [non_animals_df, pd.DataFrame([row])], ignore_index=True
                )
            else:
                animals_df = pd.concat(
                    [animals_df, pd.DataFrame([row])], ignore_index=True
                )

        logging.info(f"n_samples Non-Animals: {len(non_animals_df)}")
        logging.info(f"n_samples Person & Animals: {len(animals_df)}")

        # Write the DataFrames to Excel files
        animals_df.to_excel(animals_path)
        non_animals_df.to_excel(non_animals_path)


if __name__ == "__main__":
    config = load_config("config.yaml")
    download_data(config)
    create_data_split(config)
