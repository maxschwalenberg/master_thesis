import pandas as pd
import json
import numpy as np
import wget
import os


from utils.config import Configuration, load_config

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


def download_data(config: Configuration):
    logging.info("Processing coco train...")

    with open(config.coco_train_json_path, "r") as f:
        info = json.load(f)

    coco_train = COCO(config.coco_train_json_path)

    info = info["images"]

    logging.info("Processing coco val...")
    with open(config.coco_val_json_path, "r") as f:
        info = json.load(f)

    coco_val = COCO(config.coco_val_json_path)

    # --> find jpgs --> get corresponding url
    df = pd.read_excel(
        os.path.join(config.excel_files_target_dir, config.nsd_coco_file_path)
    )

    df = df[df["amount_participants"] == 8]

    urls = []
    labels = []

    if os.path.exists(
        os.path.join(config.excel_files_target_dir, config.nsd_labeled_subset_path)
    ):
        need_to_label = False
    else:
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
            os.path.join(config.excel_files_target_dir, config.nsd_labeled_subset_path),
            index=False,
        )

    logging.info("Downloading...")
    for i in tqdm(range(len(urls))):
        parsed_url = urlparse(urls[i])
        filename = os.path.basename(parsed_url.path)

        if os.path.exists(os.path.join(config.images_target_dir, filename)):
            continue
        else:
            wget.download(urls[i], out=config.images_target_dir)


def create_data_split(config: Configuration):
    full_dataset = pd.read_excel(
        os.path.join(config.excel_files_target_dir, config.nsd_labeled_subset_path)
    )

    # assume that in that case both splits are created
    if os.path.exists(
        os.path.join(
            config.excel_files_target_dir, config.nsd_labeled_subset_animals_humans
        )
    ):
        logging.info("Splits already exists.")
        non_animals_df = pd.read_excel(
            os.path.join(
                config.excel_files_target_dir, config.nsd_labeled_subset_non_animals
            )
        )
        animals_df = os.path.join(
            config.excel_files_target_dir, config.nsd_labeled_subset_animals_humans
        )

        logging.info(f"n_samples Non-Animals: {len(non_animals_df)}")
        logging.info(f"n_samples Person&Animals: {len(animals_df)}")

        return

    non_animals_df = pd.DataFrame(columns=full_dataset.columns)
    animals_df = pd.DataFrame(columns=full_dataset.columns)

    for i, row in full_dataset.iterrows():
        labels = row["labels"]
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
        # teddy bear?

        has_animal_or_person_label = any(
            label in labels for label in animal_or_person_labels
        )

        if has_animal_or_person_label == 0:
            non_animals_df = pd.concat(
                [non_animals_df, pd.DataFrame([row])], ignore_index=True
            )
        else:
            animals_df = pd.concat([animals_df, pd.DataFrame([row])], ignore_index=True)

    logging.info(f"n_samples Non-Animals: {len(non_animals_df)}")
    logging.info(f"n_samples Person&Animals: {len(animals_df)}")

    animals_df.to_excel(
        os.path.join(
            config.excel_files_target_dir, config.nsd_labeled_subset_animals_humans
        )
    )
    non_animals_df.to_excel(
        os.path.join(
            config.excel_files_target_dir, config.nsd_labeled_subset_non_animals
        )
    )


if __name__ == "__main__":
    config = load_config("config.yaml")
    download_data(config)
    create_data_split(config)
