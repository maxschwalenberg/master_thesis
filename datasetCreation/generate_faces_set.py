"""
Dataset Creation Module

This module generates positive and negative stimulus sets for face analysis.
It processes face detection results to create curated datasets of face stimuli
(positive set) and animate non-face stimuli (negative set) based on detection
quality metrics and area thresholds.
"""

# Standard library imports
import json
import os
import logging

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Local imports
from utils.config import load_config, Configuration
from utils.utils import logging_message, subjects_list_unifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# POSITIVE SET GENERATION (FACE STIMULI)
# =========================================================================


def generate_positive_set(config: Configuration):
    """
    Generate positive stimulus set containing high-quality face images.

    This function processes face detection results to create a curated dataset
    of face stimuli. It filters for images with exactly one person and one face
    detection with sufficient area coverage, then sorts by face area to prioritize
    high-quality detections.

    Args:
        config: Configuration object containing pipeline settings and paths
    """
    # Check if positive set generation is enabled
    if config.pipeline.step_2_dataset_creation.generate_positive_set:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step,
                "Starting face set generation",
            )
        )
    else:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step,
                "Skipping face set generation",
            )
        )
        return

    # Get subjects to process
    subjects = subjects_list_unifier(
        config.pipeline.step_2_dataset_creation.subjects, False
    )

    # Process each subject
    for nsd_subj_subset in subjects:
        # Format subject identifier
        if nsd_subj_subset != "shared":
            nsd_subj_subset = f"subj_{nsd_subj_subset:02d}"

        logging.info(f"Generating face set for subject: {nsd_subj_subset}")

        # Create output directory structure
        subdir_path = os.path.join(
            config.directories.excel_files_target_dir, nsd_subj_subset
        )
        os.makedirs(
            os.path.dirname(
                os.path.join(
                    subdir_path, config.dataset_creation.subset_animate_face_unchecked
                )
            ),
            exist_ok=True,
        )

        # Load face detection results
        face_detection_results_path = os.path.join(
            subdir_path, config.face_detection.results_path
        )

        if not os.path.exists(face_detection_results_path):
            logging.error(
                f"Face detection results not found: {face_detection_results_path}"
            )
            continue

        with open(face_detection_results_path, "r") as f:
            detection_results = json.load(f)

        # Load base animate subset
        humans_subset_path = os.path.join(
            subdir_path, config.dataset_creation.subset_animate
        )

        if not os.path.exists(humans_subset_path):
            logging.error(f"Humans subset file not found: {humans_subset_path}")
            continue

        humans_subset_excel = pd.read_excel(humans_subset_path)

        logging.info(f"Loaded {len(detection_results)} detection results")
        logging.info(f"Loaded {len(humans_subset_excel)} animate subset entries")

        # Process face detections
        total_image_area = 640 * 640  # Standard image dimensions
        bboxes = []
        bbox_image_areas = []

        # Extract bounding box information from detection results
        for det in detection_results:
            bboxes.append(
                (
                    det["detection"],  # Bounding box coordinates
                    det["file_name"],  # Image filename
                    det["n_persons_label"],  # Number of persons detected
                    len(det["detection"]),  # Number of face detections
                )
            )

        logging.info(
            "Filtering for images with exactly 1 person and 1 face of sufficient size"
        )

        # Filter detections based on quality criteria
        for i, detection_record in enumerate(bboxes):
            detections, filename, n_persons, n_faces = detection_record

            # Process each bounding box in the detection
            for bbox in detections:
                # Calculate face area
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                area_fraction = round((area / total_image_area), 2)

                # Apply quality filters:
                # 1. Face area > 2% of total image area
                # 2. Exactly 1 person detected
                # 3. Exactly 1 face detected
                if area_fraction > 0.02 and n_persons == 1 and n_faces == 1:
                    bbox_image_areas.append(
                        (area_fraction, filename, n_persons, n_faces)
                    )

        # Sort by face area (largest first) to prioritize high-quality detections
        sorted_areas = sorted(bbox_image_areas, key=lambda x: x[0], reverse=True)

        # Remove duplicates (keep only the best detection per image)
        seen = set()
        unique_detections = [
            detection
            for detection in sorted_areas
            if detection[1] not in seen and not seen.add(detection[1])
        ]

        logging.info(
            f"Found {len(unique_detections)} unique high-quality face detections"
        )

        # Create result DataFrame
        result_df = pd.DataFrame(columns=humans_subset_excel.columns)

        for detection_info in unique_detections:
            area_fraction, filename, n_persons, n_faces = detection_info

            # Verify this is a single face detection
            if n_faces == 1:
                # Find corresponding row in humans subset
                matching_rows = humans_subset_excel[
                    humans_subset_excel["file_name"].str.contains(filename, na=False)
                ]

                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    result_df = pd.concat(
                        [result_df, pd.DataFrame([row])], ignore_index=True
                    )

        # Clean up unnamed columns and save results
        result_df = result_df.loc[
            :, ~result_df.columns.str.contains("Unnamed", case=False, na=False)
        ]

        output_path = os.path.join(
            subdir_path, config.dataset_creation.subset_animate_face_unchecked
        )
        result_df.to_excel(output_path, index=False)

        logging.info(f"Saved {len(result_df)} face stimuli to {output_path}")


# =========================================================================
# NEGATIVE SET GENERATION (NON-FACE ANIMATE STIMULI)
# =========================================================================


def generate_animate_non_face(config: Configuration):
    """
    Generate negative stimulus set containing animate non-face images.

    This function creates a dataset of animate stimuli that do not contain
    face detections, serving as a control/contrast set for face analysis.
    It uses set operations to identify images in the animate subset that
    lack face detections.

    Args:
        config: Configuration object containing pipeline settings and paths
    """
    # Check if non-face set generation is enabled
    if config.pipeline.step_2_dataset_creation.generate_non_face_set:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step,
                "Starting non-face set generation",
            )
        )
    else:
        logging.info(
            logging_message(
                config.pipeline.step_2_dataset_creation.step,
                "Skipping non-face set generation",
            )
        )
        return

    # Get subjects to process
    subjects = subjects_list_unifier(
        config.pipeline.step_2_dataset_creation.subjects, False
    )

    # Process each subject
    for nsd_subj_subset in subjects:
        # Format subject identifier
        if nsd_subj_subset != "shared":
            nsd_subj_subset = f"subj_{int(nsd_subj_subset):02d}"

        logging.info(f"Generating animate non-face set for subject: {nsd_subj_subset}")

        # Define subject directory path
        subdir_path = os.path.join(
            config.directories.excel_files_target_dir, nsd_subj_subset
        )

        # Load face detection results
        face_detection_results_path = os.path.join(
            subdir_path, config.face_detection.results_path
        )

        if not os.path.exists(face_detection_results_path):
            logging.error(
                f"Face detection results not found: {face_detection_results_path}"
            )
            continue

        with open(face_detection_results_path, "r") as f:
            detection_results = json.load(f)

        # Load base animate subset
        humans_subset_path = os.path.join(
            subdir_path, config.dataset_creation.subset_animate
        )

        if not os.path.exists(humans_subset_path):
            logging.error(f"Humans subset file not found: {humans_subset_path}")
            continue

        humans_subset_excel = pd.read_excel(humans_subset_path)

        logging.info(f"Loaded {len(detection_results)} detection results")
        logging.info(f"Loaded {len(humans_subset_excel)} animate subset entries")

        # Extract filenames that have face detections
        face_file_names = []
        for det in detection_results:
            # Only include files that actually have detected faces
            if len(det["detection"]) > 0:
                face_file_names.append(det["file_name"])

        face_file_names_set = set(face_file_names)

        # Extract all filenames from animate subset
        all_file_names = humans_subset_excel["file_name"].tolist()

        # Clean filenames by extracting just the filename part (remove path)
        cleaned_file_names = []
        for filename in all_file_names:
            # Extract filename from path if it contains "/"
            if "/" in filename:
                cleaned_filename = filename.split("/")[1]
            else:
                cleaned_filename = filename
            cleaned_file_names.append(cleaned_filename)

        all_file_names_set = set(cleaned_file_names)

        # Find non-face files using set difference
        non_face_set = list(all_file_names_set - face_file_names_set)

        # Log dataset statistics
        logging.info(f"Total animate files: {len(all_file_names_set)}")
        logging.info(f"Files with face detections: {len(face_file_names_set)}")
        logging.info(f"Non-face files: {len(non_face_set)}")

        # Create result DataFrame for non-face stimuli
        result_df = pd.DataFrame(columns=humans_subset_excel.columns)

        for filename in non_face_set:
            # Find corresponding row in humans subset
            matching_rows = humans_subset_excel[
                humans_subset_excel["file_name"].str.contains(filename, na=False)
            ]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                result_df = pd.concat(
                    [result_df, pd.DataFrame([row])], ignore_index=True
                )

        # Clean up unnamed columns
        result_df = result_df.loc[
            :, ~result_df.columns.str.contains("Unnamed", case=False, na=False)
        ]

        # Create output directory and save results
        output_path = os.path.join(
            subdir_path, config.dataset_creation.subset_animate_non_face_unchecked
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        result_df.to_excel(output_path, index=False)

        logging.info(f"Saved {len(result_df)} non-face stimuli to {output_path}")


# =========================================================================
# MAIN EXECUTION
# =========================================================================


def main():
    """
    Main execution function for dataset creation pipeline.

    Runs both positive set generation (faces) and negative set generation
    (animate non-faces) according to configuration settings.
    """
    logging.info("Starting dataset creation pipeline")

    # Load configuration
    config = load_config("config.yaml")

    # Generate both positive and negative stimulus sets
    generate_animate_non_face(config)
    generate_positive_set(config)

    logging.info("Dataset creation pipeline completed")


if __name__ == "__main__":
    main()
