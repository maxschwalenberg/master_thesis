from dataclasses import dataclass
import yaml
from typing import List, Dict


@dataclass
class Directories:
    images_target_dir: str
    excel_files_target_dir: str
    image_betas_dir: str
    t_test_results_dir: str
    t_test_roi_dir: str
    mds_dir: str
    rdm_dir: str
    gaussian_fit_results_dir: str


@dataclass
class NSDData:
    nsddata_responses_tsv_dir: str
    nsddata_betas_dir: str
    full_brain_data_dir: str
    stans_thesis_repo_data: str
    mask_data_dir: str
    freesurfer_dir: str


@dataclass
class COCOData:
    coco_train_json_path: str
    coco_val_json_path: str
    nsd_coco_file_path: str
    nsd_labeled_subset_path: str


@dataclass
class DatasetCreation:
    subset_non_animate: str
    subset_animate: str
    subset_animate_face_unchecked: str
    subset_animate_non_face_unchecked: str
    subset_animate_face_nans_removed: str
    subset_animate_non_face_nans_removed: str
    subset_animate_face_labelled: str
    subset_animate_non_face_labelled: str
    subset_animate_face_final: str
    subset_animate_non_face_final: str


@dataclass
class FaceDetection:
    results_path: str


@dataclass
class NSDProject:
    nsd_data_dir: str
    mask_subdir: str
    label_subdir: str
    conds_subdir: str
    nsd_subdir: str
    proj_dir: str
    betas_subdir: str


@dataclass
class Analysis:
    subjects_to_analyze: List[int]
    rois_to_analyze: List[int]  # Now a list instead of a dictionary


@dataclass
class DatasetValidation:
    nsd_samples_subjects_to_check: List[str]


@dataclass
class Configuration:
    directories: Directories
    nsd_data: NSDData
    coco_data: COCOData
    dataset_creation: DatasetCreation
    face_detection: FaceDetection
    nsd_project: NSDProject
    analysis: Analysis
    dataset_validation: DatasetValidation


def load_config(config_file_path: str) -> Configuration:
    with open(config_file_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = Configuration(
        directories=Directories(**config_data["directories"]),
        nsd_data=NSDData(**config_data["nsd_data"]),
        coco_data=COCOData(**config_data["coco_data"]),
        dataset_creation=DatasetCreation(**config_data["dataset_creation"]),
        face_detection=FaceDetection(**config_data["face_detection"]),
        nsd_project=NSDProject(**config_data["nsd_project"]),
        analysis=Analysis(**config_data["analysis"]),
        dataset_validation=DatasetValidation(**config_data["dataset_validation"]),
    )

    # Ensure dataset validation settings are correct
    assert all(
        subject in ["shared", "1", "2", "3", "4", "5", "6", "7", "8"]
        for subject in config.dataset_validation.nsd_samples_subjects_to_check
    )

    return config
