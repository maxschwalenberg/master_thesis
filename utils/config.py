from dataclasses import dataclass
import yaml


@dataclass
class Configuration:
    """
    Dataclass that loads configuration parameters from a YAML file.
    """

    images_target_dir: str
    excel_files_target_dir: str
    image_betas_dir: str
    t_test_results_dir: str
    t_test_roi_dir: str
    mds_dir: str
    rdm_dir: str

    full_brain_data_dir: str
    stans_thesis_repo_data: str
    mask_data_dir: str
    freesurfer_dir: str

    coco_train_json_path: str
    coco_val_json_path: str

    nsd_coco_file_path: str

    nsd_labeled_subset_path: str

    nsd_labeled_subset_non_animals: str
    nsd_labeled_subset_animals_humans: str

    nsd_negative_subset: str
    nsd_positive_subset: str

    face_detection_results_path: str

    nsd_data_dir: str
    mask_subdir: str
    label_subdir: str
    conds_subdir: str
    nsd_subdir: str

    proj_dir: str
    betas_subdir: str

    subjects_to_analyze: list
    rois_to_analyze: dict[str, int]

    gaussian_fit_results_dir: str


def load_config(config_file_path: str):

    with open(config_file_path, "r") as f:
        config_data = yaml.safe_load(f)

    return Configuration(**config_data)
