from dataclasses import dataclass
import yaml


@dataclass
class Configuration:
    """
    Dataclass that loads configuration parameters from a YAML file.
    """

    # specifying subfolder names in data dir
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

    # nsd_negative_subset: str
    # nsd_positive_subset: str

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

    # for dataset creation

    nsd_samples_subjects_to_check: list[str]  # ['shared', '1', ..., '8']

    # dataset creation file names:
    # step 1
    subset_non_animate: str
    subset_animate: str

    # step 2 - face detection -> generate positive and negative set
    subset_animate_face_unchecked: str
    subset_animate_non_face_unchecked: str

    # step 3 - remove nans
    subset_animate_face_nans_removed: str
    subset_animate_non_face_nans_removed: str

    # step 4 - labelled (checked labelling)
    subset_animate_face_labelled: str
    subset_animate_non_face_labelled: str

    # step 5 - final
    subset_animate_face_final: str
    subset_animate_non_face_final: str


def load_config(config_file_path: str):

    with open(config_file_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = Configuration(**config_data)

    # assert config is correct
    assert (
        len(
            list(
                set(config.nsd_samples_subjects_to_check)
                - set(["shared", "1", "2", "3", "4", "5", "6", "7", "8"])
            )
        )
        == 0
    )

    return config
