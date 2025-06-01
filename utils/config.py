from dataclasses import dataclass
import yaml
from typing import List, Dict, Union


@dataclass
class PreprocessingConfig:
    extract_nsd_data: bool
    execute_step: bool

    subjects: List[str]  # List of subjects, including "shared"
    step: int


@dataclass
class DatasetCreationConfig:
    execute_step: bool

    download_data: bool
    create_data_split: bool
    face_detection: bool
    generate_non_face_set: bool
    generate_positive_set: bool
    remove_nans: bool
    label_data: bool
    load_betas: bool
    subjects: List[str]
    step: int


@dataclass
class TTestingConfig:
    execute_step: bool

    t_testing: bool
    step: int
    subjects: List[str]

    thresholds_map: List[float]
    clipping_value: int


@dataclass
class RSAAnalysisConfig:
    execute_step: bool

    only_face_set: bool
    distance_metric: str
    rsa_analysis: bool
    step: int
    subjects: List[str]


@dataclass
class GaussianFittingConfig:
    execute_step: bool

    fit_gaussian: bool
    visualize: bool
    evaluate: bool
    sigma_filtering: Union[int, int]

    t_test_filtering: float
    step: int
    subjects: List[str]


@dataclass
class PipelineConfig:
    step_1_preprocessing: PreprocessingConfig
    step_2_dataset_creation: DatasetCreationConfig
    step_3_t_testing: TTestingConfig
    step_4_rsa_analysis: RSAAnalysisConfig
    step_5_gaussian_fitting: GaussianFittingConfig


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
    output_dir: str
    activations_dir: str


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
    rois_to_analyze: Dict[str, int]  # Now a list instead of a dictionary


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
    pipeline: PipelineConfig  # Add the pipeline configuration

    all_rois: Dict[str, int] 


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
        pipeline=PipelineConfig(
            step_1_preprocessing=PreprocessingConfig(
                **config_data["pipeline"]["step_1_preprocessing"]
            ),
            step_2_dataset_creation=DatasetCreationConfig(
                **config_data["pipeline"]["step_2_dataset_creation"]
            ),
            step_3_t_testing=TTestingConfig(
                **config_data["pipeline"]["step_3_t_testing"]
            ),
            step_4_rsa_analysis=RSAAnalysisConfig(
                **config_data["pipeline"]["step_4_rsa_analysis"]
            ),
            step_5_gaussian_fitting=GaussianFittingConfig(
                **config_data["pipeline"]["step_5_gaussian_fitting"]
            ),
        ),
        all_rois=config_data["all_rois"],
    )

    # Ensure dataset validation settings are correct
    assert all(
        subject in ["shared", "1", "2", "3", "4", "5", "6", "7", "8"]
        for subject in config.dataset_validation.nsd_samples_subjects_to_check
    )

    return config
