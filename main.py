import logging


from utils.config import Configuration, load_config
from utils.utils import subjects_list_unifier, logging_message

from nsd_processing.nsd_preprocessing import extract_nsd_data

from datasetCreation.load_betas import load_betas_subset
from datasetCreation.create_data_split import download_data, create_data_split
from datasetCreation.face_detection import generate_face_detection_results
from datasetCreation.generate_faces_set import (
    generate_animate_non_face,
    generate_positive_set,
)

from t_testing.t_testing import set_t_testing
from t_testing.stat_to_mgz import t_test_results_to_mgz
from t_testing.clean_roi_mask import modify_mask_with_ttest


from rsa.create_rdm import create_rdm

from gaussian.fit_gaussian import fit_gaussian_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def pipeline_labelling(config: Configuration):
    # 2.
    download_data(config)
    create_data_split(config)

    # 3.
    generate_face_detection_results(config)

    # 4.
    generate_animate_non_face(config)  # Generates non-face animate set
    generate_positive_set(config)  # Generates positive face set

    # 5. remove NaNs
    raise NotImplementedError(f"Needs improvement in the NaN script")

    # 6. label!
    raise NotImplementedError("Need to label")

    # x. Load single betas
    assert len(config.dataset_validation.nsd_samples_subjects_to_check) == 1

    subj_to_pick = (
        "shared"
        if config.dataset_validation.nsd_samples_subjects_to_check[0] == "shared"
        else f"subj_{int(config.dataset_validation.nsd_samples_subjects_to_check[0]):02d}"
    )

    load_betas_subset(config, overwrite=False, subj_to_pick=subj_to_pick)

    # 3.


def pipeline_t_testing(config: Configuration):
    # 1. generate t-testing results
    subjects = subjects_list_unifier(config.pipeline.step_3_t_testing.subjects, False)

    if config.pipeline.step_1_preprocessing.extract_nsd_data:
        logging.info(
            logging_message(config.pipeline.step_3_t_testing.step, "Starting T-Testing")
        )

    else:
        logging.info(
            logging_message(config.pipeline.step_3_t_testing.step, "Skipping T-Testing")
        )
    if "shared" in subjects:
        shared = True
        subjects_to_use = list(range(1, 8 + 1))
        set_t_testing(config, subjects_to_use, shared)

        subjects.remove("shared")
        shared = False
        set_t_testing(config, subjects, shared)

    else:
        shared = False
        set_t_testing(config, subjects, shared)

    subjects = subjects_list_unifier(config.pipeline.step_3_t_testing.subjects, False)

    for subject in subjects:
        if subject != "shared":
            shared = False
            # t_test_subdirs = [f"subj_{e:02d}" for e in config.analysis.subjects_to_analyze]
            t_test_subdir = f"subj_{subject:02d}"
            for labels_subdir in ["face_animate_new"]:
                for mode in ["absolute", "signed"]:
                    for threshold in config.pipeline.step_3_t_testing.thresholds_map:
                        if threshold == 0.0:
                            thresholding = False
                        else:
                            thresholding = True

                        t_test_results_to_mgz(
                            config,
                            mode,
                            threshold,
                            labels_subdir,
                            shared,
                            thresholding=thresholding,
                            t_test_results_subdir=t_test_subdir,
                            clipping_value=config.pipeline.step_3_t_testing.clipping_value,
                        )
        else:
            shared = True
            for t_test_subdir in [f"subj_{subject:02d}" for subject in range(1, 8 + 1)]:
                for labels_subdir in ["face_animate_new"]:
                    for mode in ["absolute", "signed"]:
                        for (
                            threshold
                        ) in config.pipeline.step_3_t_testing.thresholds_map:
                            if threshold == 0.0:
                                thresholding = False
                            else:
                                thresholding = True

                            t_test_results_to_mgz(
                                config,
                                mode,
                                threshold,
                                labels_subdir,
                                shared,
                                thresholding=thresholding,
                                t_test_results_subdir=t_test_subdir,
                                clipping_value=config.pipeline.step_3_t_testing.clipping_value,
                            )

    logging.info(f"Now draw ROIs in Matlab")


def pipeline_rsa(config: Configuration):
    subjects = subjects_list_unifier(
        config.pipeline.step_4_rsa_analysis.subjects, False
    )

    if config.pipeline.step_4_rsa_analysis.rsa_analysis:
        logging.info(
            logging_message(config.pipeline.step_4_rsa_analysis.step, "Starting RSA")
        )

    else:
        logging.info(
            logging_message(config.pipeline.step_4_rsa_analysis.step, "Skipping RSA")
        )
    # modify_mask_with_ttest()
    mask_values = list(config.analysis.rois_to_analyze.values())
    for subject in subjects:
        if subject == "shared":
            set_to_take = "shared"
            subject_list = range(1, 8 + 1)

        else:
            set_to_take = f"subj_{subject:02d}"
            subject_list = [subject]

        for mask_value in mask_values:
            create_rdm(config, subject_list, mask_value, set_to_take, mode="averaged")


def pipeline_gaussian(config: Configuration):
    subjects = subjects_list_unifier(config.pipeline.step_5_gaussian_fitting.subjects)

    if config.pipeline.step_1_preprocessing.extract_nsd_data:
        logging.info(
            logging_message(
                config.pipeline.step_5_gaussian_fitting.step,
                "Starting Gaussian Fitting",
            )
        )

    else:
        logging.info(
            logging_message(
                config.pipeline.step_5_gaussian_fitting.step,
                "Skipping Gaussian Fitting",
            )
        )

    for subject in subjects:
        if subject == "shared":
            set_to_take = "shared"
            subject_list = range(1, 8 + 1)

        else:
            set_to_take = f"subj_{subject:02d}"
            subject_list = [subject]

        fit_gaussian_params(config, subject_list, set_to_take)


if __name__ == "__main__":
    config = load_config("config.yaml")
    # 1. Extract and preprocess nsd data
    extract_nsd_data(config)

    # 2.
    pipeline_labelling(config)

    # 3. perform t-testing - create siginficance maps
    pipeline_t_testing(config)

    # 4. perform RSA; create RDM and MDS spaces
    pipeline_rsa(config)

    # 5. fit gaussian in mds space
    pipeline_gaussian(config)
