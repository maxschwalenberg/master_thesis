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
from datasetCreation.check_labelling import (
    adjust_labelled_data,
    correct_sets_for_nans_and_missing_samples,
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
    if config.pipeline.step_2_dataset_creation.download_data:
        download_data(config)

    if config.pipeline.step_2_dataset_creation.create_data_split:
        create_data_split(config)

    # 3.
    if config.pipeline.step_2_dataset_creation.face_detection:
        generate_face_detection_results(config)

    # 4.
    if config.pipeline.step_2_dataset_creation.generate_non_face_set:
        generate_animate_non_face(config)  # Generates non-face animate set

    if config.pipeline.step_2_dataset_creation.generate_positive_set:
        generate_positive_set(config)  # Generates positive face set

    load_betas_subset(config, overwrite=False)

    if config.pipeline.step_2_dataset_creation.remove_nans:
        # 5. remove NaNs
        subjects = subjects_list_unifier(
            config.pipeline.step_2_dataset_creation.subjects, False
        )
        for subj in subjects:
            if subj == "shared":
                fullset = True

                adjust_labelled_data(config, subj, fullset=fullset)
                correct_sets_for_nans_and_missing_samples(config, subj)
            else:
                fullset = False

                adjust_labelled_data(config, f"subj_{subj:02d}", fullset=fullset)
                correct_sets_for_nans_and_missing_samples(config, f"subj_{subj:02d}")

    # 6. label!
    if config.pipeline.step_2_dataset_creation.label_data:
        pass
    raise NotImplementedError("Need to label")


def pipeline_t_testing(config: Configuration):
    # 1. generate t-testing results
    subjects = subjects_list_unifier(config.pipeline.step_3_t_testing.subjects, False)

    if config.pipeline.step_3_t_testing.t_testing:
        logging.info(
            logging_message(config.pipeline.step_3_t_testing.step, "Starting T-Testing")
        )

    else:
        logging.info(
            logging_message(config.pipeline.step_3_t_testing.step, "Skipping T-Testing")
        )

    if config.pipeline.step_3_t_testing.t_testing:
        if "shared" in subjects:
            shared = True
            subjects_to_use = list(range(1, 8 + 1))
            set_t_testing(config, subjects_to_use, shared, True)

            subjects.remove("shared")
            shared = False
            set_t_testing(config, subjects, shared, True)

        else:
            shared = False
            set_t_testing(config, subjects, shared, True)

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

        for sub in subject_list:
            if set_to_take == "shared":
                shared_bool = True
            else:
                shared_bool = False

            modify_mask_with_ttest(config, 3.0, shared_bool, sub)

        for mask_value in mask_values:
            create_rdm(
                config,
                subject_list,
                mask_value,
                set_to_take,
                3.0,
                mode="averaged",
                randomization=True,
            )


def pipeline_gaussian(config: Configuration):
    subjects = subjects_list_unifier(
        config.pipeline.step_5_gaussian_fitting.subjects, False
    )

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

    from rsa.sample_repeatability import sample_repeatability_distances_data
    from rsa.cortical_correlation import main as main_cortical
    from rsa.permutation_analysis import generate_results

    # generate_results()
    # sample_repeatability_distances_data(config, range(1, 8+1))
    # main_cortical()

    # 1. Extract and preprocess nsd data
    if config.pipeline.step_1_preprocessing.execute_step:
        extract_nsd_data(config)

    # 2.
    if config.pipeline.step_2_dataset_creation.execute_step:
        pipeline_labelling(config)

    # 3. perform t-testing - create siginficance maps
    if config.pipeline.step_3_t_testing.execute_step:
        pipeline_t_testing(config)

    # 4. perform RSA; create RDM and MDS spaces
    if config.pipeline.step_4_rsa_analysis.execute_step:
        pipeline_rsa(config)

    # 5. fit gaussian in mds space
    if config.pipeline.step_5_gaussian_fitting.execute_step:
        pipeline_gaussian(config)
