import pandas as pd
import os
import glob
import numpy as np
from scipy import stats
from tqdm import tqdm
import json

import logging

from utils.config import Configuration, load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def detect_nan_rows_and_columns(data, row_nan_threshold=0.5, column_nan_threshold=0.5):
    """
    Detect number of rows and columns with a high percentage of NaN values.

    Parameters:
    -----------
    data : numpy.ndarray
        Input array of shape (samples, dimensionality)
    row_nan_threshold : float, optional (default=0.5)
        Fraction of NaNs in a row to consider it mostly NaN
        (e.g., 0.5 means more than 50% of values are NaN)
    column_nan_threshold : float, optional (default=0.5)
        Fraction of NaNs in a column to consider it mostly NaN

    Returns:
    --------
    dict containing:
    - 'total_problematic_rows': number of rows with high NaN percentage
    - 'total_problematic_columns': number of columns with high NaN percentage
    - 'row_nan_percentages': percentage of NaNs in each row
    - 'column_nan_percentages': percentage of NaNs in each column
    """
    # Ensure input is a NumPy array
    data = np.asarray(data)

    total_nans = np.sum(np.isnan(data))

    # Calculate NaN percentages for rows
    row_nan_percentages = np.isnan(data).mean(axis=1)
    total_problematic_rows = np.sum(row_nan_percentages >= row_nan_threshold)

    # Calculate NaN percentages for columns
    column_nan_percentages = np.isnan(data).mean(axis=0)
    total_problematic_columns = np.sum(column_nan_percentages >= column_nan_threshold)

    return {
        "total_nans": total_nans,
        "total_problematic_rows": total_problematic_rows,
        "total_problematic_columns": total_problematic_columns,
        #'row_nan_percentages': row_nan_percentages,
        #'column_nan_percentages': column_nan_percentages
    }


def set_voxelwise_outlier_detection(
    config: Configuration, subjs_to_use: list, shared_set: bool, t_threshold: float = 2
):
    """
    For each subject, this function:
      1. Loads the previously computed voxelwise t-test results.
      2. Selects only those voxels for which the t-value is greater than t_threshold.
      3. Loads the positive and negative beta data (each sample is a vector of voxel values).
      4. For each selected voxel, computes the "centroid" (mean value) for the positive and negative groups.
      5. For each sample within that voxel, computes:
            own_distance = |sample - group centroid|
            other_distance = |sample - opposite group's centroid|
         and then computes score = (other_distance - own_distance).
      6. Saves, for each voxel (and for each sample within that voxel), a row of results containing:
            [voxel index, group, sample index, own_distance, other_distance, score]

    The results are saved as a .npy file for each subject.
    """
    import os, json, glob, logging
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy import stats

    # --- Load missing subjects list if available
    if shared_set:
        missing_subjects_file = os.path.join(
            config.directories.excel_files_target_dir, "missing_subjects.json"
        )
        with open(missing_subjects_file, "r") as f:
            missing_subjects = json.load(f)
    else:
        missing_subjects = {}

    # Process each subject
    for selected_subj in tqdm(subjs_to_use, desc="Processing subjects"):
        logging.info(
            f"Performing voxelwise outlier detection for subject {selected_subj:02d}"
        )

        # --- Load the existing t-test results for this subject
        if shared_set:
            t_results_dir = os.path.join(
                config.directories.t_test_results_dir, "shared"
            )
        else:
            t_results_dir = os.path.join(
                config.directories.t_test_results_dir, f"subj_{selected_subj:02d}"
            )
        t_results_filepath = os.path.join(
            t_results_dir, f"result_subj_{selected_subj:02d}.npy"
        )
        if not os.path.exists(t_results_filepath):
            logging.error(
                f"t-test results file not found for subject {selected_subj:02d}: {t_results_filepath}"
            )
            continue

        ttest_results = np.load(
            t_results_filepath
        )  # Expect shape (n_voxels, 2): [t_value, p_value]
        # Get indices of voxels with t-value > threshold. (Adjust if you need absolute threshold.)
        selected_voxel_indices = np.where(ttest_results[:, 0] > t_threshold)[0]
        logging.info(
            f"Subject {selected_subj:02d}: {len(selected_voxel_indices)} voxels pass t > {t_threshold}"
        )

        # --- Load beta data for positive (face) and negative (non-face) sets
        # The following blocks are very similar to the ones in your t-test script.
        if shared_set:
            pos_excel_path = os.path.join(
                config.directories.excel_files_target_dir,
                "shared",
                config.dataset_creation.subset_animate_face_final,
            )
            positive_set_excel = pd.read_excel(pos_excel_path)
            positive_set_filenames = [
                e.split("/")[1].split(".")[0]
                for e in positive_set_excel["file_name"].tolist()
            ]

            neg_excel_path = os.path.join(
                config.directories.excel_files_target_dir,
                "shared",
                config.dataset_creation.subset_animate_non_face_final,
            )
            negative_set_excel = pd.read_excel(neg_excel_path)
            negative_set_filenames = [
                e.split("/")[1].split(".")[0]
                for e in negative_set_excel["file_name"].tolist()
            ]
        else:
            pos_excel_path = os.path.join(
                config.directories.excel_files_target_dir,
                f"subj_{selected_subj:02d}",
                config.dataset_creation.subset_animate_face_final,
            )
            positive_set_excel = pd.read_excel(pos_excel_path)
            positive_set_filenames = [
                e.split("/")[1].split(".")[0]
                for e in positive_set_excel["file_name"].tolist()
            ]

            neg_excel_path = os.path.join(
                config.directories.excel_files_target_dir,
                f"subj_{selected_subj:02d}",
                config.dataset_creation.subset_animate_non_face_final,
            )
            negative_set_excel = pd.read_excel(neg_excel_path)
            negative_set_filenames = [
                e.split("/")[1].split(".")[0]
                for e in negative_set_excel["file_name"].tolist()
            ]

        # --- Load positive beta data
        positive_set_data = []
        for file_name in positive_set_filenames:
            if file_name in missing_subjects:
                if str(selected_subj) in missing_subjects[file_name]:
                    logging.info(
                        f"Skipping {file_name} because it is missing in betas data!"
                    )
                    continue

            path = os.path.join(config.directories.image_betas_dir, file_name)
            subj_path = os.path.join(path, f"subj_{selected_subj:02d}")
            files = glob.glob(os.path.join(subj_path, "*.npy"))
            for file in files:
                array = np.load(file)
                # Ensure there are no NaNs in the sample
                if np.isnan(np.sum(array)):
                    raise ValueError(f"NaN values found in {file}")
                positive_set_data.append(array)
        positive_set_data = np.array(
            positive_set_data
        )  # shape: (n_positive_samples, n_voxels)

        # --- Load negative beta data
        negative_set_data = []
        for file_name in negative_set_filenames:
            if file_name in missing_subjects:
                if str(selected_subj) in missing_subjects[file_name]:
                    logging.info(
                        f"Skipping {file_name} because it is missing in betas data!"
                    )
                    continue

            path = os.path.join(config.directories.image_betas_dir, file_name)
            subj_path = os.path.join(path, f"subj_{selected_subj:02d}")
            files = glob.glob(os.path.join(subj_path, "*.npy"))
            for file in files:
                array = np.load(file)
                if np.isnan(np.sum(array)):
                    raise ValueError(f"NaN values found in {file}")
                negative_set_data.append(array)
        negative_set_data = np.array(
            negative_set_data
        )  # shape: (n_negative_samples, n_voxels)

        # Optionally, you could call your detect_nan_rows_and_columns function here:
        # print(f"positive: {detect_nan_rows_and_columns(positive_set_data)}")
        # print(f"negative: {detect_nan_rows_and_columns(negative_set_data)}")

        # --- Now, for each voxel that passed the t-threshold, perform outlier detection.
        # Prepare a list to collect the outlier detection results.
        # Each row will have:
        # [voxel index, group, sample index, own_distance, other_distance, score]
        outlier_voxel_results = []

        for voxel_index in tqdm(selected_voxel_indices):
            # Retrieve voxel data for both groups (each is a scalar per sample)
            pos_voxel_values = positive_set_data[:, voxel_index]
            neg_voxel_values = negative_set_data[:, voxel_index]

            # Compute the centroids (mean value for that voxel within each group)
            pos_centroid = np.mean(pos_voxel_values)
            neg_centroid = np.mean(neg_voxel_values)

            # For the positive group samples:
            for sample_idx, value in enumerate(pos_voxel_values):
                own_distance = abs(value - pos_centroid)
                other_distance = abs(value - neg_centroid)
                score = other_distance - own_distance
                outlier_voxel_results.append(
                    [
                        voxel_index,
                        "positive",
                        sample_idx,
                        own_distance,
                        other_distance,
                        score,
                    ]
                )

            # For the negative group samples:
            for sample_idx, value in enumerate(neg_voxel_values):
                own_distance = abs(value - neg_centroid)
                other_distance = abs(value - pos_centroid)
                score = other_distance - own_distance
                outlier_voxel_results.append(
                    [
                        voxel_index,
                        "negative",
                        sample_idx,
                        own_distance,
                        other_distance,
                        score,
                    ]
                )

        outlier_voxel_results = np.array(outlier_voxel_results)

        # --- Save the outlier detection results to file
        if shared_set:
            results_dir = os.path.join(config.directories.t_test_results_dir, "shared")
        else:
            results_dir = os.path.join(
                config.directories.t_test_results_dir, f"subj_{selected_subj:02d}"
            )
        os.makedirs(results_dir, exist_ok=True)
        results_filepath = os.path.join(
            results_dir, f"voxel_outlier_result_subj_{selected_subj:02d}.npy"
        )
        np.save(results_filepath, outlier_voxel_results)
        logging.info(
            f"Saved voxelwise outlier results for subject {selected_subj:02d} at {results_filepath}"
        )


def set_t_testing(config: Configuration, subjs_to_use: list, shared_set: bool):
    subjs_to_use_stringed = [str(f"{e:02d}") for e in subjs_to_use]

    if shared_set:
        with open(
            os.path.join(
                config.directories.excel_files_target_dir, "missing_subjects.json"
            ),
            "r",
        ) as f:
            missing_subjects = json.load(f)
    else:
        missing_subjects = {}

    for selected_subj in tqdm(subjs_to_use):
        logging.info(f"Generating for subject {selected_subj}")
        run_both = False
        negative_sets = []

        if shared_set:
            positive_set_excel = pd.read_excel(
                os.path.join(
                    config.directories.excel_files_target_dir,
                    "shared",
                    config.dataset_creation.subset_animate_face_final,
                )
            )
            positive_set_filenames = positive_set_excel["file_name"].tolist()
            positive_set_filenames = [
                e.split("/")[1].split(".")[0] for e in positive_set_filenames
            ]

            negative_set_excel = pd.read_excel(
                os.path.join(
                    config.directories.excel_files_target_dir,
                    "shared",
                    config.dataset_creation.subset_animate_non_face_final,
                )
            )

            labels = negative_set_excel["label"].tolist()
            set_labels = list(set(labels))
            if "personen" in set_labels or "animals" in set_labels:
                run_both = True

            if run_both:
                negative_set_filenames = negative_set_excel[
                    negative_set_excel["label"] == "animals"
                ]["file_name"].tolist()
                negative_set_filenames = [
                    e.split("/")[1].split(".")[0] for e in negative_set_filenames
                ]

                negative_sets.append(("animals", negative_set_filenames))

                negative_set_filenames = negative_set_excel[
                    negative_set_excel["label"] == "personen"
                ]["file_name"].tolist()
                negative_set_filenames = [
                    e.split("/")[1].split(".")[0] for e in negative_set_filenames
                ]

                negative_sets.append(("personen", negative_set_filenames))

            else:
                negative_set_filenames = negative_set_excel["file_name"].tolist()
                negative_set_filenames = [
                    e.split("/")[1].split(".")[0] for e in negative_set_filenames
                ]

                negative_sets.append(("animate", negative_set_filenames))
        else:
            positive_set_excel = pd.read_excel(
                os.path.join(
                    config.directories.excel_files_target_dir,
                    f"subj_{selected_subj:02d}",
                    config.dataset_creation.subset_animate_face_final,
                )
            )
            positive_set_filenames = positive_set_excel["file_name"].tolist()
            positive_set_filenames = [
                e.split("/")[1].split(".")[0] for e in positive_set_filenames
            ]

            negative_set_excel = pd.read_excel(
                os.path.join(
                    config.directories.excel_files_target_dir,
                    f"subj_{selected_subj:02d}",
                    config.dataset_creation.subset_animate_non_face_final,
                )
            )

            labels = negative_set_excel["label"].tolist()
            set_labels = list(set(labels))
            if "personen" in set_labels or "animals" in set_labels:
                run_both = True

            print(f"{run_both=}\t{set_labels=}")

            if run_both:
                negative_set_filenames = negative_set_excel[
                    negative_set_excel["label"] == "animals"
                ]["file_name"].tolist()
                negative_set_filenames = [
                    e.split("/")[1].split(".")[0] for e in negative_set_filenames
                ]

                negative_sets.append(("animals", negative_set_filenames))

                negative_set_filenames = negative_set_excel[
                    negative_set_excel["label"] == "personen"
                ]["file_name"].tolist()
                negative_set_filenames = [
                    e.split("/")[1].split(".")[0] for e in negative_set_filenames
                ]

                negative_sets.append(("personen", negative_set_filenames))

            else:
                negative_set_filenames = negative_set_excel["file_name"].tolist()
                negative_set_filenames = [
                    e.split("/")[1].split(".")[0] for e in negative_set_filenames
                ]

                negative_sets.append(("animate", negative_set_filenames))

        positive_set_data = []

        for file_name in positive_set_filenames:
            if file_name in missing_subjects:
                if selected_subj in missing_subjects[file_name]:
                    logging.info(
                        f"Skipping {file_name} because it is missing in betas data!"
                    )
                    continue

            path = os.path.join(config.directories.image_betas_dir, file_name)
            subj_path = os.path.join(path, f"subj_{selected_subj:02d}")

            files = glob.glob(os.path.join(subj_path, "*.npy"))

            # ---------------Average
            arrays = []

            # Load each file and append the array to the list
            for file in files:
                array = np.load(file)
                is_nan = np.isnan(np.sum(array))
                assert not is_nan, f"NaN values found in {file}"
                arrays.append(array)

            # Concatenate all arrays along a new axis
            concatenated = np.stack(arrays, axis=0)

            # Compute the average across the new axis
            # data = np.mean(concatenated, axis=0)
            for sample in concatenated:
                positive_set_data.append(sample)

            # positive_set_data.append(data)

        for prefix_result_file, negative_set_filenames in negative_sets:
            logging.info(f"Generating for {prefix_result_file}")
            negative_set_data = []

            for file_name in negative_set_filenames:
                if file_name in missing_subjects:
                    if selected_subj in missing_subjects[file_name]:
                        logging.info(
                            f"Skipping {file_name} because it is missing in betas data!"
                        )
                        continue

                path = os.path.join(config.directories.image_betas_dir, file_name)
                subj_path = os.path.join(path, f"subj_{selected_subj:02d}")

                files = glob.glob(os.path.join(subj_path, "*.npy"))

                # ---------------Average
                arrays = []

                # Load each file and append the array to the list
                for file in files:
                    array = np.load(file)
                    is_nan = np.isnan(np.sum(array))
                    assert not is_nan, f"NaN values found in {file}"
                    arrays.append(array)

                # Concatenate all arrays along a new axis
                concatenated = np.stack(arrays, axis=0)

                # Compute the average across the new axis
                # data = np.mean(concatenated, axis=0)
                for sample in concatenated:
                    negative_set_data.append(sample)

                # negative_set_data.append(data)

            positive_set_data = np.array(positive_set_data)
            negative_set_data = np.array(negative_set_data)

            # detect_nan_rows_and_columns
            print(f"positive: {detect_nan_rows_and_columns(positive_set_data)}")
            print(f"negative: {detect_nan_rows_and_columns(negative_set_data)}")

            logging.info("Generating statistics...")
            voxel_results = []
            for i in tqdm(range(positive_set_data.shape[1])):
                positive_voxel_set = positive_set_data[:, i]
                negative_voxel_set = negative_set_data[:, i]

                t_statistic, p_value = stats.ttest_ind(
                    positive_voxel_set,
                    negative_voxel_set,
                    equal_var=False,
                )

                voxel_results.append([t_statistic, p_value])

            voxel_results = np.array(voxel_results)

            if shared_set:
                results_file_path = os.path.join(
                    config.directories.t_test_results_dir, "shared"
                )
            else:
                results_file_path = os.path.join(
                    config.directories.t_test_results_dir, f"subj_{selected_subj:02d}"
                )

            os.makedirs(results_file_path, exist_ok=True)

            results_file_path = os.path.join(
                results_file_path,
                f"result_{prefix_result_file}_subj_{selected_subj:02d}.npy",
            )

            np.save(results_file_path, voxel_results)


if __name__ == "__main__":
    shared = False
    config = load_config("config.yaml")
    subjects_to_use = [1]
    # set_t_testing(config, subjects_to_use, shared)
    set_voxelwise_outlier_detection(config, subjects_to_use, shared)
