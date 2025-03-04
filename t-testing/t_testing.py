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


def set_t_testing(config: Configuration, subjs_to_use: list, shared_set: bool):
    subjs_to_use_stringed = [str(f"{e:02d}") for e in subjs_to_use]

    if shared_set:
        with open(
            os.path.join(config.excel_files_target_dir, "missing_subjects.json"), "r"
        ) as f:
            missing_subjects = json.load(f)
    else:
        missing_subjects = {}

    for selected_subj in tqdm(subjs_to_use):
        if shared_set:
            positive_set_excel = pd.read_excel(
                os.path.join(
                    config.excel_files_target_dir,
                    "shared",
                    config.subset_animate_face_final,
                )
            )
            positive_set_filenames = positive_set_excel["file_name"].tolist()
            positive_set_filenames = [
                e.split("/")[1].split(".")[0] for e in positive_set_filenames
            ]

            negative_set_excel = pd.read_excel(
                os.path.join(
                    config.excel_files_target_dir,
                    "shared",
                    config.subset_animate_non_face_final,
                )
            )
            negative_set_filenames = negative_set_excel["file_name"].tolist()
            negative_set_filenames = [
                e.split("/")[1].split(".")[0] for e in negative_set_filenames
            ]
        else:
            positive_set_excel = pd.read_excel(
                os.path.join(
                    config.excel_files_target_dir,
                    f"subj_{selected_subj:02d}",
                    config.subset_animate_face_final,
                )
            )
            positive_set_filenames = positive_set_excel["file_name"].tolist()
            positive_set_filenames = [
                e.split("/")[1].split(".")[0] for e in positive_set_filenames
            ]

            negative_set_excel = pd.read_excel(
                os.path.join(
                    config.excel_files_target_dir,
                    f"subj_{selected_subj:02d}",
                    config.subset_animate_non_face_final,
                )
            )
            negative_set_filenames = negative_set_excel["file_name"].tolist()
            negative_set_filenames = [
                e.split("/")[1].split(".")[0] for e in negative_set_filenames
            ]

        positive_set_data = []
        negative_set_data = []

        for file_name in positive_set_filenames:
            if file_name in missing_subjects:
                if selected_subj in missing_subjects[file_name]:
                    logging.info(
                        f"Skipping {file_name} because it is missing in betas data!"
                    )
                    continue

            path = os.path.join(config.image_betas_dir, file_name)
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
            data = np.mean(concatenated, axis=0)

            positive_set_data.append(data)

        for file_name in negative_set_filenames:
            if file_name in missing_subjects:
                if selected_subj in missing_subjects[file_name]:
                    logging.info(
                        f"Skipping {file_name} because it is missing in betas data!"
                    )
                    continue

            path = os.path.join(config.image_betas_dir, file_name)
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
            data = np.mean(concatenated, axis=0)

            negative_set_data.append(data)

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
            results_file_path = os.path.join(config.t_test_results_dir, "shared")
        else:
            results_file_path = os.path.join(
                config.t_test_results_dir, f"subj_{selected_subj:02d}"
            )

        os.makedirs(results_file_path, exist_ok=True)

        results_file_path = os.path.join(
            results_file_path, f"result_subj_{selected_subj:02d}.npy"
        )

        np.save(results_file_path, voxel_results)


if __name__ == "__main__":
    shared = False
    config = load_config("config.yaml")
    subjects_to_use = config.subjects_to_analyze
    set_t_testing(config, subjects_to_use, shared)
