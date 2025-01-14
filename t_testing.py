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


def set_t_testing(config: Configuration, subjs_to_use: list):
    subjs_to_use = [str(f"{e:02d}") for e in subjs_to_use]

    positive_set_excel = pd.read_excel(
        os.path.join(config.excel_files_target_dir, config.nsd_positive_subset)
    )
    positive_set_filenames = positive_set_excel["file_name"].tolist()
    positive_set_filenames = [
        e.split("/")[1].split(".")[0] for e in positive_set_filenames
    ]

    negative_set_excel = pd.read_excel(
        os.path.join(config.excel_files_target_dir, config.nsd_negative_subset)
    )
    negative_set_filenames = negative_set_excel["file_name"].tolist()
    negative_set_filenames = [
        e.split("/")[1].split(".")[0] for e in negative_set_filenames
    ]

    with open(
        os.path.join(config.excel_files_target_dir, "missing_subjects.json"), "r"
    ) as f:
        missing_subjects = json.load(f)

    for selected_subj in tqdm(subjects_to_use):
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

            file_path = files[0]
            data = np.load(file_path)

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

            file_path = files[0]
            data = np.load(file_path)

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

        results_file_path = config.t_test_results_dir
        os.makedirs(results_file_path, exist_ok=True)

        results_file_path = os.path.join(
            results_file_path, f"result_subj_{selected_subj:02d}.npy"
        )

        np.save(results_file_path, voxel_results)


if __name__ == "__main__":
    subjects_to_use = list(range(1, 8 + 1))
    config = load_config("config.yaml")
    set_t_testing(config, subjects_to_use)
