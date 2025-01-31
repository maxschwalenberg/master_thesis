import os

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from tqdm import tqdm


from utils.config import load_config, Configuration
from utils.utils import (
    retrieve_stacked_betas,
    retrieve_roi_mask,
    filter_roi_mask,
    retrieve_stacked_betas_test,
)

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def gaussian_2d_curve(independent, x0, y0, sigma, slope, intercept):
    X, Y = independent
    x = X - x0
    y = Y - y0
    return np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2)) * slope + intercept


def gaussian_2d_curve_pol(independent, ecc, pol, sigma, slope, intercept):
    X, Y = independent
    x0, y0 = pol2cart(ecc, pol)
    x = X - x0
    y = Y - y0
    return np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma) ** 2)) * slope + intercept


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


params = {
    "random": True,
    "initial": (np.array([-0.4, -0.4, 0.01, 0.1, -2]), np.array([0.4, 0.4, 2, 10, 2])),
    "bounds": (
        np.array([-1.05, -1.05, 0.01, 1e-08, -np.inf]),
        np.array([1.05, 1.05, np.inf, np.inf, np.inf]),
    ),
    "loss": "linear",
    "method": "trf",
}


def fit_gaussian_params(config: Configuration, subj_list: list[int]):
    rois = config.rois_to_analyze

    # mds_file = "data/mds_dir/subj_01/1_mask20_mds_sample0.npy"
    # mds_file = os.path.join("data/mds_dir", f"subj_{subj:02d}", f"{subj}_mask20_mds_sample0.npy")

    # TODO
    # - load mask
    # for each ROI
    #   - get MDS space for corresponding ROI
    #   - get voxels for corresponding ROI
    #   - fit gaussian for each voxel of ROI
    #   - save fit

    columns = ["x0", "y0", "sigma", "slope", "intercept"]
    initial = params["initial"]
    bounds = params["bounds"]

    for i, subj in enumerate(subj_list):
        gaussian_fit_result_dir_path = os.path.join(
            config.gaussian_fit_results_dir, f"subj{subj:02d}"
        )
        os.makedirs(gaussian_fit_result_dir_path, exist_ok=True)

        mask = retrieve_roi_mask(config, subj)
        betas, _ = retrieve_stacked_betas(config, subj, "averaged", 0)
        betas_test = retrieve_stacked_betas_test(config, subj)

        if np.isnan(betas).any():
            raise ValueError("Found NaNs")

        logging.info(f"Starting fitting for subj{subj:02d}")

        for i, roi in enumerate(list(rois.keys())):
            logging.info(f"Fitting for ROI {roi}")
            roi_mask_value = rois[roi]

            gaussian_result_file_path = os.path.join(
                gaussian_fit_result_dir_path,
                f"fitted_voxels_mask_{roi_mask_value}.xlsx",
            )

            mds_file = os.path.join(
                "data/mds_dir",
                f"subj_{subj:02d}",
                f"mask_{roi_mask_value}_averaged_mds.npy",
            )
            mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

            mask_voxel_indices = filter_roi_mask(roi_mask_value, mask)
            mask_voxel_indices = mask_voxel_indices[0]

            if os.path.exists(gaussian_result_file_path):
                logging.info("Loading existing gaussian fit results")
                fits_roi = pd.read_excel(gaussian_result_file_path)

            else:
                fits_roi = pd.DataFrame(columns=columns)

                for voxel_i in tqdm(mask_voxel_indices):
                    voxel_activity = betas[:, voxel_i]

                    if params["random"]:
                        attempt = 1
                        solved = False
                        while not solved and attempt <= 10:
                            try:
                                initial_guess = (
                                    initial[1] - initial[0]
                                ) * np.random.random(initial[0].shape) + initial[0]
                                voxel_fit = curve_fit(
                                    gaussian_2d_curve,
                                    mds,
                                    voxel_activity,
                                    p0=initial_guess,
                                    bounds=bounds,
                                    method="trf",
                                    ftol=1e-06,
                                )[0]
                                solved = True

                            except RuntimeError:
                                if attempt >= 5:
                                    print(
                                        f"VOXEL {voxel_i}: optimal params not found after {attempt} attempts"
                                    )

                                attempt + 1

                    fits_roi.loc[voxel_i] = voxel_fit

                fits_roi.to_excel(gaussian_result_file_path)

            # if true - no computation needed
            # if "variance_explained" in fits_roi.columns:
            if False:
                logging.info("Variance explained already computed")

            else:
                logging.info("Computing variance explained")
                # Stabilizer for numerical calculations (avoids division by zero)
                EPSILON = 0  # 1e-8

                required_columns = [
                    "x0",
                    "y0",
                    "sigma",
                    "slope",
                    "intercept",
                ]  # Modify as needed

                # Extract only the relevant voxel indices from beta values
                masked_voxel_betas = betas.T[mask_voxel_indices].T
                masked_voxel_betas_test = betas_test.T[mask_voxel_indices].T

                def apply_gaussian_fit(fit_params):
                    """
                    Apply a 2D Gaussian model using the given fit parameters.

                    Parameters:
                    - fit_params: Array of fit parameters for the Gaussian model.

                    Returns:
                    - Predicted activity based on the Gaussian function.
                    """
                    return gaussian_2d_curve(mds, *fit_params)

                # Apply Gaussian function to each row in fits_roi
                predicted_activity = fits_roi[required_columns].apply(
                    apply_gaussian_fit, axis=1
                )

                # Convert the results to a NumPy array (ensure proper shape)
                predicted_activity = np.array(
                    [np.array(activity) for activity in predicted_activity]
                ).T

                # Compute Residual Sum of Squares (RSS) and Total Sum of Squares (TSS)
                residual_sum_squares = np.sum(
                    (predicted_activity - masked_voxel_betas) ** 2, axis=0
                )
                total_sum_squares = np.sum(
                    (masked_voxel_betas - masked_voxel_betas.mean(axis=0)) ** 2, axis=0
                )

                # Compute RSS and TSS for test data
                residual_sum_squares_test = np.sum(
                    (predicted_activity - masked_voxel_betas_test) ** 2, axis=0
                )
                total_sum_squares_test = np.sum(
                    (masked_voxel_betas_test - masked_voxel_betas_test.mean(axis=0))
                    ** 2,
                    axis=0,
                )

                # Compute variance explained (R² score)
                fits_roi["variance_explained"] = 1 - (
                    residual_sum_squares / (total_sum_squares + EPSILON)
                )

                # Compute variance explained (R² score)
                fits_roi["variance_explained_test"] = 1 - (
                    residual_sum_squares_test / (total_sum_squares_test + EPSILON)
                )

                fits_roi.to_excel(gaussian_result_file_path)


if __name__ == "__main__":
    config = load_config("config.yaml")
    fit_gaussian_params(config, [1])
