import os

import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as opt
import matplotlib.pyplot as plt
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


def find_positive_percentile(series):
    # Remove missing values and sort the series
    sorted_series = series.dropna().sort_values().reset_index(drop=True)

    # Check if there are any positive values
    positive_mask = sorted_series > 0
    if not positive_mask.any():
        return None  # No positive values in the column

    # Find the first index where the value is positive
    first_positive_idx = positive_mask.idxmax()
    n = len(sorted_series)

    # Handle edge case with only one data point
    if n == 1:
        return 100.0 if positive_mask.iloc[0] else 0.0

    # Calculate the percentile
    percentile = (first_positive_idx / (n - 1)) * 100
    return round(percentile, 2)  # Round for readability


class Gaussian2DFitter:
    def __init__(self, use_polar: bool = False):
        """
        Initialize the fitter.

        Parameters:
          use_polar: if True, the fitting is performed in polar coordinates;
                     if False, Cartesian coordinates are used.
        """
        self.use_polar = use_polar
        self.params = None  # Will hold [A, x0, y0, sigma]
        self.solved = False
        self.mds = (None, None)

    @staticmethod
    def gaussian_2d_cartesian(coords, A, x0, y0, sigma, intercept):
        """
        Isotropic 2D Gaussian in Cartesian coordinates with an intercept.

        Parameters:
        coords: tuple of (x, y) arrays
        A: amplitude
        x0, y0: center of the Gaussian
        sigma: standard deviation
        intercept: constant offset added to the Gaussian

        Returns:
        The evaluated Gaussian function with an intercept.
        """
        x, y = coords
        exponent = -(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        return A * np.exp(exponent) + intercept

    @staticmethod
    def gaussian_2d_polar(coords, A, r0, theta0, sigma, intercept):
        """
        Isotropic 2D Gaussian in polar coordinates with an intercept term.

        Parameters:
        coords: tuple of (r, theta) arrays
        A: amplitude
        r0, theta0: center in polar coordinates
        sigma: standard deviation
        intercept: constant offset added to the Gaussian

        The distance between a point (r, theta) and the center (r0, theta0)
        is computed via the law of cosines:

            d = sqrt(r**2 + r0**2 - 2*r*r0*cos(theta - theta0))
        """
        r, theta = coords
        d = np.sqrt(r**2 + r0**2 - 2 * r * r0 * np.cos(theta - theta0))
        exponent = -(d**2) / (2 * sigma**2)
        return A * np.exp(exponent) + intercept

    def fit(self, x_vals, y_vals, target_values):
        """
        Fit the isotropic 2D Gaussian to the provided data, running multiple fits
        and selecting the best one (based on variance explained). If consecutive fits
        yield almost the same result, the loop stops early to save computation.
        The model now includes an intercept term.

        Parameters:
        x_vals, y_vals: 1D arrays of sample coordinates (Cartesian)
        target_values: 1D array of values at (x_vals, y_vals)

        Returns:
        (popt, solved) where popt is the optimized parameters
            [A, x0, y0, sigma, intercept]  (for Cartesian) or
            [A, r0, theta0, sigma, intercept] (converted to Cartesian as [A, x0, y0, sigma, intercept])
        and solved is a boolean indicating whether the fit succeeded.
        """
        self.mds = (x_vals, y_vals)

        max_outer_attempts = 4  # Maximum number of independent fit attempts
        convergence_tol = 0.02  # Tolerance for considering two fits "almost the same"
        stable_threshold = (
            2  # Number of consecutive fits with nearly identical R² to stop early
        )

        best_r2 = -np.inf  # Initialize best variance explained
        best_params = None  # To store the best parameters found
        consecutive_stable = 0  # Counter for consecutive stable fits
        overall_solved = False  # Tracks if any fit succeeded

        for fit_iter in range(max_outer_attempts):
            # --- Prepare initial guess and bounds based on coordinate system ---
            A0 = np.max(target_values)
            if self.use_polar:
                # Convert the Cartesian samples to polar coordinates.
                r_samples = np.sqrt(x_vals**2 + y_vals**2)
                theta_samples = np.arctan2(y_vals, x_vals)
                sigma0 = max(np.std(r_samples), 1e-6)
                # Bounds: amplitude in [-10,10], r0 in [0,1], theta0 in [-pi,pi], sigma > 0, intercept in [-10,10].
                lower_bounds = [0, 0, -np.pi, 1e-6, -10]
                upper_bounds = [10, 1.0, np.pi, 30, 10]
            else:
                sigma0 = max(np.std(x_vals), np.std(y_vals), 1e-6)
                # Bounds for Cartesian fit: amplitude in [-10,10], x0,y0 within [-1, 1], sigma > 0, intercept in [-10,10].
                lower_bounds = [0, -1, -1, 1e-6, -10]
                upper_bounds = [10, 1, 1, 30, 10]

            # --- Inner loop: try up to 3 times to get a successful fit ---
            attempt = 1
            solved = False
            candidate_params = None
            while attempt <= 3 and not solved:
                try:
                    if self.use_polar:
                        # Initial guess for polar parameters: [A, r0, theta0, sigma, intercept]
                        initial_guess = [
                            np.random.uniform(0.01, A0),
                            np.random.uniform(0, 1),
                            np.random.uniform(-np.pi, np.pi),
                            sigma0,
                            np.random.uniform(-1, 1),
                        ]
                        popt, pcov = curve_fit(
                            Gaussian2DFitter.gaussian_2d_polar,
                            (r_samples, theta_samples),
                            target_values,
                            p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds),
                            maxfev=2000,
                        )
                        # Convert the polar center to Cartesian coordinates.
                        A, r0, theta0, sigma, intercept = popt
                        x0 = r0 * np.cos(theta0)
                        y0 = r0 * np.sin(theta0)
                        candidate_params = [
                            A.item(),
                            x0.item(),
                            y0.item(),
                            sigma.item(),
                            intercept.item(),
                        ]
                    else:
                        # Initial guess for Cartesian parameters: [A, x0, y0, sigma, intercept]
                        initial_guess = [
                            np.random.uniform(0.01, A0),
                            np.random.uniform(-1, 1),
                            np.random.uniform(-1, 1),
                            sigma0,
                            np.random.uniform(-1, 1),
                        ]
                        popt, pcov = curve_fit(
                            Gaussian2DFitter.gaussian_2d_cartesian,
                            (x_vals, y_vals),
                            target_values,
                            p0=initial_guess,
                            bounds=(lower_bounds, upper_bounds),
                            maxfev=2000,
                        )
                        candidate_params = popt
                    solved = True
                except Exception as e:
                    attempt += 1
                    if attempt > 3:
                        print(
                            f"Fitting failed after 3 attempts in outer iteration {fit_iter+1}: {e}"
                        )

            # In case all three attempts fail, use a fallback set of parameters.
            if not solved:
                candidate_params = [np.mean(target_values).item(), 0, 0, 100, 0]

            # Temporarily set self.params to candidate_params so we can compute R².
            saved_params = self.params
            self.params = candidate_params
            candidate_r2 = self.variance_explained(
                (x_vals, y_vals), target_values, input_polar=False
            )
            # Restore self.params (we'll update it later if this candidate is the best)
            self.params = saved_params

            # --- Check if this candidate is the best so far ---
            if candidate_r2 > best_r2:
                best_r2 = candidate_r2
                best_params = candidate_params
                overall_solved = solved
                consecutive_stable = 0
            else:
                # If the new candidate's performance is nearly identical, increment counter.
                if abs(candidate_r2 - best_r2) < convergence_tol:
                    consecutive_stable += 1
                else:
                    consecutive_stable = 0

            # If we have a couple of consecutive stable fits, stop early.
            if consecutive_stable >= stable_threshold:
                break

        # Store the best found parameters.
        self.params = best_params
        self.solved = overall_solved
        return self.params, self.solved

    def predict(self, coords, input_polar: bool = False):
        """
        Predict the Gaussian value at the provided coordinates.

        Parameters:
          coords: tuple of coordinate arrays.
                  If input_polar is False, coords should be (x, y).
                  If input_polar is True, coords should be (r, theta).
          input_polar: If True, interpret the input coordinates as polar.

        Returns:
          The predicted Gaussian value at the provided points.

        Note: Even if the fitting was performed in polar coordinates, the stored
        parameters are in Cartesian form. Thus, predictions are made via the
        Cartesian Gaussian function.
        """
        if self.params is None:
            raise ValueError("Model has not been successfully fitted yet.")

        A, x0, y0, sigma, intercept = self.params

        # Convert input coordinates to Cartesian if they are provided in polar form.
        if input_polar:
            r, theta = coords
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coords_cartesian = (x, y)
        else:
            coords_cartesian = coords

        return Gaussian2DFitter.gaussian_2d_cartesian(
            coords_cartesian, A, x0, y0, sigma, intercept
        )

    def plot_fit_cartesian(self, x_samples, y_samples, target_values):
        """
        Visualize the fitted Gaussian in Cartesian space using both a 3D surface and a 2D contour plot.

        Parameters:
          x_samples, y_samples: 1D arrays of the sample coordinates (Cartesian)
          target_values: 1D array of the measured values at (x_samples, y_samples)
        """
        if not self.solved or self.params is None:
            raise ValueError("Model has not been successfully fitted yet.")

        popt = self.params  # [A, x0, y0, sigma, sigma]

        # Create a meshgrid covering the range of your data
        x_min, x_max = np.min(x_samples), np.max(x_samples)
        y_min, y_max = np.min(y_samples), np.max(y_samples)
        X_grid, Y_grid = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
        )

        # Evaluate the fitted Gaussian on the grid using the Cartesian function.
        Z_grid = Gaussian2DFitter.gaussian_2d_cartesian((X_grid, Y_grid), *popt)

        fig = plt.figure(figsize=(12, 5))

        # --- 3D Surface Plot ---
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(x_samples, y_samples, target_values, color="r", label="Data Points")
        ax1.plot_surface(X_grid, Y_grid, Z_grid, cmap="viridis", alpha=0.6)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Value")
        ax1.set_title("Fitted Isotropic 2D Gaussian (3D view)")

        # --- 2D Contour/Heatmap ---
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X_grid, Y_grid, Z_grid, levels=50, cmap="viridis")
        ax2.scatter(
            x_samples, y_samples, c=target_values, cmap="coolwarm", edgecolors="k"
        )
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("Fitted Isotropic 2D Gaussian (2D view)")
        plt.colorbar(contour, ax=ax2, label="Gaussian Value")

        plt.tight_layout()
        plt.show()

    def variance_explained(self, coords, target_values, input_polar: bool = False):
        """
        Compute the variance explained (R² score) of the fitted Gaussian model.

        Parameters:
          coords: tuple of coordinate arrays.
                  If input_polar is False, coords should be (x, y);
                  if True, (r, theta) are expected.
          target_values: 1D array of measured values at the given coordinates.
          input_polar: Boolean flag indicating whether the provided coordinates are in polar form.

        Returns:
          r_squared: A float representing the proportion of variance explained by the model.
        """
        predictions = self.predict(coords, input_polar=input_polar)

        variance = self.compute_variance(predictions, target_values)

        return variance

    @staticmethod
    def compute_variance(predictions, target_values):
        ss_res = np.sum((target_values - predictions) ** 2)
        ss_tot = np.sum((target_values - np.mean(target_values)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        return r_squared

    @staticmethod
    def rescale(train, test):
        """
        Input
        -------
        train: 1D array of length n, to rescale
        test : 1D array of length n, target to rescale to

        Output
        --------
        new_train: 1D array of length n, rescale train array

        Rescale the train array to the test array. Linear rescale using pseudoinverse
        """
        if len(train.shape) == 1:
            train = np.array([train])  # add a dimension for concat
        # print(train.shape)
        train_ones = np.concatenate(
            (train, np.ones((train.shape[0], train.shape[1])))
        ).T
        scale = np.linalg.pinv(train_ones) @ test.T

        new_train = train_ones @ scale  # make
        return new_train.T.squeeze(), scale

    def compute_noise_ceiling(
        self, betas_train_voxel, betas_test_voxel, rescaled: bool = True
    ):

        if rescaled:
            rescaled_train_betas, _ = self.rescale(betas_train_voxel, betas_test_voxel)
        else:
            rescaled_train_betas = betas_train_voxel
        # rescaled_train_betas = betas_train_voxel

        noise_ceiling = self.compute_variance(rescaled_train_betas, betas_test_voxel)

        return noise_ceiling

    def variance_explained_noise_ceiling(
        self, coords, betas_test_voxel, noise_ceiling: float
    ):
        # results = []
        # noise_ceilling_all_vox = np.zeros(betas_train.shape[1])
        # predictions = self.predict(coords, input_polar=input_polar)

        # print(betas_train_voxel)
        # print(betas_test_voxel)

        variance = self.variance_explained(coords, betas_test_voxel)

        noise_ceiling_variance = variance / noise_ceiling

        return noise_ceiling_variance
        # # loop through each voxel (and n == n_dim_images)
        # for row_i in tqdm(range(betas_train.shape[1])):

        #     new_row, _ = self.rescale(current_row, betas_test[:, row_i])

        #     rss = np.sum((new_row - betas_test[:, row_i]) ** 2)

        #     variance = np.sum((betas_test[:, row_i] - np.mean(betas_test[:, row_i])) ** 2)
        #     ve_voxels = 1 - rss / variance  # noise ceilling of all voxels in current roi

        # noise_ceilling_all_vox[row_i] = ve_voxels

    def linearly_rescale_gaussian(self, voxel_activity_test):
        """
        Linearly rescales the Gaussian fit to match the amplitude and offset of the test data.

        Parameters:
        voxel_activity_test (np.ndarray): The test voxel activity data to determine the scaling factors.

        Returns:
        list: Rescaled Gaussian parameters [slope, x0, y0, sigma, intercept].
        """
        epsilon = 0.001

        # Extract parameters from the fitted Gaussian model
        slope, x0, y0, sigma, intercept = self.params  # Unpacking parameters
        # Initial guess for Cartesian parameters: [A, x0, y0, sigma, intercept]
        lower_bounds = [0, x0 - epsilon, y0 - epsilon, sigma - epsilon, -np.inf]
        upper_bounds = [np.inf, x0 + epsilon, y0 + epsilon, sigma + epsilon, np.inf]

        # print(f"{self.mds=}")
        # print(f"{self.params=}")

        popt, solved = curve_fit(
            self.gaussian_2d_cartesian,
            self.mds,
            voxel_activity_test,
            p0=self.params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=2000,
        )

        rescaled_slope, x0, y0, sigma, rescaled_intercept = popt

        # Compute scaling factors based on mean voxel activity
        # scale_slope = np.mean(voxel_activity_test) / slope  # Scale factor for amplitude
        # delta_intercept = (
        #     np.mean(voxel_activity_test) - intercept
        # )  # Difference in means for offset

        # # Apply scaling
        # rescaled_slope = slope * scale_slope
        # rescaled_intercept = intercept + delta_intercept

        # Return the updated parameters with unchanged x0, y0, and sigma
        return [rescaled_slope, x0, y0, sigma, rescaled_intercept]


# def fit_gaussian_params(config: Configuration, subj_list: list[int], set_to_take: str):
#     rois = config.rois_to_analyze

#     # mds_file = "data/mds_dir/subj_01/1_mask20_mds_sample0.npy"
#     # mds_file = os.path.join("data/mds_dir", f"subj_{subj:02d}", f"{subj}_mask20_mds_sample0.npy")

#     # TODO
#     # - load mask
#     # for each ROI
#     #   - get MDS space for corresponding ROI
#     #   - get voxels for corresponding ROI
#     #   - fit gaussian for each voxel of ROI
#     #   - save fit

#     columns_polar = ["ecc", "pol", "sigma", "slope", "intercept"]
#     columns = [
#         "x0",
#         "y0",
#         "sigma",
#         "slope",
#         "intercept",
#         "solved",
#         "var_train",
#         "var_test",
#         "noise_ceiling",
#         "model_performance",
#     ]
#     initial = params["initial"]
#     bounds = params["bounds"]

#     for i, subj in enumerate(subj_list):
#         gaussian_fit_result_dir_path = os.path.join(
#             config.gaussian_fit_results_dir, set_to_take, f"subj{subj:02d}"
#         )
#         os.makedirs(gaussian_fit_result_dir_path, exist_ok=True)

#         mask = retrieve_roi_mask(config, subj, set_to_take)
#         betas, _ = retrieve_stacked_betas(
#             config, subj, "averaged", 0, subj_to_check=set_to_take
#         )
#         betas_test = retrieve_stacked_betas_test(
#             config, subj, subj_to_check=set_to_take
#         )

#         if np.isnan(betas).any():
#             raise ValueError("Found NaNs")

#         logging.info(f"Starting fitting for subj{subj:02d}")

#         for i, roi in enumerate(list(rois.keys())):
#             logging.info(f"Fitting for ROI {roi}")
#             roi_mask_value = rois[roi]

#             gaussian_result_file_path = os.path.join(
#                 gaussian_fit_result_dir_path,
#                 f"fitted_voxels_mask_{roi_mask_value}.xlsx",
#             )

#             mds_file = os.path.join(
#                 "data/mds_dir",
#                 set_to_take,
#                 f"subj_{subj:02d}",
#                 f"mask_{roi_mask_value}_averaged_mds.npy",
#             )
#             mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

#             mask_voxel_indices = filter_roi_mask(roi_mask_value, mask)
#             mask_voxel_indices = mask_voxel_indices[0]

#             # masked_voxel_betas_test = betas_test.T[mask_voxel_indices].T

#             if os.path.exists(gaussian_result_file_path):
#                 logging.info(
#                     f"Loading existing gaussian fit results {gaussian_result_file_path}"
#                 )
#                 fits_roi = pd.read_excel(gaussian_result_file_path)

#             else:
#                 fits_roi = pd.DataFrame(columns=columns)

#                 for voxel_i in tqdm(mask_voxel_indices):
#                     voxel_activity = betas[:, voxel_i]
#                     voxel_activity_test = betas_test[:, voxel_i]

#                     fitter = Gaussian2DFitter(use_polar=True)

#                     x_samples, y_samples = (mds[0, :], mds[1, :])
#                     popt, solved = fitter.fit(x_samples, y_samples, voxel_activity)

#                     noise_ceiling = fitter.compute_noise_ceiling(
#                         voxel_activity, voxel_activity_test
#                     )

#                     var_train = fitter.variance_explained(
#                         (x_samples, y_samples), voxel_activity
#                     )
#                     var_test = fitter.variance_explained(
#                         (x_samples, y_samples), voxel_activity_test
#                     )

#                     model_performance_ceiling = fitter.variance_explained_noise_ceiling(
#                         (x_samples, y_samples), voxel_activity_test, noise_ceiling
#                     )

#                     # print(f"{var_test=}")

#                     # print(f"{var_noise_ceiling=}")
#                     # quit()
#                     A, x0, y0, sigma, intercept = popt

#                     slope = A  # Directly assigning A as amplitude
#                     voxel_fit = [
#                         x0,
#                         y0,
#                         sigma,
#                         slope,
#                         intercept,
#                         solved,
#                         var_train,
#                         var_test,
#                         noise_ceiling,
#                         model_performance_ceiling,
#                     ]

#                     fits_roi.loc[voxel_i] = voxel_fit

#                 # Add the current index as a new column
#                 fits_roi["original_index"] = fits_roi.index

#                 # Reset the index to 0, 1, 2, ...
#                 fits_roi = fits_roi.reset_index(drop=True)

#                 fits_roi.to_excel(gaussian_result_file_path, index=False)

#             # if true - no computation needed
#             # if "variance_explained" in fits_roi.columns:
#             if False:
#                 logging.info("Variance explained already computed")

#             else:
#                 # print(fits_roi.head())
#                 logging.info("Computing variance stats")
#                 # Stabilizer for numerical calculations (avoids division by zero)
#                 EPSILON = 0  # 1e-8

#                 required_columns = [
#                     "x0",
#                     "y0",
#                     "sigma",
#                     "slope",
#                     "intercept",
#                 ]  # Modify as needed

#                 # Extract only the relevant voxel indices from beta values
#                 # (n_samples, n_voxels)
#                 masked_voxel_betas = betas.T[mask_voxel_indices].T
#                 masked_voxel_betas_test = betas_test.T[mask_voxel_indices].T

#                 var_train_stat = fits_roi["var_train"].describe()
#                 # Calculate the positive percentile
#                 positive_percentile = find_positive_percentile(fits_roi["var_train"])

#                 # Add the custom statistic to the results
#                 if positive_percentile is not None:
#                     var_train_stat["positive_percentile"] = positive_percentile
#                 else:
#                     var_train_stat["positive_percentile"] = "No positive values"

#                 print(f"Stats for train set:\n{var_train_stat}")

#                 var_test_stat = fits_roi["var_test"].describe()
#                 # Calculate the positive percentile
#                 positive_percentile = find_positive_percentile(fits_roi["var_test"])

#                 # Add the custom statistic to the results
#                 if positive_percentile is not None:
#                     var_test_stat["positive_percentile"] = positive_percentile
#                 else:
#                     var_test_stat["positive_percentile"] = "No positive values"

#                 print(f"Stats for test set:\n{var_test_stat}")

#                 fits_roi.to_excel(gaussian_result_file_path, index=False)

#                 print("-" * 30)


def fit_gaussian_params(config: Configuration, subj_list: list[int], set_to_take: str):
    rois = config.analysis.rois_to_analyze

    columns = [
        "x0",
        "y0",
        "sigma",
        "slope",
        "intercept",
        "solved",
        "var_train",
        "var_test",
        "var_test_rescaled",
        "noise_ceiling",
        "model_performance",
        "noise_ceiling_non_rescaled",
        "model_performance_non_rescaled",
        "rescaled_slope",
        "rescaled_intercept",
    ]

    for i, subj in enumerate(subj_list):
        gaussian_fit_result_dir_path = os.path.join(
            config.directories.gaussian_fit_results_dir, set_to_take, f"subj{subj:02d}"
        )
        os.makedirs(gaussian_fit_result_dir_path, exist_ok=True)

        mask = retrieve_roi_mask(config, subj, set_to_take, False)
        betas, _, mds_mapping = retrieve_stacked_betas(
            config, subj, "averaged", 0, subj_to_check=set_to_take
        )
        betas_test = retrieve_stacked_betas_test(
            config, subj, subj_to_check=set_to_take
        )

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
                set_to_take,
                f"subj_{subj:02d}",
                f"mask_{roi_mask_value}_averaged_mds.npy",
            )
            mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

            mask_voxel_indices = filter_roi_mask(roi_mask_value, mask)
            mask_voxel_indices = mask_voxel_indices[0]

            if os.path.exists(gaussian_result_file_path):
                logging.info(
                    f"Loading existing gaussian fit results {gaussian_result_file_path}"
                )
                fits_roi = pd.read_excel(gaussian_result_file_path)
            else:
                fits_roi = pd.DataFrame(columns=columns)
                for voxel_i in tqdm(mask_voxel_indices):
                    voxel_activity = betas[:, voxel_i]
                    voxel_activity_test = betas_test[:, voxel_i]

                    fitter = Gaussian2DFitter(use_polar=True)
                    x_samples, y_samples = (mds[0, :], mds[1, :])

                    popt, solved = fitter.fit(x_samples, y_samples, voxel_activity)

                    A, x0, y0, sigma, intercept = popt
                    slope = A

                    voxel_fit = [
                        x0,
                        y0,
                        sigma,
                        slope,
                        intercept,
                        solved,
                        np.nan,  # var_train
                        np.nan,  # var_test
                        np.nan,  # var_test_rescaled
                        np.nan,  # noise_ceiling
                        np.nan,  # model_performance
                        np.nan,  # noise_ceiling_non_rescaled
                        np.nan,  # model_performance_non_rescaled
                        np.nan,  # rescaled_slope
                        np.nan,  # rescaled_intercept
                    ]
                    fits_roi.loc[voxel_i] = voxel_fit

                fits_roi["original_index"] = fits_roi.index
                fits_roi = fits_roi.reset_index(drop=True)
                fits_roi.to_excel(gaussian_result_file_path, index=False)

            # Recalculate metrics for all voxels
            logging.info("Recalculating metrics for all voxels...")
            x_samples, y_samples = mds[0, :], mds[1, :]
            for index, row in tqdm(
                fits_roi.iterrows(), total=len(fits_roi), desc="Processing voxels"
            ):
                voxel_i = row["original_index"]
                fitter = Gaussian2DFitter(use_polar=False)
                fitter.mds = (x_samples, y_samples)

                fitter.params = [
                    row["slope"],
                    row["x0"],
                    row["y0"],
                    row["sigma"],
                    row["intercept"],
                ]
                fitter.solved = row["solved"]

                voxel_train = betas[:, voxel_i]
                voxel_test = betas_test[:, voxel_i]

                var_train = fitter.variance_explained(
                    (x_samples, y_samples), voxel_train
                )

                # predictions = fitter.predict((x_samples, y_samples))
                # rescaled_predictions, _ = fitter.rescale(predictions, voxel_test)
                # var_test = fitter.compute_variance(rescaled_predictions, voxel_test)

                noise_ceiling = fitter.compute_noise_ceiling(
                    voxel_train, voxel_test, rescaled=True
                )
                noise_ceiling_unscaled = fitter.compute_noise_ceiling(
                    voxel_train, voxel_test, rescaled=False
                )

                var_test = fitter.variance_explained((x_samples, y_samples), voxel_test)

                rescaled_slope, x0, y0, sigma, rescaled_intercept = (
                    fitter.linearly_rescale_gaussian(voxel_test)
                )
                fitter.params = [
                    rescaled_slope,
                    row["x0"],
                    row["y0"],
                    row["sigma"],
                    rescaled_intercept,
                ]
                var_test_rescaled = fitter.variance_explained(
                    (x_samples, y_samples), voxel_test
                )

                model_perf = (
                    var_test_rescaled / noise_ceiling if noise_ceiling != 0 else 0.0
                )

                model_performance_non_rescaled = var_test / noise_ceiling_unscaled

                fits_roi.at[index, "var_train"] = var_train
                fits_roi.at[index, "var_test"] = var_test
                fits_roi.at[index, "var_test_rescaled"] = var_test_rescaled

                fits_roi.at[index, "noise_ceiling"] = noise_ceiling
                fits_roi.at[index, "model_performance"] = model_perf

                fits_roi.at[index, "noise_ceiling_non_rescaled"] = (
                    noise_ceiling_unscaled
                )
                fits_roi.at[index, "model_performance_non_rescaled"] = (
                    model_performance_non_rescaled
                )

                fits_roi.at[index, "rescaled_slope"] = rescaled_slope
                fits_roi.at[index, "rescaled_intercept"] = rescaled_intercept

            # Compute and log stats
            logging.info("Computing variance stats...")
            var_train_stat = fits_roi["var_train"].describe()
            pos_percentile = find_positive_percentile(fits_roi["var_train"])
            var_train_stat["positive_percentile"] = pos_percentile or "No positives"
            logging.info(f"Train stats:\n{var_train_stat}")

            var_test_stat = fits_roi["var_test"].describe()
            pos_percentile = find_positive_percentile(fits_roi["var_test"])
            var_test_stat["positive_percentile"] = pos_percentile or "No positives"
            logging.info(f"Test stats:\n{var_test_stat}")

            fits_roi.to_excel(gaussian_result_file_path, index=False)
            logging.info(f"Updated results saved to {gaussian_result_file_path}")


if __name__ == "__main__":
    config = load_config("config.yaml")
    fit_gaussian_params(config, [1], "subj_01")
