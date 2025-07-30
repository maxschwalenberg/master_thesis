"""
Gaussian Fitting Analysis Module

This module implements 2D Gaussian fitting for neural voxel responses in MDS space.
It includes functionality for fitting isotropic Gaussians in both Cartesian and polar
coordinates, with support for positive/negative slopes and comprehensive performance metrics.
"""

# Standard library imports
import os
import pickle
import logging
from functools import partial

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.optimize as opt
from tqdm import tqdm

# Local imports
from utils.config import load_config, Configuration
from utils.utils import (
    retrieve_stacked_betas,
    retrieve_roi_mask,
    retrieve_roi_mask_extended,
    filter_roi_mask,
)
from rsa.create_rdm import rdm_mds_from_betas


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)




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
    def __init__(
        self,
        use_polar: bool = False,
        use_negative_slope: bool = False,
        use_mixed_slope: bool = False,
    ):
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
        self.use_negative_slope = use_negative_slope
        self.use_mixed_slope = use_mixed_slope

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

    def fit(self, x_vals, y_vals, target_values_train, target_values_test):
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
        convergence_tol = 0.04  # Tolerance for considering two fits "almost the same"
        stable_threshold = (
            2  # Number of consecutive fits with nearly identical R² to stop early
        )

        best_r2 = -np.inf  # Initialize best variance explained
        best_params = None  # To store the best parameters found
        consecutive_stable = 0  # Counter for consecutive stable fits
        overall_solved = False  # Tracks if any fit succeeded

        for fit_iter in range(max_outer_attempts):
            # --- Prepare initial guess and bounds based on coordinate system ---
            A0 = np.max(target_values_train)
            if self.use_polar:
                # Convert the Cartesian samples to polar coordinates.
                r_samples = np.sqrt(x_vals**2 + y_vals**2)
                theta_samples = np.arctan2(y_vals, x_vals)
                # sigma0 = max(np.std(r_samples), 1e-6)

                # Bounds: amplitude in [-10,10], r0 in [0,1], theta0 in [-pi,pi], sigma > 0, intercept in [-10,10].
                # A, r0, theta0, sigma, intercept

                if self.use_mixed_slope:
                    lower_bounds = [-np.inf, 0, -np.pi, 0.01, -np.inf]
                    upper_bounds = [np.inf, 1.0, np.pi, 20, np.inf]
                else:
                    if self.use_negative_slope:
                        lower_bounds = [-np.inf, 0, -np.pi, 0.01, -np.inf]
                        upper_bounds = [-0.1, 1.0, np.pi, 20, np.inf]
                    else:
                        lower_bounds = [0.1, 0, -np.pi, 0.01, -np.inf]
                        upper_bounds = [np.inf, 1.0, np.pi, 20, np.inf]

            else:
                raise (f"Only want to fit in polar space!!!")
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
                        if self.use_mixed_slope:
                            initial_guess = [
                                np.random.uniform(-3, 3),
                                np.random.uniform(0, 0.5),
                                np.random.uniform(-np.pi, np.pi),
                                np.random.uniform(0.01, 2),
                                np.random.uniform(-2, 2),
                            ]
                        else:
                            # Initial guess for polar parameters: [A, r0, theta0, sigma, intercept]
                            if self.use_negative_slope:
                                initial_guess = [
                                    np.random.uniform(-3, -0.1),
                                    np.random.uniform(0, 0.5),
                                    np.random.uniform(-np.pi, np.pi),
                                    np.random.uniform(0.01, 2),
                                    np.random.uniform(-2, 2),
                                ]

                            else:
                                initial_guess = [
                                    np.random.uniform(0.1, 3),
                                    np.random.uniform(0, 0.5),
                                    np.random.uniform(-np.pi, np.pi),
                                    np.random.uniform(0.01, 2),
                                    np.random.uniform(-2, 2),
                                ]
                        popt, pcov = curve_fit(
                            Gaussian2DFitter.gaussian_2d_polar,
                            (r_samples, theta_samples),
                            target_values_train,
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
                        raise (f"Only want to fit in polar space!!!")

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
                pcov = np.zeros((5,5))
                if self.use_mixed_slope:
                    candidate_params = [np.mean(target_values_train).item(), 0, 0, 100, 0]
                else:
                    if self.use_negative_slope:
                        candidate_params = [
                            min(0, np.mean(target_values_train).item()),
                            0,
                            0,
                            100,
                            0,
                        ]
                    else:
                        candidate_params = [
                            max(0, np.mean(target_values_train).item()),
                            0,
                            0,
                            100,
                            0,
                        ]

            # Temporarily set self.params to candidate_params so we can compute R².
            saved_params = self.params
            self.params = candidate_params
            candidate_r2 = self.variance_explained(
                (x_vals, y_vals), target_values_test, input_polar=False
            )
            # Restore self.params (we'll update it later if this candidate is the best)
            self.params = saved_params

            # --- Check if this candidate is the best so far ---
            if candidate_r2 > best_r2:
                best_r2 = candidate_r2
                best_params = candidate_params
                best_pcov = pcov

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
        self.pcov = best_pcov
        self.solved = overall_solved
        return self.params, self.solved, self.pcov

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
        variance = self.variance_explained(coords, betas_test_voxel)

        noise_ceiling_variance = variance / noise_ceiling

        return noise_ceiling_variance


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

        if self.use_mixed_slope:
            lower_bounds = [
                -np.inf,
                x0 - epsilon,
                y0 - epsilon,
                sigma - epsilon,
                -np.inf,
            ]
            upper_bounds = [np.inf, x0 + epsilon, y0 + epsilon, sigma + epsilon, np.inf]
        else:
            if self.use_negative_slope:
                lower_bounds = [
                    -np.inf,
                    x0 - epsilon,
                    y0 - epsilon,
                    sigma - epsilon,
                    -np.inf,
                ]
                upper_bounds = [0, x0 + epsilon, y0 + epsilon, sigma + epsilon, np.inf]
            else:
                lower_bounds = [0, x0 - epsilon, y0 - epsilon, sigma - epsilon, -np.inf]
                upper_bounds = [
                    np.inf,
                    x0 + epsilon,
                    y0 + epsilon,
                    sigma + epsilon,
                    np.inf,
                ]

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

        

        # Return the updated parameters with unchanged x0, y0, and sigma
        return [rescaled_slope, x0, y0, sigma, rescaled_intercept]

    def gaussian_fixed_params(self, mds, intercept, slope, x0, y0, sigma):
        # we’ll pass slope,x0,y0,sigma via partial()
        return self.gaussian_2d_cartesian(mds, slope, x0, y0, sigma, intercept)

    def offset_rescale_with_fit(self, voxel_activity_test):
        slope, x0, y0, sigma, intercept = self.params

        # freeze all but intercept
        fit_func = partial(
            self.gaussian_fixed_params, slope=slope, x0=x0, y0=y0, sigma=sigma
        )

        popt, _ = curve_fit(
            fit_func,
            self.mds,
            voxel_activity_test,
            p0=[intercept],  # only intercept is free
            bounds=([-np.inf], [np.inf]),
            maxfev=1000,
        )

        (new_intercept,) = popt
        return [slope, x0, y0, sigma, new_intercept]


def fit_gaussian_params(
    config: Configuration,
    subj_list: list[int],
    set_to_take: str,
    use_negative_slope: bool = False,
    use_mixed_slope: bool = False,
    t_test_threshold: float = 3.0
):

    rois = config.analysis.rois_to_analyze

    columns = [
        "mds_hemi",
        "x0",
        "y0",
        "sigma",
        "slope",
        "intercept",
        "solved",
        "var_train",
        "var_test",
        "var_test_rescaled",
        "var_test_rescaled_offset",
        "noise_ceiling",
        "model_performance",
        "noise_ceiling_non_rescaled",
        "model_performance_non_rescaled",
        "rescaled_slope",
        "rescaled_intercept",
        "original_index"
    ]

    randomization = True
    randomize_offset = 0

    augment_shared_set = True
    logging.info(
        f"Augmenting with shared set: {augment_shared_set}\n{augment_shared_set=}\nROIs: {rois}"
    )

    for i, subj in enumerate(subj_list):
        gaussian_fit_result_dir_path = os.path.join(
            config.directories.gaussian_fit_results_dir, set_to_take, f"subj{subj:02d}"
        )
        os.makedirs(gaussian_fit_result_dir_path, exist_ok=True)

        # retrieve_roi_mask_extended
        mask, lh_size = retrieve_roi_mask_extended(
            config, subj, set_to_take, True, t_test_threshold=t_test_threshold
        )
        # mask = retrieve_roi_mask(config, subj, set_to_take, False)
        betas, _, mds_mapping = retrieve_stacked_betas(
            config,
            subj,
            "averaged",
            0,
            subj_to_check=set_to_take,
            augment_shared_set=augment_shared_set,
            randomization=randomization,
            seed_offset=randomize_offset,
        )

        betas_test, _, _ = retrieve_stacked_betas(
            config,
            subj,
            "averaged",
            0,
            subj_to_check=set_to_take,
            test=True,
            augment_shared_set=augment_shared_set,
            randomization=randomization,
            seed_offset=randomize_offset,
        )
        

        if np.isnan(betas).any():
            raise ValueError("Found NaNs")

        logging.info(f"Starting fitting for subj{subj:02d}")
        hemis = ["both", "lh", "rh"]

        for i, roi in enumerate(list(rois.keys())):
            if rois[roi] == 6 or rois[roi] == 9:
                use_negative_slope = True
            else:
                use_negative_slope = False

            logging.info(f"Fitting for ROI {roi} --> {use_negative_slope=}")
            roi_mask_value = rois[roi]

            rdms, mdss = rdm_mds_from_betas(
                config, betas, mask, lh_size, hemis, subj, roi_mask_value
            )

            gaussian_result_file_path = os.path.join(
                gaussian_fit_result_dir_path,
                f"fitted_voxels_mask_{roi_mask_value}.xlsx",
            )

            pcov_results_file = os.path.join(
                gaussian_fit_result_dir_path,
                f"pcov_{roi_mask_value}.pkl"
            )

            pcov_results = []

            mds_paths = []

            for hemi in hemis:
                if randomization:
                    mds_file = os.path.join(
                        config.directories.mds_dir,
                        set_to_take,
                        f"subj_{subj:02d}",
                        f"mask_{roi_mask_value}_averaged_{hemi}_rand_{randomize_offset}_mds.npy",
                    )
                else:
                    mds_file = os.path.join(
                        config.directories.mds_dir,
                        set_to_take,
                        f"subj_{subj:02d}",
                        f"mask_{roi_mask_value}_averaged_{hemi}_mds.npy",
                    )

                mds_paths.append(mds_file)

            

            mask_voxel_indices = filter_roi_mask(roi_mask_value, mask)
            mask_voxel_indices = mask_voxel_indices[0]

            if os.path.exists(gaussian_result_file_path):
                logging.info(
                    f"Loading existing gaussian fit results {gaussian_result_file_path}"
                )
                fits_roi = pd.read_excel(gaussian_result_file_path)
            else:
                fits_roi = pd.DataFrame(columns=columns)
                # check voxel index for hemisphere belonging

                for iteration, voxel_i in enumerate(tqdm(mask_voxel_indices)):
                    if voxel_i < lh_size:
                        hemis_to_check = ["both", "lh"]
                    else:
                        hemis_to_check = ["both", "rh"]

                    for hemi in hemis_to_check:
                        # hemi_index = hemis.index(hemi)

                        # need to transpose
                        mds = mdss[hemi].T

                        x_samples, y_samples = (mds[0, :], mds[1, :])

                        voxel_activity = betas[:, voxel_i]
                        voxel_activity_test = betas_test[:, voxel_i]

                        fitter = Gaussian2DFitter(
                            use_polar=True,
                            use_negative_slope=use_negative_slope,
                            use_mixed_slope=use_mixed_slope,
                        )

                        popt, solved, pcov = fitter.fit(x_samples, y_samples, voxel_activity, voxel_activity_test)

                        A, x0, y0, sigma, intercept = popt
                        slope = A

                        voxel_fit = [
                            hemi,
                            x0,
                            y0,
                            sigma,
                            slope,
                            intercept,
                            solved,
                            np.nan,  # var_train
                            np.nan,  # var_test
                            np.nan,  # var_test_rescaled
                            np.nan,  # var_test_rescaled_offset
                            np.nan,  # noise_ceiling
                            np.nan,  # model_performance
                            np.nan,  # noise_ceiling_non_rescaled
                            np.nan,  # model_performance_non_rescaled
                            np.nan,  # rescaled_slope
                            np.nan,  # rescaled_intercept
                            voxel_i
                        ]
                        fits_roi.loc[len(fits_roi)] = voxel_fit
                        pcov_results.append(
                            {
                                "hemi":hemi,
                                "original_index" : voxel_i,
                                "pcov" : pcov
                            }
                        )
                        # fits_roi.loc[voxel_i] = voxel_fit


                    if iteration % 200 == 0:
                        fits_roi.to_excel(gaussian_result_file_path, index=False)

                        # Save to file
                        with open(pcov_results_file, "wb") as f:
                            pickle.dump(pcov_results, f)


                
                fits_roi.to_excel(gaussian_result_file_path, index=False)

                # Save to file
                with open(pcov_results_file, "wb") as f:
                    pickle.dump(pcov_results, f)

            # make sure all columns are actually present
            for col in columns:
                # 1. Initialize the new column
                if col not in fits_roi.columns:
                    fits_roi[col] = np.nan

           
            # Recalculate metrics for all voxels
            logging.info("Recalculating metrics for all voxels...")
            for index, row in tqdm(
                fits_roi.iterrows(), total=len(fits_roi), desc="Processing voxels"
            ):

                hemi = row["mds_hemi"]
                # hemi_index = hemis.index(hemi)

                # need to transpose
                mds = mdss[hemi]

                mds = mds.T

                x_samples, y_samples = mds[0, :], mds[1, :]

                voxel_i = row["original_index"]
                fitter = Gaussian2DFitter(
                    use_polar=False,
                    use_negative_slope=use_negative_slope,
                    use_mixed_slope=use_mixed_slope,
                )
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
                # this computes var_test_rescaled bc the params were adjusted to the test set in the above step
                var_test_rescaled = fitter.variance_explained(
                    (x_samples, y_samples), voxel_test
                )

                # re-init to actual fit so that these params are used for offset rescaling
                fitter.params = [
                    row["slope"],
                    row["x0"],
                    row["y0"],
                    row["sigma"],
                    row["intercept"],
                ]
                same_slope, same_x0, same_y0, same_sigma, new_intercept = (
                    fitter.offset_rescale_with_fit(voxel_test)
                )
                assert same_slope == row["slope"] and same_x0 == row["x0"]  # ....
                fitter.params = [
                    row["slope"],
                    row["x0"],
                    row["y0"],
                    row["sigma"],
                    new_intercept,
                ]

                var_test_rescaled_offset = fitter.variance_explained(
                    (x_samples, y_samples), voxel_test
                )

                model_perf = (
                    var_test_rescaled / noise_ceiling if noise_ceiling != 0 else 0.0
                )

                model_performance_non_rescaled = var_test / noise_ceiling_unscaled

                fits_roi.at[index, "var_train"] = var_train
                fits_roi.at[index, "var_test"] = var_test
                fits_roi.at[index, "var_test_rescaled"] = var_test_rescaled
                fits_roi.at[index, "var_test_rescaled_offset"] = (
                    var_test_rescaled_offset
                )

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



