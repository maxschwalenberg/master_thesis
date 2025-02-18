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


# ------------------------------------------------------------------
# 1) Define an isotropic 2D Gaussian in Cartesian space
#    (center = (x0, y0), single spread = sigma for both x and y)
# ------------------------------------------------------------------
def gaussian_2d_cartesian_isotropic(coords, A, x0, y0, sigma):
    """
    coords: tuple of (x, y) arrays
    A:       amplitude (peak value)
    x0, y0:  center of the Gaussian in Cartesian coordinates
    sigma:   standard deviation (equal in x and y)
    """
    x, y = coords
    exponent = -((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)
    return A * np.exp(exponent)


# ------------------------------------------------------------------
# 2) Fit the isotropic 2D Gaussian using curve_fit
# ------------------------------------------------------------------
def fit_cartesian_gaussian_isotropic(x_vals, y_vals, target_values):
    """
    x_vals, y_vals: 1D arrays of sample coordinates
    target_values:   1D array of values at (x_vals, y_vals)
    Returns:
      popt: optimized parameters [A, x0, y0, sigma]
    """
    # Initial guess:
    #   - Amplitude: max(target_values)
    #   - Center: mean(x), mean(y)
    #   - Spread: use the average of the standard deviations in x and y (or one of them)
    A0 = np.max(target_values)
    x0 = np.mean(x_vals)
    y0 = np.mean(y_vals)
    sigma0 = max(np.std(x_vals), np.std(y_vals), 1e-6)

    # initial_guess = [np.random.uniform(0.1, A0), x0, y0, sigma0]
    initial_guess = [
        np.random.uniform(0.1, A0),
        np.random.uniform(-1, 1),
        np.random.uniform(-1, 1),
        sigma0,
    ]

    # print(initial_guess)

    # Set bounds: amplitude >= 0, sigma > 0
    lower_bounds = [0, -1.1, -1.1, 0.0001]
    upper_bounds = [np.inf, 1.1, 1.1, 100]

    attempt = 1
    solved = False
    while attempt <= 3 and not solved:
        try:
            popt, pcov = curve_fit(
                gaussian_2d_cartesian_isotropic,
                (x_vals, y_vals),
                target_values,
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                # maxfev=2000,
            )
            solved = True
        except:
            if attempt > 2:
                print(f"{attempt=}")

            attempt += 1

    if solved:
        return popt, solved
    else:
        popt = np.mean(target_values), 0, 0, 100
        # raise RuntimeError
        return popt, solved


# ------------------------------------------------------------------
# 3) Compute Variance Explained (R²)
# ------------------------------------------------------------------
def variance_explained(y_true, y_pred):
    """
    Returns R^2 = 1 - (SS_res / SS_tot).
    In the worst-case scenario the Gaussian can mimic a constant (the mean),
    so the model will never be worse than predicting the mean.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # total variance
    ss_residual = np.sum((y_true - y_pred) ** 2)  # residual variance
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


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

    @staticmethod
    def gaussian_2d_cartesian(coords, A, x0, y0, sigma):
        """
        Isotropic 2D Gaussian in Cartesian coordinates.

        Parameters:
          coords: tuple of (x, y) arrays
          A: amplitude
          x0, y0: center of the Gaussian
          sigma: standard deviation
        """
        x, y = coords
        exponent = -(((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))
        return A * np.exp(exponent)

    @staticmethod
    def gaussian_2d_polar(coords, A, r0, theta0, sigma):
        """
        Isotropic 2D Gaussian in polar coordinates.

        Parameters:
          coords: tuple of (r, theta) arrays
          A: amplitude
          r0, theta0: center in polar coordinates
          sigma: standard deviation

        The distance between a point (r, theta) and the center (r0, theta0)
        is computed via the law of cosines:

            d = sqrt(r**2 + r0**2 - 2*r*r0*cos(theta - theta0))
        """
        r, theta = coords
        d = np.sqrt(r**2 + r0**2 - 2 * r * r0 * np.cos(theta - theta0))
        exponent = -(d**2) / (2 * sigma**2)
        return A * np.exp(exponent)

    def fit(self, x_vals, y_vals, target_values):
        """
        Fit the isotropic 2D Gaussian to the provided data.

        Parameters:
          x_vals, y_vals: 1D arrays of sample coordinates
          target_values: 1D array of values at (x_vals, y_vals)

        Returns:
          (popt, solved) where popt is the optimized parameters [A, x0, y0, sigma]
          and solved is a boolean indicating whether the fit succeeded.
        """
        A0 = np.max(target_values)
        attempt = 1
        solved = False

        if self.use_polar:
            # Convert the Cartesian samples to polar coordinates.
            r_samples = np.sqrt(x_vals**2 + y_vals**2)
            theta_samples = np.arctan2(y_vals, x_vals)
            sigma0 = max(np.std(r_samples), 1e-6)
            # Set bounds: amplitude >= 0, r0 in [0,1] (unit circle), theta0 in [-pi, pi], sigma > 0.
            lower_bounds = [0, 0, -np.pi, 1e-6]
            upper_bounds = [np.inf, 1.0, np.pi, 100]
        else:
            sigma0 = max(np.std(x_vals), np.std(y_vals), 1e-6)
            # Bounds for Cartesian fit: amplitude >= 0, x0,y0 within [-1.1, 1.1], sigma > 0.
            lower_bounds = [0, -1.1, -1.1, 1e-6]
            upper_bounds = [np.inf, 1.1, 1.1, 100]

        while attempt <= 3 and not solved:
            try:
                if self.use_polar:
                    # Initial guess for polar parameters: [A, r0, theta0, sigma]
                    initial_guess = [
                        np.random.uniform(0.1, A0),
                        np.random.uniform(0, 1),
                        np.random.uniform(-np.pi, np.pi),
                        sigma0,
                    ]
                    # Fit using the polar version of the Gaussian.
                    popt, pcov = curve_fit(
                        Gaussian2DFitter.gaussian_2d_polar,
                        (r_samples, theta_samples),
                        target_values,
                        p0=initial_guess,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=2000,
                    )
                    # Convert the polar center to Cartesian coordinates.
                    A, r0, theta0, sigma = popt
                    x0 = r0 * np.cos(theta0)
                    y0 = r0 * np.sin(theta0)
                    self.params = [A.item(), x0.item(), y0.item(), sigma.item()]
                else:
                    # Initial guess for Cartesian parameters: [A, x0, y0, sigma]
                    initial_guess = [
                        np.random.uniform(0.1, A0),
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1),
                        sigma0,
                    ]
                    popt, pcov = curve_fit(
                        Gaussian2DFitter.gaussian_2d_cartesian,
                        (x_vals, y_vals),
                        target_values,
                        p0=initial_guess,
                        bounds=(lower_bounds, upper_bounds),
                        maxfev=2000,
                    )
                    self.params = popt
                solved = True
            except Exception as e:
                attempt += 1
                if attempt > 3:
                    print("Fitting failed after 3 attempts:", e)
        self.solved = solved

        # In case of failure, return a fallback set of parameters.
        if not solved:
            self.params = [np.mean(target_values).item(), 0, 0, 100]
        return self.params, solved

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

        A, x0, y0, sigma = self.params

        # Convert input coordinates to Cartesian if they are provided in polar form.
        if input_polar:
            r, theta = coords
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coords_cartesian = (x, y)
        else:
            coords_cartesian = coords

        return Gaussian2DFitter.gaussian_2d_cartesian(
            coords_cartesian, A, x0, y0, sigma
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

        popt = self.params  # [A, x0, y0, sigma]

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
        ss_res = np.sum((target_values - predictions) ** 2)
        ss_tot = np.sum((target_values - np.mean(target_values)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        return r_squared


def fit_gaussian_params(config: Configuration, subj_list: list[int], set_to_take: str):
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

    columns_polar = ["ecc", "pol", "sigma", "slope", "intercept"]
    columns = [
        "x0",
        "y0",
        "sigma",
        "slope",
        "intercept",
        "solved",
        "var_train",
        "var_test",
    ]
    initial = params["initial"]
    bounds = params["bounds"]

    for i, subj in enumerate(subj_list):
        gaussian_fit_result_dir_path = os.path.join(
            config.gaussian_fit_results_dir, set_to_take, f"subj{subj:02d}"
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
                set_to_take,
                f"subj_{subj:02d}",
                f"mask_{roi_mask_value}_averaged_mds.npy",
            )
            mds = np.load(mds_file, allow_pickle=True).astype(np.float32).T

            mask_voxel_indices = filter_roi_mask(roi_mask_value, mask)
            mask_voxel_indices = mask_voxel_indices[0]

            # masked_voxel_betas_test = betas_test.T[mask_voxel_indices].T

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

                    fitter = Gaussian2DFitter(use_polar=False)

                    x_samples, y_samples = (mds[0, :], mds[1, :])
                    popt, solved = fitter.fit(x_samples, y_samples, voxel_activity)

                    var_train = fitter.variance_explained(
                        (x_samples, y_samples), voxel_activity
                    )
                    var_test = fitter.variance_explained(
                        (x_samples, y_samples), voxel_activity_test
                    )

                    A, x0, y0, sigma = popt

                    slope = A  # Directly assigning A as amplitude
                    intercept = 0  # Assuming no offset
                    voxel_fit = [
                        x0,
                        y0,
                        sigma,
                        slope,
                        intercept,
                        solved,
                        var_train,
                        var_test,
                    ]

                    fits_roi.loc[voxel_i] = voxel_fit

                # fits_roi["x0"], fits_roi["y0"] = pol2cart(
                #     fits_roi["ecc"], fits_roi["pol"]
                # )
                # Add the current index as a new column
                fits_roi["original_index"] = fits_roi.index

                # Reset the index to 0, 1, 2, ...
                fits_roi = fits_roi.reset_index(drop=True)

                fits_roi.to_excel(gaussian_result_file_path, index=False)

            # if true - no computation needed
            # if "variance_explained" in fits_roi.columns:
            if False:
                logging.info("Variance explained already computed")

            else:
                # print(fits_roi.head())
                logging.info("Computing variance stats")
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
                # (n_samples, n_voxels)
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

                # # Apply Gaussian function to each row in fits_roi
                # predicted_activity = fits_roi[required_columns].apply(
                #     apply_gaussian_fit, axis=1
                # )

                # print(f"{predicted_activity.shape=}")

                # # Convert the results to a NumPy array (ensure proper shape)
                # predicted_activity = np.array(
                #     [np.array(activity) for activity in predicted_activity]
                # ).T

                # predicted_activity = masked_voxel_betas

                # Compute amplitudes for each trial and voxel
                predicted_activity = np.zeros((mds.shape[1], len(mask_voxel_indices)))

                activity_mean = np.zeros((mds.shape[1], len(mask_voxel_indices)))

                for v in range(len(mask_voxel_indices)):
                    x0, y0, sigma, slope = fits_roi.loc[
                        v, ["x0", "y0", "sigma", "slope"]
                    ]
                    x_trials, y_trials = (
                        mds.T[:, 0],
                        mds.T[:, 1],
                    )  # Extract trial coordinates

                    # Compute Gaussian amplitude for each trial at this voxel
                    predicted_activity[:, v] = slope * np.exp(
                        -((x_trials - x0) ** 2 + (y_trials - y0) ** 2) / (2 * sigma**2)
                    )

                    activity_mean[:, v] = masked_voxel_betas[:, v].mean()

                # print(f"{predicted_activity.shape=}")
                # print(f"{masked_voxel_betas.shape=}")

                # Compute Residual Sum of Squares (RSS) and Total Sum of Squares (TSS)
                residual_sum_squares = np.sum(
                    (predicted_activity - masked_voxel_betas) ** 2, axis=0
                )

                residual_sum_squares_mean = np.sum(
                    (activity_mean - masked_voxel_betas) ** 2, axis=0
                )

                # print(residual_sum_squares[:5])
                total_sum_squares = np.sum(
                    (masked_voxel_betas - masked_voxel_betas.mean(axis=0)) ** 2, axis=0
                )

                # print(total_sum_squares[:5])

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

                fits_roi["sanity_mean"] = 1 - (
                    residual_sum_squares_mean / (total_sum_squares + EPSILON)
                )

                fits_roi["sanity_correct"] = 1 - (
                    np.sum((masked_voxel_betas - masked_voxel_betas) ** 2, axis=0)
                    / (total_sum_squares + EPSILON)
                )

                # Compute variance explained (R² score)
                fits_roi["variance_explained_test"] = 1 - (
                    residual_sum_squares_test / (total_sum_squares_test + EPSILON)
                )

                fits_roi["sanity_correct_test"] = 1 - (
                    np.sum((masked_voxel_betas - masked_voxel_betas_test) ** 2, axis=0)
                    / (total_sum_squares_test + EPSILON)
                )

                sanity_correct_test_stat = fits_roi["sanity_correct_test"].describe()
                # Calculate the positive percentile
                positive_percentile = find_positive_percentile(
                    fits_roi["sanity_correct_test"]
                )

                # Add the custom statistic to the results
                if positive_percentile is not None:
                    sanity_correct_test_stat["positive_percentile"] = (
                        positive_percentile
                    )
                else:
                    sanity_correct_test_stat["positive_percentile"] = (
                        "No positive values"
                    )

                print(
                    f"Stats for test set with actual train values as predictions:\n{sanity_correct_test_stat}"
                )

                variance_explained_stat = fits_roi["variance_explained"].describe()
                # Calculate the positive percentile
                positive_percentile = find_positive_percentile(
                    fits_roi["variance_explained"]
                )

                # Add the custom statistic to the results
                if positive_percentile is not None:
                    variance_explained_stat["positive_percentile"] = positive_percentile
                else:
                    variance_explained_stat["positive_percentile"] = (
                        "No positive values"
                    )

                print(
                    f"Stats for train set with predicted values by model:\n{variance_explained_stat}"
                )

                fits_roi.to_excel(gaussian_result_file_path, index=False)

                print("-" * 30)


if __name__ == "__main__":
    config = load_config("config.yaml")
    fit_gaussian_params(config, [1], "shared")
