import sys
import os
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback_context
from scipy.optimize import curve_fit
from utils.config import load_config, Configuration
from utils.utils import (
    retrieve_stacked_betas,
    # retrieve_stacked_betas_test,
    retrieve_roi_mask,
    filter_roi_mask,
    logging_message,
    subjects_list_unifier,
)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
from tqdm import tqdm
import logging
from t_testing.clean_roi_mask import modify_mask_with_ttest


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def rescale(train, test):
    if len(train.shape) == 1:
        train = np.array([train])  # add a dimension for concat
    train_ones = np.concatenate((train, np.ones((train.shape[0], train.shape[1])))).T
    scale = np.linalg.pinv(train_ones) @ test.T
    new_train = train_ones @ scale
    return new_train.T.squeeze(), scale


class Gaussian2DFitter:
    @staticmethod
    def gaussian_2d_cartesian(xy_tuple, amplitude, x0, y0, sigma, offset):
        x, y = xy_tuple
        return (
            amplitude
            * np.exp(
                -(((x - x0) ** 2) / (2 * sigma**2) + ((y - y0) ** 2) / (2 * sigma**2))
            )
            + offset
        )

    def __init__(self):
        self.params = None
        self.solved = False

    def fit(self, x_samples, y_samples, target_values):
        p0 = [
            np.max(target_values) - np.min(target_values),
            np.mean(x_samples),
            np.mean(y_samples),
            (np.max(x_samples) - np.min(x_samples)) / 4,
            np.min(target_values),
        ]
        try:
            params, _ = curve_fit(
                Gaussian2DFitter.gaussian_2d_cartesian,
                (x_samples, y_samples),
                target_values,
                p0=p0,
            )
            self.params = params
            self.solved = True
            return params
        except Exception as e:
            self.solved = False
            print(f"Fitting failed: {str(e)}")
            return None


def load_data(config: Configuration, subj, mask_value, set_to_take):

    if set_to_take == "shared":
        use_shared = True
    else:
        use_shared = False

    threshold_mask = config.pipeline.step_5_gaussian_fitting.t_test_filtering
    sub_filename = f"cleanedrois_{str(threshold_mask).replace('.', '_')}"

    modify_mask_with_ttest(config, threshold_mask, use_shared, subj, sub_filename)

    mask = retrieve_roi_mask(config, subj, set_to_take, True, sub_filename)
    betas, _, mds_mapping = retrieve_stacked_betas(
        config, subj, "averaged", 0, subj_to_check=set_to_take
    )
    # betas_test = retrieve_stacked_betas_test(config, subj, subj_to_check=set_to_take)

    betas_test, _, _ = retrieve_stacked_betas(
        config, subj, "averaged", 0, subj_to_check=set_to_take, test=True
    )


    mask_voxel_indices = filter_roi_mask(mask_value, mask)[0]
    betas_masked = betas.T[mask_voxel_indices]
    betas_test_masked = betas_test.T[mask_voxel_indices]
    data = np.load(
        f"data/mds_dir/{set_to_take}/subj_{subj:02d}/mask_{mask_value}_averaged_mds.npy"
    )
    metadata = np.load(f"data/rdm_dir/{set_to_take}/subj_{subj:02d}/metadata.npy")
    fitted_voxels_df = pd.read_excel(
        f"data/gaussian_results/{set_to_take}/subj{subj:02d}/fitted_voxels_mask_{mask_value}.xlsx",
        index_col=None,
    )
    return data, metadata, betas_masked, betas_test_masked, fitted_voxels_df


def create_visualization(fitted_voxels_df, data, amplitudes, test_amplitudes):
    app = Dash(__name__)
    app.layout = html.Div(
        [html.H1("2D Gaussian Visualization"), dcc.Graph(id="main-plot")]
    )

    @app.callback(Output("main-plot", "figure"), [Input("main-plot", "id")])
    def update_plot(_):
        fig = go.Figure()
        fig.update_layout(title="Gaussian Visualization")
        return fig

    return app


def create_integrated_visualization(
    fitted_voxels_df, data, amplitudes, test_amplitudes, width=1500, height=800
):
    app = Dash(
        __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    )

    # Create a mapping from original_index to DataFrame index
    original_to_df_index = {
        int(row["original_index"]): i for i, row in fitted_voxels_df.iterrows()
    }
    voxel_ids = sorted(list(original_to_df_index.keys()))

    app.layout = html.Div(
        [
            html.H1("2D Gaussian Visualization", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Button(
                        "Previous Voxel",
                        id="prev-btn",
                        n_clicks=0,
                        style={"margin": "10px", "padding": "10px 20px"},
                    ),
                    html.Button(
                        "Next Voxel",
                        id="next-btn",
                        n_clicks=0,
                        style={"margin": "10px", "padding": "10px 20px"},
                    ),
                ],
                style={"textAlign": "center"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Controls"),
                            dcc.RadioItems(
                                id="view-selector",
                                options=[
                                    {"label": " 3D View", "value": "3D"},
                                    {"label": " 2D View", "value": "2D"},
                                ],
                                value="3D",
                                style={"margin": "20px 0"},
                            ),
                            dcc.Checklist(
                                id="toggle-points",
                                options=[
                                    {
                                        "label": " Show Training Points",
                                        "value": "Train",
                                    },
                                    {"label": " Show Test Points", "value": "Test"},
                                    {
                                        "label": " Show Rescaled Train Points",
                                        "value": "RescaledTrain",
                                    },
                                ],
                                value=["Train", "Test"],
                                style={"margin": "20px 0"},
                            ),
                            html.H4("Gaussian Fit Parameters"),
                            dcc.RadioItems(
                                id="fit-selector",
                                options=[
                                    {"label": " Original Fit", "value": "original"},
                                    {"label": " Rescaled Fit", "value": "rescaled"},
                                ],
                                value="original",
                                style={"margin": "20px 0"},
                            ),
                            html.H4("Search by Voxel ID"),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="voxel-id-input",
                                        type="number",
                                        placeholder="Enter voxel ID",
                                        min=min(voxel_ids) if voxel_ids else 0,
                                        max=max(voxel_ids) if voxel_ids else 1000,
                                        step=1,
                                        style={
                                            "width": "150px",
                                            "marginRight": "10px",
                                            "padding": "5px",
                                        },
                                    ),
                                    html.Button(
                                        "Find",
                                        id="find-voxel-btn",
                                        n_clicks=0,
                                        style={"padding": "5px 15px"},
                                    ),
                                ],
                                style={
                                    "marginBottom": "20px",
                                    "display": "flex",
                                    "alignItems": "center",
                                },
                            ),
                            html.Div(
                                id="search-result",
                                style={
                                    "marginTop": "10px",
                                    "marginBottom": "10px",
                                    "color": "#555",
                                },
                            ),
                            html.Div(
                                id="voxel-info",
                                style={
                                    "marginTop": "20px",
                                    "padding": "10px",
                                    "backgroundColor": "#f8f9fa",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px",
                                },
                            ),
                            dcc.Store(id="index-store", data=0),
                            dcc.Store(id="params-store", data={}),
                            dcc.Store(
                                id="original-index-map", data=original_to_df_index
                            ),
                            dcc.Store(id="rescaled-train-store", data={}),
                        ],
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "padding": "20px",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="main-plot",
                                style={"height": "600px"},
                                config={"displayModeBar": True},
                            ),
                        ],
                        style={"width": "75%", "display": "inline-block"},
                    ),
                ]
            ),
            html.Div(
                [
                    html.H4("Visualization Legend:"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        style={
                                            "backgroundColor": "blue",
                                            "width": "20px",
                                            "height": "20px",
                                            "display": "inline-block",
                                            "borderRadius": "50%",
                                            "margin": "0 10px 0 0",
                                        }
                                    ),
                                    "Training data",
                                ],
                                style={
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        style={
                                            "backgroundColor": "red",
                                            "width": "15px",
                                            "height": "15px",
                                            "display": "inline-block",
                                            "borderRadius": "50%",
                                            "margin": "0 10px 0 0",
                                        }
                                    ),
                                    "Test data",
                                ],
                                style={
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        style={
                                            "backgroundColor": "green",
                                            "width": "15px",
                                            "height": "15px",
                                            "display": "inline-block",
                                            "borderRadius": "50%",
                                            "margin": "0 10px 0 0",
                                        }
                                    ),
                                    "Rescaled Training data",
                                ],
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                ],
                style={"textAlign": "center", "marginTop": "20px"},
            ),
        ]
    )

    @app.callback(
        [
            Output("params-store", "data"),
            Output("voxel-info", "children"),
            Output("rescaled-train-store", "data"),
        ],
        [Input("index-store", "data"), Input("fit-selector", "value")],
    )
    def initialize_parameters(voxel_index, fit_type):
        row = fitted_voxels_df.iloc[voxel_index]

        # Choose the appropriate slope and intercept based on fit type
        if fit_type == "original":
            amplitude = float(row.get("slope", 1.0))
            offset = float(row.get("intercept", 0.0))
            fit_name = "Original Fit"
        else:  # rescaled
            amplitude = float(row.get("rescaled_slope", 1.0))
            offset = float(row.get("rescaled_intercept", 0.0))
            fit_name = "Rescaled Fit"

        # Common parameters for both fits
        params = {
            "amplitude": amplitude,
            "x0": float(row.x0),
            "y0": float(row.y0),
            "sigma": float(row.sigma),
            "offset": offset,
            "var_train": float(row.get("var_train", 0)),
            "var_test": float(row.get("var_test", 0)),
            "var_test_rescaled": float(row.get("var_test_rescaled", 0)),
            "noise_ceiling": float(row.get("noise_ceiling", 0)),
            "fit_type": fit_type,
        }

        # Calculate rescaled training values
        train_values = amplitudes[voxel_index]
        test_values = test_amplitudes[voxel_index]

        # Reshape for the rescale function
        train_reshaped = train_values.reshape(1, -1)
        test_reshaped = test_values.reshape(1, -1)

        rescaled_train, scale = rescale(train_reshaped, test_reshaped)

        # Store for visualization
        rescaled_train_data = {
            "values": rescaled_train.tolist(),
            "scale": scale.tolist() if hasattr(scale, "tolist") else scale,
        }

        # Create the information panel with side-by-side comparison
        info_content = [
            html.H4(
                f"Voxel {voxel_index + 1} of {len(fitted_voxels_df)} (id: {row['original_index']})"
            ),
            html.H5(fit_name, style={"color": "#2c3e50", "marginTop": "10px"}),
            # Main parameters
            html.Div(
                [
                    # Left column - parameters
                    html.Div(
                        [
                            html.P(
                                f"Position: ({params['x0']:.2f}, {params['y0']:.2f})"
                            ),
                            html.P(f"Slope: {params['amplitude']:.2f}"),
                            html.P(f"Intercept: {params['offset']:.2f}"),
                            html.P(f"Sigma: {params['sigma']:.2f}"),
                            html.P(f"Train Variance: {params['var_train']:.2f}"),
                            html.P(f"Test Variance: {params['var_test']:.2f}"),
                            html.P(
                                f"Rescaled Gaussian Variance (test): {params['var_test_rescaled']:.2f}"
                            ),
                            html.P(f"Noise Ceiling: {params['noise_ceiling']:.2f}"),
                        ],
                        style={
                            "width": "50%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                    # Right column - comparison (if available)
                    html.Div(
                        [
                            (
                                html.H5("Fit Comparison")
                                if "slope" in row and "rescaled_slope" in row
                                else None
                            ),
                        ],
                        style={
                            "width": "50%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                        },
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap"},
            ),
        ]

        # Add comparison if both fits are available
        if "slope" in row and "rescaled_slope" in row:
            original_amp = float(row.get("slope", 0.0))
            rescaled_amp = float(row.get("rescaled_slope", 0.0))
            original_offset = float(row.get("intercept", 0.0))
            rescaled_offset = float(row.get("rescaled_intercept", 0.0))

            percent_diff_amp = (
                abs(
                    (original_amp - rescaled_amp)
                    / ((original_amp + rescaled_amp) / 2)
                    * 100
                )
                if (original_amp + rescaled_amp) != 0
                else 0
            )
            percent_diff_offset = (
                abs(
                    (original_offset - rescaled_offset)
                    / ((abs(original_offset) + abs(rescaled_offset)) / 2)
                    * 100
                )
                if (abs(original_offset) + abs(rescaled_offset)) != 0
                else 0
            )

            # Insert comparison into the right column
            info_content[2].children[1].children.extend(
                [
                    html.P(f"Rescaled Slope: {rescaled_amp:.2f}"),
                    html.P(f"Difference: {percent_diff_amp:.1f}%"),
                    html.P(f"Rescaled Intercept: {rescaled_offset:.2f}"),
                    html.P(f"Difference: {percent_diff_offset:.1f}%"),
                ]
            )

        return params, info_content, rescaled_train_data

    @app.callback(
        Output("index-store", "data"),
        [
            Input("prev-btn", "n_clicks"),
            Input("next-btn", "n_clicks"),
            Input("find-voxel-btn", "n_clicks"),
        ],
        [
            State("index-store", "data"),
            State("voxel-id-input", "value"),
            State("original-index-map", "data"),
        ],
    )
    def update_voxel_index(
        prev_clicks,
        next_clicks,
        find_clicks,
        current_index,
        voxel_id,
        original_index_map,
    ):
        ctx = callback_context
        trigger_id = (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        )

        if trigger_id == "prev-btn":
            return max(0, current_index - 1)
        elif trigger_id == "next-btn":
            return min(len(fitted_voxels_df) - 1, current_index + 1)
        elif trigger_id == "find-voxel-btn" and voxel_id is not None:
            # Convert to int for dictionary lookup
            voxel_id = int(voxel_id)
            if str(voxel_id) in original_index_map:
                return original_index_map[str(voxel_id)]
            elif voxel_id in original_index_map:
                return original_index_map[voxel_id]
            # If voxel ID not found, keep current index

        return current_index

    @app.callback(
        Output("search-result", "children"),
        [Input("find-voxel-btn", "n_clicks")],
        [State("voxel-id-input", "value"), State("original-index-map", "data")],
    )
    def display_search_result(n_clicks, voxel_id, original_index_map):
        if n_clicks == 0 or voxel_id is None:
            return ""

        voxel_id = int(voxel_id)
        if str(voxel_id) in original_index_map:
            return f"Voxel ID {voxel_id} found!"
        elif voxel_id in original_index_map:
            return f"Voxel ID {voxel_id} found!"
        else:
            return f"Voxel ID {voxel_id} not found."

    @app.callback(
        Output("main-plot", "figure"),
        [
            Input("index-store", "data"),
            Input("params-store", "data"),
            Input("toggle-points", "value"),
            Input("view-selector", "value"),
            Input("rescaled-train-store", "data"),
        ],
    )
    def update_plot(voxel_index, params, toggle_value, view_mode, rescaled_train_data):
        # Common parameters
        amp = params["amplitude"]
        x0 = params["x0"]
        y0 = params["y0"]
        sigma = params["sigma"]
        offset = params["offset"]
        fit_type = params.get("fit_type", "original")

        # Generate grid
        x_range = np.linspace(-1, 1, 50)
        y_range = np.linspace(-1, 1, 50)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        Z_grid = Gaussian2DFitter.gaussian_2d_cartesian(
            (X_grid, Y_grid), amp, x0, y0, sigma, offset
        )

        fig = go.Figure()
        show_train = "Train" in toggle_value
        show_test = "Test" in toggle_value
        show_rescaled_train = "RescaledTrain" in toggle_value

        x_samples, y_samples = data[:, 0], data[:, 1]
        train_values = amplitudes[voxel_index]
        test_values = test_amplitudes[voxel_index]

        # Get rescaled training values from the store
        rescaled_train_values = (
            np.array(rescaled_train_data.get("values", []))
            if rescaled_train_data
            else []
        )

        if view_mode == "3D":
            # 3D View
            fig.add_trace(
                go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale="Viridis",
                    opacity=0.7,
                    showscale=True,
                    name="Gaussian",
                    colorbar=dict(title="Gaussian Value", x=1.0),
                )
            )

            if show_train:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_samples,
                        y=y_samples,
                        z=train_values,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=train_values,
                            colorscale="Blues",
                            opacity=0.8,
                            showscale=False,
                        ),
                        name="Training Data",
                    )
                )

            if show_test:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_samples,
                        y=y_samples,
                        z=test_values,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=test_values,
                            colorscale="Reds",
                            opacity=0.8,
                            showscale=False,
                        ),
                        name="Test Data",
                    )
                )

            if show_rescaled_train and len(rescaled_train_values) > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=x_samples,
                        y=y_samples,
                        z=rescaled_train_values,
                        mode="markers",
                        marker=dict(
                            size=5,
                            color=rescaled_train_values,
                            colorscale="Greens",
                            opacity=0.8,
                            showscale=False,
                        ),
                        name="Rescaled Training Data",
                    )
                )

            fig.update_layout(
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Value",
                    camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )
        else:
            # 2D View
            # Calculate unified color scale for all data points
            all_values = []
            if len(train_values) > 0:
                all_values.append(train_values)
            if len(test_values) > 0:
                all_values.append(test_values)
            if show_rescaled_train and len(rescaled_train_values) > 0:
                all_values.append(rescaled_train_values)

            if all_values:
                all_values_array = np.concatenate(all_values)
                cmin, cmax = np.nanmin(all_values_array), np.nanmax(all_values_array)
            else:
                cmin, cmax = None, None

            # Gaussian contour
            fig.add_trace(
                go.Contour(
                    x=x_range,
                    y=y_range,
                    z=Z_grid,
                    colorscale="Viridis",
                    showscale=True,
                    name="Gaussian",
                    zmin=cmin,
                    zmax=cmax,
                    colorbar=dict(
                        title="Values", x=1.02, len=0.4, y=0.75, yanchor="middle"
                    ),
                )
            )

            # Rescaled Training points
            if show_rescaled_train and len(rescaled_train_values) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=x_samples,
                        y=y_samples,
                        mode="markers",
                        marker=dict(
                            size=30,
                            color=rescaled_train_values,
                            colorscale="Viridis",
                            cmin=cmin,
                            cmax=cmax,
                            opacity=0.8,
                            line=dict(width=1, color="black"),
                        ),
                        name="Rescaled Training Data",
                        showlegend=False,
                    )
                )

            # Training points
            if show_train:
                fig.add_trace(
                    go.Scatter(
                        x=x_samples,
                        y=y_samples,
                        mode="markers",
                        marker=dict(
                            size=20,
                            color=train_values,
                            colorscale="Viridis",
                            cmin=cmin,
                            cmax=cmax,
                            opacity=0.8,
                            line=dict(width=1, color="black"),
                        ),
                        name="Training Data",
                        showlegend=False,
                    )
                )

            # Test points
            if show_test:
                fig.add_trace(
                    go.Scatter(
                        x=x_samples,
                        y=y_samples,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=test_values,
                            colorscale="Viridis",
                            cmin=cmin,
                            cmax=cmax,
                            opacity=0.8,
                            line=dict(width=1, color="black"),
                        ),
                        name="Test Data",
                        showlegend=False,
                    )
                )

            fig.update_layout(
                xaxis_title="X",
                yaxis_title="Y",
                # Force square axes
                xaxis=dict(
                    scaleanchor="y", scaleratio=1, range=[-1, 1], constrain="domain"
                ),
                yaxis=dict(range=[-1, 1], constrain="domain"),
                # Square figure dimensions
                autosize=False,
                width=800,
                height=800,
                # Margins to accommodate colorbar
                margin=dict(l=50, r=100, b=50, t=50),
                # Disable legend scaling
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
            )

        # Add visualization details to the title
        visible_data = []
        if show_train:
            visible_data.append("Train")
        if show_test:
            visible_data.append("Test")
        if show_rescaled_train:
            visible_data.append("Rescaled Train")

        data_label = " | ".join(visible_data)
        fit_label = "Original Fit" if fit_type == "original" else "Rescaled Fit"

        fig.update_layout(
            title=f"Voxel {voxel_index + 1} - {view_mode} View - {fit_label} - {data_label}",
            width=width,
            height=height,
            template="plotly_white",
        )

        return fig

    return app


def visualize_gaussian_results(config: Configuration):

    subjects = subjects_list_unifier(
        config.pipeline.step_5_gaussian_fitting.subjects, False
    )

    if config.pipeline.step_5_gaussian_fitting.visualize:
        logging.info(
            logging_message(
                config.pipeline.step_5_gaussian_fitting.step, "Starting Visualization"
            )
        )

    else:
        logging.info(
            logging_message(
                config.pipeline.step_5_gaussian_fitting.step, "Skipping Visualization"
            )
        )

    assert len(subjects) == 1, "This visualization can take only one subject"
    for subject in subjects:
        if subject == "shared":
            set_to_take = "shared"
            subject_list = range(1, 8 + 1)
            raise NotImplementedError(f"Sorry, not implemented yet...")

        else:
            set_to_take = f"subj_{subject:02d}"

        mask_value = 8
        data, metadata, betas_masked, betas_test_masked, fitted_voxels_df = load_data(
            config, subject, mask_value, set_to_take
        )
        # app = create_visualization(
        #     fitted_voxels_df, data, betas_masked, betas_test_masked
        # )
        app = create_integrated_visualization(
            fitted_voxels_df, data, betas_masked, betas_test_masked
        )
        app.run_server(debug=True, port=8051, host="0.0.0.0")


def plot_roi_boxplot(
    all_data,
    value_column,
    pdf,
    plot_title="Within ROI Model (Boxplot with Means)",
    xlabel="ROI",
    ylabel="Variance explained",
    ylim=(-1, 1),
    figsize=(8, 6),
    palette="rainbow",
    log_scale=False,
):
    """
    Loads data for each ROI from Excel files and creates a boxplot with
    Seaborn. The function overlays the mean of each ROI as a diamond marker.

    Parameters:
        all_data (list):
        value_column (str): Column name in the Excel files to analyze.
        plot_title (str): Plot title.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        ylim (tuple): Y-axis limits.
        figsize (tuple): Figure size.
        palette (str): Name of the Seaborn color palette.
    """
    df = pd.concat(all_data, ignore_index=True)

    # Set Seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=figsize)
    ax = sns.boxplot(x="ROI", y=value_column, data=df, palette=palette)

    # Overlay the mean values as diamond markers
    means = df.groupby("ROI", as_index=False)[value_column].mean()
    means["x"] = means["ROI"] - 1  # Adjust x positions to match the boxplot positions

    sns.scatterplot(
        x="x",
        y=value_column,
        data=means,
        color="black",
        marker="D",
        s=100,
        ax=ax,
        zorder=10,
    )

    # Customize labels and title
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if ylim:
        ax.set(ylim=ylim)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()
    pdf.savefig()  # Speichert die aktuelle Figur
    plt.close()


def evaluate(config: Configuration):
    threshold_mask = config.pipeline.step_5_gaussian_fitting.t_test_filtering
    sub_filename = f"cleanedrois_{str(threshold_mask).replace('.', '_')}"

    subjects = subjects_list_unifier(
        config.pipeline.step_5_gaussian_fitting.subjects, False
    )

    if config.pipeline.step_5_gaussian_fitting.evaluate:
        logging.info(
            logging_message(
                config.pipeline.step_5_gaussian_fitting.step, "Starting Evaluation"
            )
        )

    else:
        logging.info(
            logging_message(
                config.pipeline.step_5_gaussian_fitting.step, "Skipping Evaluation"
            )
        )

    for subject in subjects:
        if subject == "shared":
            set_to_take = "shared"
            subject_list = list(range(1, 8 + 1))
            use_shared = True

        else:
            set_to_take = f"subj_{subject:02d}"
            subject_list = [subject]
            use_shared = False

        for s in subject_list:
            modify_mask_with_ttest(config, threshold_mask, use_shared, s, sub_filename)

            mask = retrieve_roi_mask(config, s, set_to_take, True, sub_filename)

            rois = list(range(1, 8 + 1))
            all_data = []
            for roi in rois:
                file_path = os.path.join(
                    config.directories.gaussian_fit_results_dir,
                    set_to_take,
                    f"subj{s:02d}",
                    f"fitted_voxels_mask_{roi}.xlsx",
                )
                try:
                    data = pd.read_excel(file_path)

                    mask_voxel_indices = filter_roi_mask(roi, mask)[0]
                    data = data[data["original_index"].isin(mask_voxel_indices)]
                    # quit()

                    sigma_lower_threshold, sigma_upper_threshold = (
                        config.pipeline.step_5_gaussian_fitting.sigma_filtering
                    )

                    if (
                        sigma_lower_threshold is not None
                        and sigma_upper_threshold is not None
                    ):
                        total_before = data.shape[0]
                        n_below = (data["sigma"] < sigma_lower_threshold).sum()
                        n_above = (data["sigma"] > sigma_upper_threshold).sum()
                        data = data[
                            (data["sigma"] >= sigma_lower_threshold)
                            & (data["sigma"] <= sigma_upper_threshold)
                        ]
                        total_after = data.shape[0]
                        logging.info(
                            f"ROI {roi}: Filtered out {n_below} samples below and {n_above} samples above thresholds. Remaining samples: {total_after}/{total_before}."
                        )

                    data["ROI"] = roi
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

            create_variance_mask(
                config, pd.concat(all_data, ignore_index=True), set_to_take, s
            )

            create_sigma_mask(
                config, pd.concat(all_data, ignore_index=True), set_to_take, s
            )

            with PdfPages(
                os.path.join("data", "plots", f"subj{s:02d}_{threshold_mask}.pdf")
            ) as pdf:

                plot_roi_boxplot(
                    all_data,
                    value_column="var_train",
                    pdf=pdf,
                    plot_title="Variance Train set",
                    xlabel="ROI",
                    ylabel="Variance explained (train)",
                    ylim=(-1, 1),
                )

                plot_roi_boxplot(
                    all_data,
                    value_column="var_test",
                    pdf=pdf,
                    plot_title="Variance Test set",
                    xlabel="ROI",
                    ylabel="Variance explained (test)",
                    ylim=(-1, 1),
                )

                plot_roi_boxplot(
                    all_data,
                    value_column="var_test_rescaled",
                    pdf=pdf,
                    plot_title="Variance Test set rescaled",
                    xlabel="ROI",
                    ylabel="Variance explained (test rescaled)",
                    ylim=(-1, 1),
                )

                plot_roi_boxplot(
                    all_data,
                    value_column="sigma",
                    pdf=pdf,
                    plot_title="Sigmas",
                    xlabel="ROI",
                    ylabel="Sigma",
                    ylim=(-0.1, 10),
                    log_scale=False,
                )
                plot_roi_boxplot(
                    all_data,
                    value_column="sigma",
                    pdf=pdf,
                    plot_title="Sigmas Log scale",
                    xlabel="ROI",
                    ylabel="Sigma",
                    ylim=(10e-4, 10e3),
                    log_scale=True,
                )


def create_variance_mask(
    config: Configuration,
    results_df,
    subset: str,
    subj_to_pick: int,
):
    """
    Creates variance mask files based on Gaussian fit results.

    Args:
        config: Configuration object containing necessary paths
        subj_to_pick_shared: Boolean indicating whether to use shared subject data
        subj_to_pick: Subject ID number to process
    """
    hemis = ["lh", "rh"]
    mask_dir = config.directories.t_test_roi_dir

    # Create a lookup dictionary for faster access
    var_train_lookup = {}
    if not results_df.empty:
        # Create a dictionary mapping original_index to var_train for O(1) lookups
        var_train_lookup = dict(
            zip(results_df["original_index"], results_df["var_train"])
        )

    # Process each hemisphere
    vertex_offset = 0
    for hemi in hemis:
        # Construct file paths
        maskdata_in_file = os.path.join(
            mask_dir, subset, f"{hemi}.subj{subj_to_pick:02d}.testrois.mgz"
        )
        data_out_file = os.path.join(
            config.nsd_data.freesurfer_dir,
            f"subj{subj_to_pick:02d}",
            "label",
            f"{hemi}.variance_map.mgz",
        )

        # Load mask data
        maskdata = nib.load(maskdata_in_file).get_fdata().squeeze()

        # Pre-allocate output array
        data_out = np.zeros(maskdata.shape)

        # Process vertices
        for i in tqdm(range(maskdata.shape[0]), desc=f"Processing {hemi}"):
            index = i + vertex_offset
            var_train_value = var_train_lookup.get(index)
            if var_train_value is not None:
                data_out[i] = var_train_value

        # Save the output
        img = nib.Nifti1Image(np.expand_dims(data_out, axis=(1, 2)), np.eye(4))
        nib.loadsave.save(img, data_out_file)
        logging.info(f"Saved to {data_out_file}")

        # Update vertex offset for right hemisphere
        if hemi == "lh":
            vertex_offset = maskdata.shape[0]


def create_sigma_mask(
    config: Configuration,
    results_df,
    subset: str,
    subj_to_pick: int,
):
    """
    Creates sigma mask files based on Gaussian fit results and stores values in log scale.

    Args:
        config: Configuration object containing necessary paths
        results_df: DataFrame containing Gaussian fit results
        subset: Subset name
        subj_to_pick: Subject ID number to process
    """
    hemis = ["lh", "rh"]
    mask_dir = config.directories.t_test_roi_dir

    # Create a lookup dictionary for faster access
    sigma_lookup = {}
    if not results_df.empty:
        # Create a dictionary mapping original_index to log10(sigma)
        sigma_lookup = dict(
            zip(results_df["original_index"], np.log10(results_df["sigma"]))
        )

    # Process each hemisphere
    vertex_offset = 0
    for hemi in hemis:
        # Construct file paths
        maskdata_in_file = os.path.join(
            mask_dir, subset, f"{hemi}.subj{subj_to_pick:02d}.testrois.mgz"
        )
        data_out_file = os.path.join(
            config.nsd_data.freesurfer_dir,
            f"subj{subj_to_pick:02d}",
            "label",
            f"{hemi}.sigma_map.mgz",
        )

        # Load mask data
        maskdata = nib.load(maskdata_in_file).get_fdata().squeeze()

        # Pre-allocate output array
        data_out = np.zeros(maskdata.shape)

        # Process vertices
        for i in tqdm(range(maskdata.shape[0]), desc=f"Processing {hemi}"):
            index = i + vertex_offset
            sigma_value = sigma_lookup.get(index)
            if sigma_value is not None:
                # offset bc of negative values
                data_out[i] = sigma_value + 1

        # Save the output
        img = nib.Nifti1Image(np.expand_dims(data_out, axis=(1, 2)), np.eye(4))
        nib.loadsave.save(img, data_out_file)
        logging.info(f"Saved to {data_out_file}")

        # Update vertex offset for right hemisphere
        if hemi == "lh":
            vertex_offset = maskdata.shape[0]


if __name__ == "__main__":
    config = load_config("config.yaml")
    visualize_gaussian_results(config)
    evaluate(config)
