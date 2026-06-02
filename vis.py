import json
import math
import os
from collections import Counter
from typing import Dict, List

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from utils.config import load_config


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


@st.cache_data
def load_df_to_check(path: str) -> pd.DataFrame:
    """
    Read the sheet of all images you need to label.
    Handles various column name variations for file names.
    """
    df = pd.read_excel(path)

    # Standardize column names for file paths
    if "file_name" not in df.columns:
        if "filename" in df.columns:
            df.rename(columns={"filename": "file_name"}, inplace=True)
        elif "image_name" in df.columns:
            df.rename(columns={"image_name": "file_name"}, inplace=True)
        else:
            raise RuntimeError(
                "Your df_to_check must have a 'file_name' column (or 'filename', 'image_name')"
            )

    # Unify to basename for consistent filename handling
    df["filename"] = df["file_name"].apply(lambda x: os.path.basename(str(x)))
    return df


def load_nonface_df(path: str) -> pd.DataFrame:
    """
    Read the animate-nonface master sheet with current labels.
    Ensures required columns exist with default values.
    """
    df = pd.read_excel(path)

    if "cocoId" not in df.columns:
        raise RuntimeError("Your nonface_xlsx must have a 'cocoId' column.")

    # Generate standardized filename from cocoId
    df["filename"] = df["cocoId"].apply(lambda x: f"{int(x):012d}.jpg")

    # Ensure label column exists with default value
    if "label" not in df.columns:
        df["label"] = "animate"

    return df


def load_faces_df(path: str) -> pd.DataFrame:
    """
    Read the faces master sheet with current labels.
    Face images are always labeled as 'face'.
    """
    df = pd.read_excel(path)

    if "cocoId" not in df.columns:
        raise RuntimeError("Your faces_xlsx must have a 'cocoId' column.")

    # Generate standardized filename from cocoId
    df["filename"] = df["cocoId"].apply(lambda x: f"{int(x):012d}.jpg")

    # Face images are always labeled as 'face'
    if "label" not in df.columns:
        df["label"] = "face"

    return df


def load_or_create_excel(path: str, essential_columns: List[str]) -> pd.DataFrame:
    """
    Loads an Excel file if it exists, or returns an empty DataFrame with essential columns.
    Handles missing files gracefully with appropriate user feedback.
    """
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)

            # Ensure all essential columns are present, add if missing
            for col in essential_columns:
                if col not in df.columns:
                    if col == "cocoId":
                        df[col] = 0  # Default for cocoId
                    elif col == "label":
                        df[col] = "unknown"  # Default for label
                    else:
                        df[col] = pd.NA  # Default for other columns

            # Return only essential columns in defined order
            return df[essential_columns]

        except Exception as e:
            st.error(f"Error reading Excel {path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=essential_columns)
    else:
        # Handle missing files with appropriate user feedback
        is_newly_labelled_file = any(
            term in path.lower() for term in ["labelled", "new"]
        )

        if is_newly_labelled_file:
            st.info(
                f"'{os.path.basename(path)}' not found. It will be created if you label images for this category."
            )
        else:
            st.warning(
                f"Required file '{os.path.basename(path)}' not found. App functionality may be limited."
            )

        return pd.DataFrame(columns=essential_columns)


@st.cache_data
def load_face_detection_results_filenames(
    path: str, target_detections: int = 0
) -> List[str]:
    """
    Load face detection results and return filenames that match the target detection count.
    By default, returns filenames with 0 detections.
    """
    filenames_with_target_detections = []

    try:
        with open(path, "r") as f:
            data = json.load(f)

        for item in data:
            # Validate item structure and detection count
            if (
                isinstance(item, dict)
                and "detection" in item
                and isinstance(item["detection"], list)
                and len(item["detection"]) == target_detections
                and "file_name" in item
            ):

                filenames_with_target_detections.append(
                    os.path.basename(str(item["file_name"]))
                )

    except FileNotFoundError:
        st.error(f"Error: Face detection results file not found at {path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {path}")
        return []

    # Return unique filenames
    return list(set(filenames_with_target_detections))


@st.cache_data
def load_labels(faces_xlsx: str, nonface_xlsx: str) -> Dict[str, str]:
    """
    Build a filename→label map from two Excel files.
    Face labels take precedence over non-face labels for duplicate filenames.
    """
    df_f = pd.read_excel(faces_xlsx)
    df_nf = pd.read_excel(nonface_xlsx)

    fmt = lambda x: f"{int(x):012d}.jpg"

    # Process faces dataframe
    if "cocoId" not in df_f.columns:
        raise RuntimeError(f"'cocoId' column missing in faces_xlsx: {faces_xlsx}")

    df_f["filename"] = df_f["cocoId"].apply(fmt)
    df_f["label"] = "face"  # These are definitively faces

    # Process non-faces dataframe
    if "cocoId" not in df_nf.columns:
        raise RuntimeError(f"'cocoId' column missing in nonface_xlsx: {nonface_xlsx}")

    df_nf["filename"] = df_nf["cocoId"].apply(fmt)

    # Handle label column for non-faces
    if "label" not in df_nf.columns:
        df_nf["label"] = "animate"  # Default for new non-face images
    else:
        # Convert 'Accepted' to 'animate', keep other existing labels
        df_nf["label"] = df_nf["label"].apply(
            lambda x: "animate" if x == "Accepted" else x
        )
        df_nf["label"] = df_nf["label"].fillna("animate")  # Fill NaN values

        # Ensure labels are within expected set, otherwise default to 'animate'
        known_nf_labels = [
            "animate",
            "animate_persons",
            "animate_animals",
            "animate_faces",
            "unknown",
        ]
        df_nf["label"] = df_nf["label"].apply(
            lambda x: x if x in known_nf_labels else "animate"
        )

    # Create label dictionary - faces take precedence over non-faces
    label_dict = dict(zip(df_nf["filename"], df_nf["label"]))
    label_dict.update(dict(zip(df_f["filename"], df_f["label"])))

    return label_dict


@st.cache_data
def load_initial_labels_from_all_sources(
    faces_final_xlsx: str,
    nonface_final_xlsx: str,
    faces_newly_labelled_xlsx: str,
    nonface_newly_labelled_xlsx: str,
    essential_cols_for_labels: List[str],
) -> Dict[str, str]:
    """
    Load labels from all sources with proper precedence order.
    Order: newly_labelled overrides final, faces override non-faces.
    """
    fmt = lambda x: f"{int(x):012d}.jpg" if pd.notna(x) and x != 0 else None

    # Load all dataframes
    df_f_final = load_or_create_excel(faces_final_xlsx, essential_cols_for_labels)
    df_nf_final = load_or_create_excel(nonface_final_xlsx, essential_cols_for_labels)
    df_f_new = load_or_create_excel(
        faces_newly_labelled_xlsx, essential_cols_for_labels
    )
    df_nf_new = load_or_create_excel(
        nonface_newly_labelled_xlsx, essential_cols_for_labels
    )

    label_dict = {}

    # Load in precedence order: final non-faces → final faces → new non-faces → new faces

    # 1. Final Non-Faces
    if "cocoId" in df_nf_final.columns and "label" in df_nf_final.columns:
        df_nf_final["filename"] = df_nf_final["cocoId"].apply(fmt)
        df_nf_final.dropna(subset=["filename"], inplace=True)
        label_dict.update(dict(zip(df_nf_final["filename"], df_nf_final["label"])))

    # 2. Final Faces
    if "cocoId" in df_f_final.columns:
        df_f_final["filename"] = df_f_final["cocoId"].apply(fmt)
        df_f_final.dropna(subset=["filename"], inplace=True)
        df_f_final["label"] = "face"  # Faces are always 'face'
        label_dict.update(dict(zip(df_f_final["filename"], df_f_final["label"])))

    # 3. Newly Labelled Non-Faces
    if "cocoId" in df_nf_new.columns and "label" in df_nf_new.columns:
        df_nf_new["filename"] = df_nf_new["cocoId"].apply(fmt)
        df_nf_new.dropna(subset=["filename"], inplace=True)
        label_dict.update(dict(zip(df_nf_new["filename"], df_nf_new["label"])))

    # 4. Newly Labelled Faces
    if "cocoId" in df_f_new.columns:
        df_f_new["filename"] = df_f_new["cocoId"].apply(fmt)
        df_f_new.dropna(subset=["filename"], inplace=True)
        df_f_new["label"] = "face"
        label_dict.update(dict(zip(df_f_new["filename"], df_f_new["label"])))

    # Return only valid entries
    return {k: v for k, v in label_dict.items() if pd.notna(k) and pd.notna(v)}


# =============================================================================
# DATA MANIPULATION FUNCTIONS
# =============================================================================


def save_df(df: pd.DataFrame, path: str):
    """
    Save DataFrame to Excel file, excluding 'unknown' labels and temporary columns.
    """
    # Filter out rows with label 'unknown'
    df_filtered = df[df["label"] != "unknown"]

    # Drop temporary 'filename' column if it exists
    if "filename" in df_filtered.columns:
        df_to_save = df_filtered.drop(columns=["filename"])
    else:
        df_to_save = df_filtered

    # Save to Excel
    df_to_save.to_excel(path, index=False)
    st.success(f"Saved updates to {os.path.basename(path)}")


def add_or_update_in_df(
    df: pd.DataFrame,
    record: Dict,
    filename_col: str = "filename",
    defined_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Add a new record to DataFrame or update existing record by filename.
    Ensures only defined columns are kept.
    """
    fn = record.get(filename_col)
    if not fn:
        st.error("Record to add/update is missing a filename.")
        return df if df is not None else pd.DataFrame(columns=defined_columns or [])

    # Ensure filename_col is in defined_columns
    if defined_columns and filename_col not in defined_columns:
        defined_columns = [filename_col] + [
            col for col in defined_columns if col != filename_col
        ]

    # Create DataFrame from new record with only defined columns
    new_row_df = pd.DataFrame([record])
    if defined_columns:
        for col in defined_columns:
            if col not in new_row_df.columns:
                if col == "cocoId":
                    new_row_df[col] = 0
                elif col == "label":
                    new_row_df[col] = "unknown"
                else:
                    new_row_df[col] = pd.NA
        new_row_df = new_row_df[defined_columns]

    if df is None or df.empty:
        return new_row_df

    # Remove existing entry for this filename
    df_filtered = df[df[filename_col] != fn]

    # Ensure existing df conforms to defined_columns before concatenation
    if defined_columns:
        for col in defined_columns:
            if col not in df_filtered.columns:
                default_val = (
                    0 if col == "cocoId" else ("unknown" if col == "label" else pd.NA)
                )
                df_filtered[col] = default_val
        df_filtered = df_filtered[defined_columns]

    return pd.concat([df_filtered, new_row_df], ignore_index=True)


# =============================================================================
# IMAGE HANDLING FUNCTIONS
# =============================================================================


@st.cache_data
def load_and_border_image(path: str, color: str, border_width: int) -> Image.Image:
    """
    Load an image and add a colored border. Creates placeholder if image not found.
    """
    try:
        img = Image.open(path)
        return ImageOps.expand(img, border=border_width, fill=color)
    except FileNotFoundError:
        # Create placeholder image if not found
        placeholder = Image.new("RGB", (200, 200), color="lightgray")
        st.warning(f"Image not found: {path}. Displaying placeholder.")
        return ImageOps.expand(placeholder, border=border_width, fill=color)


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main Streamlit application for image labeling."""

    # Page configuration
    st.set_page_config(page_title="NSD Image Browser & Label")
    st.title("🖼️ NSD Image Browser & Label")

    # Initialize session state
    if "labels" not in st.session_state:
        st.session_state.labels = {}
    if "df_nonface" not in st.session_state:
        st.session_state.df_nonface = pd.DataFrame()

    # Load configuration and set paths
    config = load_config("config.yaml")
    subdir = "subj_01"  # TODO: Make this configurable

    # Define file paths
    df_to_check_path_faces = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_face_unchecked,
    )

    df_to_check_path_animate = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_non_face_unchecked,
    )

    faces_xlsx_final = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_face_final,
    )

    nonface_xlsx_final = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_non_face_final,
    )

    faces_xlsx_new_labelled_path = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_face_labelled,
    )

    nonface_xlsx_new_labelled_path = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_non_face_labelled,
    )

    images_dir = config.directories.images_target_dir

    # Sidebar controls
    st.sidebar.header("Dataset Mode")
    dataset_mode = st.sidebar.selectbox(
        "Choose image set to label:", ["animate", "faces"], key="dataset_mode_select"
    )

    st.sidebar.header("Display Settings")
    n_cols = st.sidebar.slider("Columns", 1, 10, 2, key="n_cols_slider")
    images_per_page = st.sidebar.slider(
        "Images per page", 1, 500, 26, key="imgs_per_page_slider"
    )
    border_width = st.sidebar.slider(
        "Border width", 1, 20, 12, key="border_width_slider"
    )

    # Load data based on selected mode
    if dataset_mode == "faces":
        df_master_list = load_df_to_check(df_to_check_path_faces)
        df_full = df_master_list.copy()
        st.sidebar.info(
            "Showing images with 1 face detection from `face_detection_result.json`."
        )
    else:  # "animate" mode (default)
        df_master_list = load_df_to_check(df_to_check_path_animate)
        st.sidebar.info("Showing all images from the 'animate non-face' master sheet.")
        df_full = df_master_list.copy()

    # Load labels and dataframes (once per session)
    if not st.session_state.labels or "initial_load_done" not in st.session_state:
        st.session_state.labels = load_labels(faces_xlsx_final, nonface_xlsx_final)
        st.session_state.df_nonface = load_nonface_df(nonface_xlsx_final)
        st.session_state.df_faces = load_faces_df(faces_xlsx_final)
        st.session_state.initial_load_done = True

    # Use labels from session state for dynamic updates
    current_labels = st.session_state.labels
    df_nonface_to_update = st.session_state.df_nonface
    df_faces_to_update = st.session_state.df_faces

    # Pagination setup
    total = len(df_full)
    if images_per_page == 0:
        images_per_page = 1  # Avoid division by zero

    total_pages = math.ceil(total / images_per_page) if total > 0 else 1
    page_num_key = f"page_num_input_{dataset_mode}"  # Unique key per mode
    page = st.sidebar.number_input("Page", 1, total_pages, 1, key=page_num_key)

    start, end = (page - 1) * images_per_page, page * images_per_page
    subset = df_full.iloc[start:end]

    # Display pagination info
    if not df_full.empty:
        st.write(
            f"Showing {start+1}–{min(end, total)} of {total} images (page {page}/{total_pages}) for mode: **{dataset_mode}**"
        )
    else:
        st.write(f"Mode: **{dataset_mode}**. No images to display.")

    # Define label colors and options
    label_colors = {
        "face": "#1f77b4",
        "animate": "#ff7f0e",
        "animate_persons": "#2ca02c",
        "animate_animals": "#d62728",
        "animate_faces": "#9467bd",
        "unknown": "gray",
    }
    options = [
        "face",
        "animate",
        "animate_persons",
        "animate_animals",
        "animate_faces",
        "unknown",
    ]

    # Display label counts in sidebar
    st.sidebar.header("Label Status")
    if current_labels and not df_full.empty:
        relevant_filenames = set(df_full["filename"])
        labels_in_view = {
            fn: lbl for fn, lbl in current_labels.items() if fn in relevant_filenames
        }

        if labels_in_view:
            label_counts = Counter(labels_in_view.values())
            st.sidebar.subheader(f"Counts for current set ({dataset_mode}):")
            for label, count in sorted(label_counts.items()):
                st.sidebar.write(f"- {label}: {count}")
        else:
            st.sidebar.write("No labels applicable to the current filtered set.")
    elif not df_full.empty:
        st.sidebar.write("Labels not loaded or empty.")
    else:
        st.sidebar.write("No images in current view to count labels for.")

    # Render image grid with labeling interface
    if subset.empty:
        st.info("No images to display for the current selection and page.")
    else:
        # Create rows of images based on column count
        rows = [
            subset["filename"].tolist()[i : i + n_cols]
            for i in range(0, len(subset), n_cols)
        ]

        for row_idx, row_fns in enumerate(rows):
            cols = st.columns(len(row_fns))

            for col_idx, fn in enumerate(row_fns):
                img_path = os.path.join(images_dir, fn)

                # Create unique widget key to avoid conflicts
                widget_key = f"lbl_{fn}_{page}_{row_idx}_{col_idx}"

                # Get current label from session state
                initial_label = current_labels.get(fn, "unknown")
                if initial_label not in options:  # Safety check
                    initial_label = "unknown"

                current_label_idx = options.index(initial_label)

                # Create selectbox for label selection
                choice = cols[col_idx].selectbox(
                    f"Label ({fn})", options, index=current_label_idx, key=widget_key
                )

                display_label_for_border = initial_label

                # Handle label changes
                if choice != initial_label:
                    # Update session state
                    st.session_state.labels[fn] = choice
                    display_label_for_border = choice

                    # Determine which dataframe and paths to use
                    if dataset_mode == "animate":
                        df_update = df_nonface_to_update
                        df_source = pd.read_excel(df_to_check_path_animate)
                        save_path = nonface_xlsx_new_labelled_path
                        session_key = "df_nonface"
                    elif dataset_mode == "faces":
                        df_update = df_faces_to_update
                        df_source = pd.read_excel(df_to_check_path_faces)
                        save_path = faces_xlsx_new_labelled_path
                        session_key = "df_faces"

                    # Build boolean mask by comparing basenames
                    mask = df_update["file_name"].apply(os.path.basename) == fn

                    # Get source rows by basename check
                    new_rows = df_source.loc[
                        df_source["file_name"].apply(os.path.basename) == fn
                    ].copy()

                    if mask.any():
                        # Update existing entry
                        df_update.loc[mask, "label"] = choice
                    else:
                        # Add new entry
                        if new_rows.empty:
                            raise KeyError(
                                f"No entry for filename {fn} in source DataFrame"
                            )
                        new_rows["label"] = choice
                        df_update = pd.concat([df_update, new_rows], ignore_index=True)

                    # Save and update session state
                    save_df(df_update, save_path)
                    st.session_state[session_key] = df_update.copy()

                    # Update local variables for consistency
                    if dataset_mode == "animate":
                        df_nonface_to_update = df_update
                    else:
                        df_faces_to_update = df_update

                    # Rerun to reflect changes immediately
                    st.rerun()

                # Display image with colored border
                color = label_colors.get(display_label_for_border, "gray")
                try:
                    img_bordered = load_and_border_image(img_path, color, border_width)
                    cols[col_idx].image(
                        img_bordered,
                        caption=f"{fn} - {display_label_for_border}",
                        use_container_width=True,
                    )
                except Exception as e:
                    cols[col_idx].error(f"Error loading {fn}: {e}")


if __name__ == "__main__":
    main()
