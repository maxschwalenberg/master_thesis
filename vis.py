

import os
import math
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
from utils.config import load_config
import json
from collections import Counter # For counting labels


# ---------------------------------------------------------







@st.cache_data
def load_df_to_check(path: str) -> pd.DataFrame:
    """Read the sheet of all images you need to label."""
    df = pd.read_excel(path)
    if 'file_name' not in df.columns:
        # Try to find a similar column if 'file_name' is missing
        if 'filename' in df.columns:
            df.rename(columns={'filename': 'file_name'}, inplace=True)
        elif 'image_name' in df.columns:
            df.rename(columns={'image_name': 'file_name'}, inplace=True)
        else:
            raise RuntimeError("Your df_to_check must have a 'file_name' column (or 'filename', 'image_name')")
    # unify to basename
    df['filename'] = df['file_name'].apply(lambda x: os.path.basename(str(x)))
    return df

def load_nonface_df(path: str) -> pd.DataFrame:
    """Read the animate‐nonface master sheet (with its current labels)."""
    df = pd.read_excel(path)
    if 'cocoId' not in df.columns:
        raise RuntimeError("Your nonface_xlsx must have a 'cocoId' column.")
    df['filename'] = df['cocoId'].apply(lambda x: f"{int(x):012d}.jpg")
    if 'label' not in df.columns: # Ensure label column exists
        df['label'] = 'animate'
    return df

def load_faces_df(path: str) -> pd.DataFrame:
    """Read the faces master sheet (with its current labels)."""
    df = pd.read_excel(path)
    return df

def save_df(df: pd.DataFrame, path: str):
    """Overwrite the animate‐nonface sheet in place."""
    df_to_save = df.drop(columns=['filename'])
   
   
    df_to_save.to_excel(path, index=False)
    st.success(f"Saved updates to {os.path.basename(path)}")


@st.cache_data
def load_labels(faces_xlsx: str, nonface_xlsx: str) -> dict:
    """Build a filename→label map from your two Excel files."""
    df_f = pd.read_excel(faces_xlsx)
    df_nf = pd.read_excel(nonface_xlsx)

    fmt = lambda x: f"{int(x):012d}.jpg"

    if 'cocoId' not in df_f.columns:
        raise RuntimeError(f"'cocoId' column missing in faces_xlsx: {faces_xlsx}")
    df_f['filename'] = df_f['cocoId'].apply(fmt)
    df_f['label'] = 'face'  # These are definitively faces

    if 'cocoId' not in df_nf.columns:
        raise RuntimeError(f"'cocoId' column missing in nonface_xlsx: {nonface_xlsx}")
    df_nf['filename'] = df_nf['cocoId'].apply(fmt)

    if 'label' not in df_nf.columns:
        df_nf['label'] = 'animate'  # Default for new non-face images
    else:
        # Convert 'Accepted' to 'animate', keep other existing labels
        df_nf['label'] = df_nf['label'].apply(lambda x: 'animate' if x == 'Accepted' else x)
        # Fill any potential NaN values with 'animate' if 'label' column existed
        df_nf['label'] = df_nf['label'].fillna('animate')
        # Ensure labels are within the expected set, otherwise default to 'animate'
        known_nf_labels = ['animate', 'animate_persons', 'animate_animals', 'animate_faces', 'unknown']
        df_nf['label'] = df_nf['label'].apply(lambda x: x if x in known_nf_labels else 'animate')


    # Create the label dictionary.
    # Start with non-face, then update with faces, so 'face' takes precedence if a filename is in both.
    label_dict = dict(zip(df_nf['filename'], df_nf['label']))
    label_dict.update(dict(zip(df_f['filename'], df_f['label'])))

    return label_dict

@st.cache_data
def load_and_border_image(path: str, color: str, border_width: int) -> Image.Image:
    try:
        img = Image.open(path)
        return ImageOps.expand(img, border=border_width, fill=color)
    except FileNotFoundError:
        # Create a placeholder image if not found
        placeholder = Image.new('RGB', (200, 200), color='lightgray')
        # draw = ImageDraw.Draw(placeholder) # If you want to draw text
        # draw.text((10,10), "Not Found", fill="black")
        st.warning(f"Image not found: {path}. Displaying placeholder.")
        return ImageOps.expand(placeholder, border=border_width, fill=color)


@st.cache_data
def load_face_detection_results_filenames(path: str, target_detections: int = 0) -> list:
    """
    Load face detection results and return filenames that match the target_detections count.
    By default, it returns filenames with 0 detections.
    """
    filenames_with_target_detections = []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        for item in data:
            # Ensure item is a dictionary and has 'detection' and 'file_name' keys
            if isinstance(item, dict) and \
               "detection" in item and \
               isinstance(item["detection"], list) and \
               len(item["detection"]) == target_detections and \
               "file_name" in item:
                filenames_with_target_detections.append(os.path.basename(str(item["file_name"])))
            # elif isinstance(item, dict) and "file_name" in item: # for debugging
            #     print(f"Skipping item (detection wrong or missing): {item['file_name']}, detection: {item.get('detection')}")

    except FileNotFoundError:
        st.error(f"Error: Face detection results file not found at {path}")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {path}")
        return []
    return list(set(filenames_with_target_detections)) # Return unique filenames



# Add this with your other helper functions
def load_or_create_excel(path: str, essential_columns: list) -> pd.DataFrame:
    """Loads an excel file if it exists, or returns an empty DataFrame with essential_columns."""
    if os.path.exists(path):
        try:
            df = pd.read_excel(path)
            # Ensure all essential columns are present, add if missing
            for col in essential_columns:
                if col not in df.columns:
                    if col == 'cocoId': df[col] = 0 # Default for cocoId
                    elif col == 'label': df[col] = 'unknown' # Default for label
                    else: df[col] = pd.NA # Default for other text/data columns
            return df[essential_columns] # Return only essential columns in defined order
        except Exception as e:
            st.error(f"Error reading Excel {path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame(columns=essential_columns)
    else:
        # For 'newly_labelled' files, it's okay if they don't exist yet.
        is_newly_labelled_file = any(term in path.lower() for term in ["labelled", "new"])
        if is_newly_labelled_file:
            st.info(f"'{os.path.basename(path)}' not found. It will be created if you label images for this category.")
        else: # For final or unchecked files, non-existence might be a setup issue
            st.warning(f"Required file '{os.path.basename(path)}' not found. App functionality may be limited.")
        return pd.DataFrame(columns=essential_columns)



# Add this with your other helper functions
def add_or_update_in_df(df: pd.DataFrame, record: dict, filename_col: str = 'filename',
                         defined_columns: list = None) -> pd.DataFrame:
    """
    Adds a new record to the DataFrame or updates it if the filename already exists.
    Ensures only defined_columns are kept.
    """
    fn = record.get(filename_col)
    if not fn:
        st.error("Record to add/update is missing a filename.")
        return df if df is not None else pd.DataFrame(columns=defined_columns or [])

    # Ensure defined_columns has filename_col
    if defined_columns and filename_col not in defined_columns:
        # Prepend filename_col if not already there to avoid issues
        defined_columns = [filename_col] + [col for col in defined_columns if col != filename_col]


    # Create a DataFrame from the new record, ensuring it only has defined_columns
    new_row_df = pd.DataFrame([record])
    if defined_columns:
        for col in defined_columns: # Ensure all defined_columns exist in new_row_df
            if col not in new_row_df.columns:
                if col == 'cocoId': new_row_df[col] = 0
                elif col == 'label': new_row_df[col] = 'unknown'
                else: new_row_df[col] = pd.NA
        new_row_df = new_row_df[defined_columns] # Select and order columns

    if df is None or df.empty:
        return new_row_df

    # Remove existing entry for this filename, if any
    df_filtered = df[df[filename_col] != fn]
    
    # Concatenate, ensuring consistent columns based on defined_columns
    if defined_columns:
        # Ensure existing df also conforms to defined_columns before concat
        for col in defined_columns:
            if col not in df_filtered.columns:
                df_filtered[col] = pd.NA if col not in ['cocoId', 'label'] else (0 if col == 'cocoId' else 'unknown')
        df_filtered = df_filtered[defined_columns]
    
    return pd.concat([df_filtered, new_row_df], ignore_index=True)



@st.cache_data # Caching this initial consolidated label dictionary is fine
def load_initial_labels_from_all_sources(
    faces_final_xlsx: str, nonface_final_xlsx: str,
    faces_newly_labelled_xlsx: str, nonface_newly_labelled_xlsx: str,
    essential_cols_for_labels: list
) -> dict:
    fmt = lambda x: f"{int(x):012d}.jpg" if pd.notna(x) and x !=0 else None # Handle potential NaN/0 cocoId

    df_f_final = load_or_create_excel(faces_final_xlsx, essential_cols_for_labels)
    df_nf_final = load_or_create_excel(nonface_final_xlsx, essential_cols_for_labels)
    df_f_new = load_or_create_excel(faces_newly_labelled_xlsx, essential_cols_for_labels)
    df_nf_new = load_or_create_excel(nonface_newly_labelled_xlsx, essential_cols_for_labels)

    label_dict = {}

    # Order of loading determines precedence: newly_labelled overrides final, faces override non-faces.
    # 1. Final Non-Faces
    if 'cocoId' in df_nf_final.columns and 'label' in df_nf_final.columns:
        df_nf_final['filename'] = df_nf_final['cocoId'].apply(fmt)
        df_nf_final.dropna(subset=['filename'], inplace=True) # Remove rows where filename couldn't be formed
        label_dict.update(dict(zip(df_nf_final['filename'], df_nf_final['label'])))

    # 2. Final Faces
    if 'cocoId' in df_f_final.columns:
        df_f_final['filename'] = df_f_final['cocoId'].apply(fmt)
        df_f_final.dropna(subset=['filename'], inplace=True)
        df_f_final['label'] = 'face' # Faces are always 'face'
        label_dict.update(dict(zip(df_f_final['filename'], df_f_final['label'])))

    # 3. Newly Labelled Non-Faces
    if 'cocoId' in df_nf_new.columns and 'label' in df_nf_new.columns:
        df_nf_new['filename'] = df_nf_new['cocoId'].apply(fmt)
        df_nf_new.dropna(subset=['filename'], inplace=True)
        label_dict.update(dict(zip(df_nf_new['filename'], df_nf_new['label'])))

    # 4. Newly Labelled Faces
    if 'cocoId' in df_f_new.columns:
        df_f_new['filename'] = df_f_new['cocoId'].apply(fmt)
        df_f_new.dropna(subset=['filename'], inplace=True)
        df_f_new['label'] = 'face'
        label_dict.update(dict(zip(df_f_new['filename'], df_f_new['label'])))
    
    return {k: v for k, v in label_dict.items() if pd.notna(k) and pd.notna(v)}

# ---------------------------------------------------------

def main():
    st.set_page_config(page_title="NSD Image Browser & Label")
    st.title("🖼️ NSD Image Browser & Label")

    # --- 0) Initialize session state for labels if not present ---
    if 'labels' not in st.session_state:
        st.session_state.labels = {}
    if 'df_nonface' not in st.session_state:
        st.session_state.df_nonface = pd.DataFrame()


    # --- 1) Paths from your config.yaml ------------------------
    config = load_config("config.yaml")
    subdir = "subj_02"  # adjust as needed
    
    # Main list of images to potentially check
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

    # faces_xlsx_new_labelled_path = os.path.join(
    #     config.directories.excel_files_target_dir,
    #     subdir,
    #     "faces",
    #     config.dataset_creation.subset_animate_face_labelled,
    # )
    # nonface_xlsx_new_labelled_path = os.path.join(
    #     config.directories.excel_files_target_dir,
    #     subdir,
    #     "animate_nonface",
    #     config.dataset_creation.subset_animate_non_face_labelled,
    # )

    faces_xlsx_new_labelled_path = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_face_labelled
        # "faces",
        # "test.xlsx",
    )
    nonface_xlsx_new_labelled_path = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_non_face_labelled
        
        # "animate_nonface",
        # "test.xlsx",
    )




    images_dir = config.directories.images_target_dir
    

    # --- 2) Sidebar controls ----------------------------------
    st.sidebar.header("Dataset Mode")
    dataset_mode = st.sidebar.selectbox(
        "Choose image set to label:",
        ["animate", "faces"],
        key="dataset_mode_select"
    )

    st.sidebar.header("Display Settings")
    n_cols = st.sidebar.slider("Columns", 1, 10, 2, key="n_cols_slider")
    images_per_page = st.sidebar.slider("Images per page", 1, 500, 26, key="imgs_per_page_slider") # Increased max
    border_width = st.sidebar.slider("Border width", 1, 20, 12, key="border_width_slider")

    # --- 3) Load data based on mode ------------------------------------------
    # Load the master list of images first

    if dataset_mode == "faces":
        df_master_list = load_df_to_check(df_to_check_path_faces)
        df_full = df_master_list.copy()
        st.sidebar.info("Showing images with 1 face detections from `face_detection_result.json`.")
        


    else: # "animate" mode (default)
        df_master_list = load_df_to_check(df_to_check_path_animate)
        st.sidebar.info("Showing all images from the 'animate non-face' master sheet.")
        df_full = df_master_list.copy()


    # Load labels and non-face dataframe (for saving)
    # These are loaded once and then potentially modified in session_state
    if not st.session_state.labels or 'initial_load_done' not in st.session_state:
        st.session_state.labels = load_labels(faces_xlsx_final, nonface_xlsx_final)
        st.session_state.df_nonface = load_nonface_df(nonface_xlsx_final)
        st.session_state.df_faces = load_faces_df(faces_xlsx_final)

        st.session_state.initial_load_done = True # Mark that initial load happened

    # Use labels from session state for dynamic updates
    current_labels = st.session_state.labels
    df_nonface_to_update = st.session_state.df_nonface
    df_faces_to_update = st.session_state.df_faces

    # --- 4) Pagination ----------------------------------------
    total = len(df_full)
    if images_per_page == 0: images_per_page = 1 # Avoid division by zero
    total_pages = math.ceil(total / images_per_page) if total > 0 else 1
    
    page_num_key = f"page_num_input_{dataset_mode}" # Ensure page number is unique per mode
    page = st.sidebar.number_input("Page", 1, total_pages, 1, key=page_num_key)
    
    start, end = (page - 1) * images_per_page, page * images_per_page
    subset = df_full.iloc[start:end]

    if not df_full.empty:
        st.write(f"Showing {start+1}–{min(end, total)} of {total} images (page {page}/{total_pages}) for mode: **{dataset_mode}**")
    else:
        st.write(f"Mode: **{dataset_mode}**. No images to display.")


    # --- 5) Colors & Options ---------------------------------
    label_colors = {
        'face': '#1f77b4',
        'animate': '#ff7f0e',  # fallback
        'animate_persons': '#2ca02c',
        'animate_animals': '#d62728',
        'animate_faces': '#9467bd', # Changed color for distinction
        'unknown': 'gray',
    }
    options = ['face', 'animate', 'animate_persons', 'animate_animals', 'animate_faces', "unknown"]

    # --- 6) Label Counts / Status Info -------------------------
    st.sidebar.header("Label Status")
    if current_labels and not df_full.empty:
        # Consider only labels for images currently in df_full (filtered set)
        relevant_filenames = set(df_full['filename'])
        labels_in_view = {fn: lbl for fn, lbl in current_labels.items() if fn in relevant_filenames}
        if labels_in_view:
            label_counts = Counter(labels_in_view.values())
            st.sidebar.subheader(f"Counts for current set ({dataset_mode}):")
            for label, count in sorted(label_counts.items()):
                st.sidebar.write(f"- {label}: {count}")
        else:
            st.sidebar.write("No labels applicable to the current filtered set.")

    elif not df_full.empty :
         st.sidebar.write("Labels not loaded or empty.")
    else:
        st.sidebar.write("No images in current view to count labels for.")


    # --- 7) Render grid with relabeling ----------------------
    if subset.empty:
        st.info("No images to display for the current selection and page.")
    else:
        rows = [
            subset['filename'].tolist()[i: i + n_cols]
            for i in range(0, len(subset), n_cols)
        ]
        for row_idx, row_fns in enumerate(rows):
            cols = st.columns(len(row_fns))
            for col_idx, fn in enumerate(row_fns):
                img_path = os.path.join(images_dir, fn)
                
                # Use a unique key for each selectbox based on filename and row/col to avoid conflicts
                widget_key = f"lbl_{fn}_{page}_{row_idx}_{col_idx}"

                # Get current label for the image from our session state
                initial_label = current_labels.get(fn, 'unknown')
                if initial_label not in options: # Safety net for unexpected labels
                    initial_label = 'unknown'
                
                current_label_idx = options.index(initial_label)

                choice = cols[col_idx].selectbox(
                    f"Label ({fn})", # Display filename in label for clarity
                    options,
                    index=current_label_idx,
                    key=widget_key
                )
                
                display_label_for_border = initial_label # Default to initial for border

                if choice != initial_label:
                    # Update the label in our session state dictionary
                    st.session_state.labels[fn] = choice
                    display_label_for_border = choice # Update for immediate border color change

                    if dataset_mode == "animate":
                        df_update   = df_nonface_to_update
                        df_source   = pd.read_excel(df_to_check_path_animate)
                        save_path   = nonface_xlsx_new_labelled_path
                        session_key = "df_nonface"

                    elif dataset_mode == "faces":
                        df_update   = df_faces_to_update
                        df_source   = pd.read_excel(df_to_check_path_faces)
                        save_path   = faces_xlsx_new_labelled_path
                        session_key = "df_faces"

                    # # 1) find existing rows
                    # mask = df_update['file_name'] == os.path.join("train2017", fn)

                    # if mask.any():
                    #     # just update the label
                    #     df_update.loc[mask, 'label'] = choice
                    # else:
                    #     # pull the full row(s) from the unchecked df
                    #     new_rows = df_source.loc[df_source['file_name'] == os.path.join("train2017", fn)].copy()
                    #     if new_rows.empty:
                    #         raise KeyError(f"No entry for filename {fn} in source DataFrame")
                    #     # add your new label
                    #     new_rows['label'] = choice
                    #     # append it (preserves all original columns + label)
                    #     df_update = pd.concat([df_update, new_rows], ignore_index=True)

                    # build a boolean mask by comparing basenames only
                    mask = df_update['file_name'].apply(os.path.basename) == fn

                    # grab the source‐rows by the same basename check
                    new_rows = (
                        df_source
                        .loc[df_source['file_name'].apply(os.path.basename) == fn]
                        .copy()
                    )

                    if mask.any():
                        df_update.loc[mask, 'label'] = choice
                    else:
                        if new_rows.empty:
                            raise KeyError(f"No entry for filename {fn} in source DataFrame")
                        new_rows['label'] = choice
                        df_update = pd.concat([df_update, new_rows], ignore_index=True)

                    # persist & update session state
                    save_df(df_update, save_path)
                    st.session_state[session_key] = df_update.copy()

                    # if you need the originals back in those variables:
                    if dataset_mode == "animate":
                        df_nonface_to_update = df_update
                    else:
                        df_faces_to_update   = df_update




                    # Rerun to reflect the change immediately, especially for label counts and border
                    st.rerun()
                
                color = label_colors.get(display_label_for_border, 'gray')
                try:
                    img_bordered = load_and_border_image(img_path, color, border_width)
                    cols[col_idx].image(img_bordered, caption=f"{fn} - {display_label_for_border}", use_container_width=True)
                except Exception as e:
                    cols[col_idx].error(f"Error loading {fn}: {e}")

if __name__ == "__main__":
    main()






