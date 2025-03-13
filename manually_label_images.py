# import pandas as pd
# import cv2
# from utils.config import Configuration, load_config
# import os


# def label_images(config: Configuration, df_to_check_path: str):

#     print(
#         """Instructions:

#           Press -ENTER- to accept the sample to the final set, anything else to skip."""
#     )

#     # Read the CSV file
#     df = pd.read_excel(df_to_check_path)

#     # Create an empty dataframe to store labeled images
#     labeled_df = pd.DataFrame(columns=df.columns)

#     # Loop through each row in the dataframe
#     for index, row in df.iterrows():
#         # Read the image
#         image_filename = os.path.basename(row["file_name"])
#         img_path = os.path.join(config.images_target_dir, image_filename)
#         image = cv2.imread(img_path)

#         if image is None:
#             print(f"Could not read image: {img_path}")
#             continue

#         # Create window and set it to normal size (not fullscreen)
#         cv2.namedWindow("Image Labeler", cv2.WINDOW_NORMAL)

#         # Resize window to a reasonable size while maintaining aspect ratio
#         height, width = image.shape[:2]
#         max_display_size = 800
#         if max(height, width) > max_display_size:
#             scaling = max_display_size / max(height, width)
#             image = cv2.resize(
#                 image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
#             )

#         # Show the image
#         cv2.imshow("Image Labeler", image)

#         # Wait for key press
#         key = cv2.waitKey(0)

#         # If Enter key is pressed (13 is Enter key code)
#         if key == 13:
#             # Append the current row to labeled dataframe
#             labeled_df = pd.concat([labeled_df, pd.DataFrame([row])], ignore_index=True)
#             print(f"Labeled: {img_path}")

#         # Close the image window
#         cv2.destroyAllWindows()

#     # Return the labeled dataframe
#     return labeled_df


# if __name__ == "__main__":
#     config = load_config("config.yaml")

#     # Example usage
#     df_to_check_path = os.path.join(
#         config.excel_files_target_dir, "animate_non_face_unchecked.xlsx"
#     )

#     labeled_images = label_images(config, df_to_check_path)
#     labeled_images = labeled_images.loc[
#         :, ~labeled_images.columns.str.contains("Unnamed", case=False, na=False)
#     ]

#     labeled_images.to_excel(
#         os.path.join(config.excel_files_target_dir, "animate_non_face.xlsx"),
#         index=False,
#     )


import streamlit as st
import pandas as pd
import cv2
from utils.config import Configuration, load_config
import os
import streamlit.components.v1 as components

try:
    from streamlit_javascript import st_javascript
except ImportError:
    st.error(
        "Please install streamlit_javascript via pip: pip install streamlit_javascript"
    )
    st.stop()


def save_labels(df, target_path):
    """Save current labels to the Excel file."""
    df_copy = df.copy()
    df_copy["label"] = st.session_state.labels

    df_filtered = df_copy[df_copy["label"] == "Accepted"]

    base_path = os.path.dirname(target_path)
    base, ext = os.path.splitext(os.path.basename(target_path))
    filtered_target_path = os.path.join(base_path, f"{base}_filtered{ext}")

    df_copy.to_excel(target_path, index=False)
    df_filtered.to_excel(filtered_target_path, index=False)

    st.success(f"Labeled data saved to {target_path}")


def label_images(config: Configuration, df_to_check_path: str, target_path: str):
    # Read the Excel file with the full image metadata.
    df = pd.read_excel(df_to_check_path)

    # --- Initialize session state ---
    if os.path.exists(target_path):
        # If a labeled file exists, load its labels if they match the master file.
        labelled_df = pd.read_excel(target_path)
        if "label" in labelled_df.columns and len(labelled_df) == len(df):
            if "labels" not in st.session_state:
                st.session_state.labels = labelled_df["label"].tolist()
            if "current_index" not in st.session_state:
                # Set current_index to the first unlabelled sample.
                for i, lab in enumerate(st.session_state.labels):
                    if pd.isna(lab) or lab is None:
                        st.session_state.current_index = i
                        break
                else:
                    st.session_state.current_index = len(df)
        else:
            if "labels" not in st.session_state:
                st.session_state.labels = [None] * len(df)
            if "current_index" not in st.session_state:
                st.session_state.current_index = 0
    else:
        if "labels" not in st.session_state:
            st.session_state.labels = [None] * len(df)
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0

    # If all images have been processed, show summary and save option.
    if st.session_state.current_index >= len(df):
        st.write("All images processed!")
        df["label"] = st.session_state.labels
        df_filtered = df[df["label"] == "Accepted"]

        base_path = os.path.dirname(target_path)
        base, ext = os.path.splitext(os.path.basename(target_path))
        filtered_target_path = os.path.join(base_path, f"{base}_filtered{ext}")

        df_filtered.to_excel(filtered_target_path, index=False)

        st.write(df)
        if st.button("Save Labeled Data"):
            df.to_excel(target_path, index=False)
            st.success(f"Labeled data saved to {target_path}")
        return

    st.title("Image Labeler")
    st.write(f"Image {st.session_state.current_index + 1} of {len(df)}")

    # --- Capture key events using st_javascript ---
    # This returns the keyCode when a key is pressed.
    key_code = st_javascript(
        """
        new Promise(resolve => {
            document.addEventListener('keydown', event => {
                // Prevent default behavior (like backspace navigation).
                event.preventDefault();
                resolve(event.keyCode);
            }, {once: true});
        });
        """
    )

    if key_code is not None:
        if key_code == 32:  # Spacebar pressed.
            st.session_state.labels[st.session_state.current_index] = "Accepted"
            save_labels(df, target_path)
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
            st.rerun()
        elif key_code == 8:  # Backspace pressed.
            st.session_state.labels[st.session_state.current_index] = "Skipped"
            save_labels(df, target_path)
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
            st.rerun()

    # --- Manual navigation buttons ---
    if st.button("Previous"):
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()

    # --- Show current label status with color ---
    current_label = st.session_state.labels[st.session_state.current_index]
    if current_label is None or pd.isna(current_label):
        st.markdown(
            "<span style='color: blue;'>Unlabelled</span>", unsafe_allow_html=True
        )
    elif current_label == "Accepted":
        st.markdown(
            "<span style='color: green;'>Accepted</span>", unsafe_allow_html=True
        )
    elif current_label == "Skipped":
        st.markdown("<span style='color: red;'>Skipped</span>", unsafe_allow_html=True)
    else:
        st.write("Unknown label")

    # --- Load and display the current image ---
    row = df.iloc[st.session_state.current_index]
    image_filename = os.path.basename(row["file_name"])
    img_path = os.path.join(config.directories.images_target_dir, image_filename)
    image = cv2.imread(img_path)
    if image is None:
        st.write(f"Could not read image: {img_path}")
        st.session_state.labels[st.session_state.current_index] = "Skipped"
        save_labels(df, target_path)
        if st.session_state.current_index < len(df) - 1:
            st.session_state.current_index += 1
        st.rerun()

    height, width = image.shape[:2]
    max_display_size = 800
    if max(height, width) > max_display_size:
        scaling = max_display_size / max(height, width)
        image = cv2.resize(
            image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption=image_filename)
    st.write("Press **Spacebar** to Accept, **Backspace** to Skip.")

    # --- Fallback manual buttons ---
    col_accept, col_skip = st.columns(2)
    with col_accept:
        if st.button("Accept"):
            st.session_state.labels[st.session_state.current_index] = "Accepted"
            save_labels(df, target_path)
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
            st.rerun()
    with col_skip:
        if st.button("Skip"):
            st.session_state.labels[st.session_state.current_index] = "Skipped"
            save_labels(df, target_path)
            if st.session_state.current_index < len(df) - 1:
                st.session_state.current_index += 1
            st.rerun()


if __name__ == "__main__":
    faces_set = False

    # Adjust these paths and configuration as needed.
    subdir = "subj_02"
    st.title("Image Labeler")
    config = load_config(
        "config.yaml"
    )  # Ensure your load_config() function is defined and accessible.

    if faces_set:
        df_to_check_path = os.path.join(
            config.directories.excel_files_target_dir,
            subdir,
            config.dataset_creation.subset_animate_face_nans_removed,
        )
        target_path = os.path.join(
            config.directories.excel_files_target_dir,
            subdir,
            config.dataset_creation.subset_animate_face_labelled,
        )
    else:
        df_to_check_path = os.path.join(
            config.directories.excel_files_target_dir,
            subdir,
            config.dataset_creation.subset_animate_non_face_nans_removed,
        )
        target_path = os.path.join(
            config.directories.excel_files_target_dir,
            subdir,
            config.dataset_creation.subset_animate_non_face_labelled,
        )

    label_images(config, df_to_check_path, target_path)
