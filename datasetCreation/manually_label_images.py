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


def label_images(config: Configuration, df_to_check_path: str, target_path: str):
    # Read the Excel file with image metadata
    print(df_to_check_path)
    df = pd.read_excel(df_to_check_path)

    # Initialize session state variables if not already set
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "labeled_rows" not in st.session_state:
        st.session_state.labeled_rows = []

    # If all images have been processed, show completion and option to save results
    if st.session_state.current_index >= len(df):
        st.write("All images processed!")
        labeled_df = pd.DataFrame(st.session_state.labeled_rows)
        st.write(labeled_df)
        if st.button("Save Labeled Data"):

            labeled_df.to_excel(target_path, index=False)
            st.success(f"Labeled data saved to {target_path}")
        return

    # Get the current row and load the corresponding image
    row = df.iloc[st.session_state.current_index]
    image_filename = os.path.basename(row["file_name"])
    img_path = os.path.join(config.images_target_dir, image_filename)
    image = cv2.imread(img_path)

    # If image cannot be read, skip it and rerun the app
    if image is None:
        st.write(f"Could not read image: {img_path}")
        st.session_state.current_index += 1
        st.experimental_rerun()

    # Resize the image if necessary to fit on the page
    height, width = image.shape[:2]
    max_display_size = 800
    if max(height, width) > max_display_size:
        scaling = max_display_size / max(height, width)
        image = cv2.resize(
            image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
        )

    # Convert image from BGR (OpenCV format) to RGB (Streamlit display)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption=image_filename)
    st.write(
        "Press **Accept** to add the image to the final set, or **Skip** to move on."
    )

    # Layout two buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept"):
            # Append the current row to the list of accepted images
            st.session_state.labeled_rows.append(row.to_dict())
            st.write(f"Labeled: {img_path}")
            st.session_state.current_index += 1
            st.experimental_rerun()
    with col2:
        if st.button("Skip"):
            st.session_state.current_index += 1
            st.experimental_rerun()


if __name__ == "__main__":

    subdir = "subj_01"

    st.title("Image Labeler")
    # Load configuration from file
    config = load_config("config.yaml")
    # Define the path to the Excel file with images to label
    df_to_check_path = os.path.join(
        config.excel_files_target_dir, subdir, "faces", "faces_unchecked.xlsx"
    )

    target_path = os.path.join(
        config.excel_files_target_dir, subdir, "faces", "faces.xlsx"
    )
    label_images(config, df_to_check_path, target_path)
