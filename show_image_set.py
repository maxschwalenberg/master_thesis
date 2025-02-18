from utils.config import load_config, Configuration


import streamlit as st
import pandas as pd
import os

config = load_config("config.yaml")


# Use the image directory from your configuration
IMAGE_FOLDER = config.images_target_dir

# List image files (filtering for common image extensions)
df = pd.read_excel("data/labels/shared/faces/faces_unchecked.xlsx")
image_files = [e.split("/")[1] for e in df["file_name"]]
# image_files = sorted(
#     [
#         f
#         for f in os.listdir(IMAGE_FOLDER)
#         if f.lower().endswith(("png", "jpg", "jpeg", "gif"))
#     ]
# )

if not image_files:
    st.error("No images found in the directory!")
    st.stop()

# Initialize session state to keep track of the current image index
if "img_index" not in st.session_state:
    st.session_state.img_index = 0


def next_image():
    st.session_state.img_index = (st.session_state.img_index + 1) % len(image_files)


# Button to show next image
if st.button("Next Image"):
    next_image()
    # No need to call experimental_rerun()—the script re-runs automatically

# Display the current image
current_image_path = os.path.join(IMAGE_FOLDER, image_files[st.session_state.img_index])
st.image(current_image_path, use_column_width=True)

# Optionally display the filename for reference
st.caption(f"Displaying: {image_files[st.session_state.img_index]}")
