import pandas as pd
import cv2
from utils.config import Configuration, load_config
import os


def label_images(config: Configuration, df_to_check_path: str):

    print(
        """Instructions:
          
          Press -ENTER- to accept the sample to the final set, anything else to skip."""
    )

    # Read the CSV file
    df = pd.read_excel(df_to_check_path)

    # Create an empty dataframe to store labeled images
    labeled_df = pd.DataFrame(columns=df.columns)

    # Loop through each row in the dataframe
    for index, row in df.iterrows():
        # Read the image
        image_filename = os.path.basename(row["file_name"])
        img_path = os.path.join(config.images_target_dir, image_filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Could not read image: {img_path}")
            continue

        # Create window and set it to normal size (not fullscreen)
        cv2.namedWindow("Image Labeler", cv2.WINDOW_NORMAL)

        # Resize window to a reasonable size while maintaining aspect ratio
        height, width = image.shape[:2]
        max_display_size = 800
        if max(height, width) > max_display_size:
            scaling = max_display_size / max(height, width)
            image = cv2.resize(
                image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
            )

        # Show the image
        cv2.imshow("Image Labeler", image)

        # Wait for key press
        key = cv2.waitKey(0)

        # If Enter key is pressed (13 is Enter key code)
        if key == 13:
            # Append the current row to labeled dataframe
            labeled_df = pd.concat([labeled_df, pd.DataFrame([row])], ignore_index=True)
            print(f"Labeled: {img_path}")

        # Close the image window
        cv2.destroyAllWindows()

    # Return the labeled dataframe
    return labeled_df


if __name__ == "__main__":
    config = load_config("config.yaml")

    # Example usage
    df_to_check_path = os.path.join(
        config.excel_files_target_dir, "nsd_positive_subset_unchecked.xlsx"
    )

    labeled_images = label_images(config, df_to_check_path)
    labeled_images = labeled_images.loc[
        :, ~labeled_images.columns.str.contains("Unnamed", case=False, na=False)
    ]

    labeled_images.to_excel(
        os.path.join(config.excel_files_target_dir, config.nsd_positive_subset),
        index=False,
    )
