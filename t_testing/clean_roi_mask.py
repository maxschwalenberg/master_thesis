from utils.config import Configuration, load_config
import os
import nibabel as nib
import numpy as np
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def modify_mask_with_ttest(
    config: Configuration,
    threshold: float,
    subj_to_pick_shared: bool,
    subj_to_pick: int,
    sub_filename: str = "cleanedrois",
):
    """
    Modifies existing mask files based on t-test results. Sets mask values to 0
    where t-test values are below threshold.

    Args:
        config: Configuration object containing necessary paths
        threshold: T-test value threshold below which mask values will be set to 0
        label: Label string for output filename
        subj_to_pick_shared: Boolean indicating whether to use shared subject data
        t_test_results_subdir: Optional subdirectory for t-test results
    """
    # hemis = ["lh", "rh"]
    # mask_dir = os.path.join(config.directories.t_test_roi_dir)
    # if subj_to_pick_shared:
    #     mask_dir = os.path.join(mask_dir, "shared")
    # else:
    #     mask_dir = os.path.join(mask_dir, f"subj_{subj_to_pick:02d}")

    # mask_path_lh = os.path.join(mask_dir, f"lh.subj{subj_to_pick:02d}.testrois.mgz")
    # mask_path_rh = os.path.join(mask_dir, f"rh.subj{subj_to_pick:02d}.testrois.mgz")

    # new_mask_path_rh = os.path.join(
    #     mask_dir, f"rh.subj{subj_to_pick:02d}.{sub_filename}.mgz"
    # )
    # new_mask_path_lh = os.path.join(
    #     mask_dir, f"lh.subj{subj_to_pick:02d}.{sub_filename}.mgz"
    # )

    # mask_img_lh = nib.load(mask_path_lh)
    # mask_lh = mask_img_lh.get_fdata().squeeze()
    # original_dtype = mask_img_lh.get_data_dtype()

    # mask_img_rh = nib.load(mask_path_rh)
    # mask_rh = mask_img_rh.get_fdata().squeeze()

    # mask_lh_len = mask_lh.shape[0]
    # mask_rh_len = mask_rh.shape[0]

    # mask = np.concatenate((mask_lh, mask_rh)).astype(int)

    # logging.info(f"Processing {subj_to_pick} with threshold {threshold}")

    # # Load t-test data
    # if subj_to_pick_shared:
    #     subj_subdir = os.path.join("shared", f"subj_{(subj_to_pick):02d}")
    # else:
    #     subj_subdir = f"subj_{(subj_to_pick):02d}"

    # t_data = np.load(
    #     os.path.join(
    #         config.directories.t_test_results_dir,
    #         subj_subdir,
    #         f"result_subj_{subj_to_pick:02d}.npy",
    #     )
    # )

    # assert t_data.shape[0] == mask.shape[0]

    # for voxel_i in range(t_data.shape[0]):
    #     if mask[voxel_i] == 0:
    #         continue
    #     else:
    #         if np.abs(t_data[voxel_i][0]) < threshold:
    #             mask[voxel_i] = 0

    # new_mask_lh = mask[0:mask_lh_len].copy()
    # new_mask_rh = mask[mask_lh_len:].copy()

    # img = nib.Nifti1Image(
    #     np.expand_dims(new_mask_lh, axis=(1, 2)),
    #     mask_img_lh.affine,  # Use the original affine transformation
    #     dtype=original_dtype,
    # )
    # nib.save(img, new_mask_path_lh)

    # logging.info(f"Saving modified mask to {new_mask_path_lh}")
    # img = nib.Nifti1Image(
    #     np.expand_dims(new_mask_lh, axis=(1, 2)),
    #     mask_img_lh.affine,  # Use the original affine transformation
    #     dtype=original_dtype,
    # )
    # nib.save(img, new_mask_path_lh)

    # logging.info(f"Saving modified mask to {new_mask_path_rh}")
    # img = nib.Nifti1Image(
    #     np.expand_dims(new_mask_rh, axis=(1, 2)),
    #     mask_img_rh.affine,  # Use the original affine transformation
    #     dtype=original_dtype,
    # )
    # nib.save(img, new_mask_path_rh)
    # (Assuming config, subj_to_pick, subj_to_pick_shared, sub_filename, and threshold are defined above)
    hemis = ["lh", "rh"]
    mask_dir = os.path.join(config.directories.t_test_roi_dir)
    if subj_to_pick_shared:
        mask_dir = os.path.join(mask_dir, "shared")
    else:
        mask_dir = os.path.join(mask_dir, f"subj_{subj_to_pick:02d}")

    mask_path_lh = os.path.join(mask_dir, f"lh.subj{subj_to_pick:02d}.testrois.mgz")
    mask_path_rh = os.path.join(mask_dir, f"rh.subj{subj_to_pick:02d}.testrois.mgz")

    new_mask_path_rh = os.path.join(
        mask_dir, f"rh.subj{subj_to_pick:02d}.{sub_filename}.mgz"
    )
    new_mask_path_lh = os.path.join(
        mask_dir, f"lh.subj{subj_to_pick:02d}.{sub_filename}.mgz"
    )

    # Load the mask images for left and right hemispheres
    mask_img_lh = nib.load(mask_path_lh)
    mask_lh = mask_img_lh.get_fdata().squeeze()
    original_dtype = mask_img_lh.get_data_dtype()

    mask_img_rh = nib.load(mask_path_rh)
    mask_rh = mask_img_rh.get_fdata().squeeze()

    mask_lh_len = mask_lh.shape[0]
    mask_rh_len = mask_rh.shape[0]

    # Concatenate the two masks to one 1D (or 1D-like) array so that it aligns with t_data.
    mask = np.concatenate((mask_lh, mask_rh)).astype(int)

    logging.info(f"Processing subject {subj_to_pick} with threshold {threshold}")

    # Load t-test data
    if subj_to_pick_shared:
        subj_subdir = os.path.join("shared", f"subj_{subj_to_pick:02d}")
    else:
        subj_subdir = f"subj_{subj_to_pick:02d}"

    t_data = np.load(
        os.path.join(
            config.directories.t_test_results_dir,
            subj_subdir,
            f"result_subj_{subj_to_pick:02d}.npy",
        )
    )
    # Ensure that the number of voxels in t_data matches the mask
    assert t_data.shape[0] == mask.shape[0]

    # Instead of iterating voxel by voxel, loop over unique ROI labels in the mask
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label == 0:
            continue  # 0 typically denotes background
        # Get the indices in the mask that belong to this ROI
        roi_indices = np.where(mask == label)[0]

        # Extract the corresponding t-values; assuming the t-value is in the first column
        roi_t_values = t_data[roi_indices, 0]

        # Count how many are positive versus negative in this ROI
        pos_count = np.sum(roi_t_values > 0)
        neg_count = np.sum(roi_t_values < 0)

        # Determine thresholding strategy based on the majority sign
        if pos_count >= neg_count:
            # Majority are positive: keep voxels with t >= threshold
            keep_voxels = roi_t_values >= threshold
            logging.info(
                f"ROI {label}: majority positive (pos={pos_count}, neg={neg_count}) - keeping voxels with t >= {threshold} --> keep={keep_voxels.sum()}"
            )
        else:
            # Majority are negative: keep voxels with t <= -threshold
            keep_voxels = roi_t_values <= -threshold
            logging.info(
                f"ROI {label}: majority negative (pos={pos_count}, neg={neg_count}) - keeping voxels with t <= {-threshold} --> keep={keep_voxels.sum()}"
            )

        # Set voxels that do not meet the condition to 0 in the mask
        roi_to_zero = roi_indices[~keep_voxels]
        mask[roi_to_zero] = 0

    # Split the mask back into left and right hemispheres
    new_mask_lh = mask[0:mask_lh_len].copy()
    new_mask_rh = mask[mask_lh_len:].copy()

    # Save the modified left hemisphere mask
    img_lh = nib.Nifti1Image(
        np.expand_dims(new_mask_lh, axis=(1, 2)),
        mask_img_lh.affine,  # Use the original affine transformation
        dtype=original_dtype,
    )
    nib.save(img_lh, new_mask_path_lh)
    logging.info(f"Saving modified left hemisphere mask to {new_mask_path_lh}")

    # Save the modified right hemisphere mask
    img_rh = nib.Nifti1Image(
        np.expand_dims(new_mask_rh, axis=(1, 2)),
        mask_img_rh.affine,  # Use the original affine transformation
        dtype=original_dtype,
    )
    nib.save(img_rh, new_mask_path_rh)
    logging.info(f"Saving modified right hemisphere mask to {new_mask_path_rh}")

    


if __name__ == "__main__":
    config = load_config("config.yaml")
    modify_mask_with_ttest(config, 3.0, False, 1)
