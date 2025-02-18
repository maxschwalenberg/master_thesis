import os

from utils.config import Configuration, load_config

import numpy as np
import nibabel as nib


def filter_mask_t_test(config: Configuration, list_subj: list[int], threshold: float):
    for i, sub in enumerate(list_subj):

        mask_path_lh = os.path.join(
            config.t_test_roi_dir,
            "face_animate",
            f"lh.subj{sub:02d}.testrois.mgz",
        )
        mask_path_rh = os.path.join(
            config.t_test_roi_dir,
            "face_animate",
            f"rh.subj{sub:02d}.testrois.mgz",
        )

        updated_mask_path_lh = os.path.splitext(mask_path_lh)[0] + "_thresholded.mgz"
        updated_mask_path_rh = os.path.splitext(mask_path_rh)[0] + "_thresholded.mgz"

        mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()
        mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()

        mask = np.concatenate((mask_lh, mask_rh)).astype(int)

        t_test_result_path = os.path.join(
            config.t_test_results_dir, "face_animate", f"result_subj_{sub:02d}.npy"
        )
        t_test_result = np.load(t_test_result_path)

        # Apply thresholding: Set mask to 0 where t-test result is below the threshold
        mask[np.abs(t_test_result[:, 0]) < threshold] = 0

        thresholded_mask_lh = mask[: mask_lh.shape[0]]
        thresholded_mask_rh = mask[mask_lh.shape[0] :]

        # Ensure integer type (e.g., int16 to save space)
        thresholded_mask_lh = thresholded_mask_lh.astype(np.int16)
        thresholded_mask_rh = thresholded_mask_rh.astype(np.int16)

        # Create headers with explicit integer data type
        header_lh = nib.Nifti1Header()
        header_lh.set_data_dtype(np.int16)

        header_rh = nib.Nifti1Header()
        header_rh.set_data_dtype(np.int16)

        # Save updated left hemisphere mask
        out_mask_lh = nib.Nifti1Image(
            np.expand_dims(thresholded_mask_lh, axis=(1, 2)),
            np.eye(4),
            header=header_lh,
        )
        nib.save(out_mask_lh, updated_mask_path_lh)
        print(f"Saved updated left hemisphere mask to: {updated_mask_path_lh}")

        # Save updated right hemisphere mask
        out_mask_rh = nib.Nifti1Image(
            np.expand_dims(thresholded_mask_rh, axis=(1, 2)),
            np.eye(4),
            header=header_rh,
        )
        nib.save(out_mask_rh, updated_mask_path_rh)
        print(f"Saved updated right hemisphere mask to: {updated_mask_path_rh}")


if __name__ == "__main__":
    config = load_config("config.yaml")
    filter_mask_t_test(config, [1], 2)
