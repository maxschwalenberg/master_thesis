import numpy as np
import nibabel as nib
import os
from utils.config import load_config, Configuration
from tqdm import tqdm
from typing import Union

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def mutually_exclusive_t_test_export(
    config: Configuration,
    threshold: float,
    face_animate_t_test_subdir: str,
    animate_nonanimate_t_test_subdir: str,
):

    os.makedirs(
        os.path.join(config.t_test_results_dir, "mutually_exclusive"), exist_ok=True
    )

    subj = 1

    face_animate_t_test_res = np.load(
        os.path.join(
            config.t_test_results_dir,
            face_animate_t_test_subdir,
            f"result_subj_{subj:02d}.npy",
        )
    )
    animate_nonanimate_t_test_res = np.load(
        os.path.join(
            config.t_test_results_dir,
            animate_nonanimate_t_test_subdir,
            f"result_subj_{subj:02d}.npy",
        )
    )

    for index in tqdm(range(animate_nonanimate_t_test_res.shape[0])):
        # if face_animate_t_test_res[index][0] < [index][0]
        if face_animate_t_test_res[index][0] > threshold:
            # if animate_nonanimate_t_test_res[index][0] > threshold:
            #     face_animate_t_test_res[index][0] = 0
            if (
                face_animate_t_test_res[index][0]
                - animate_nonanimate_t_test_res[index][0]
                < 3
            ):
                face_animate_t_test_res[index][0] = 0

    np.save(
        os.path.join(
            config.t_test_results_dir,
            "mutually_exclusive",
            f"result_subj_{subj:02d}.npy",
        ),
        face_animate_t_test_res,
    )


def t_test_results_to_mgz(
    config: Configuration,
    mode: str,
    threshold: float,
    label: str,
    subj_to_pick_shared: bool,
    thresholding: bool = True,
    t_test_results_subdir: str = "",
    clipping_value: Union[float, None] = None,
):
    hemis = ["lh", "rh"]

    subs = [f"subj{i:02d}" for i in config.subjects_to_analyze]
    mask_dir = os.path.join(config.stans_thesis_repo_data, config.mask_data_dir)

    if subj_to_pick_shared:
        shared_string = "shared"
    else:
        shared_string = "no_shared"

    assert mode in ["absolute", "signed"]

    for sub_i, sub in enumerate(subs):
        logging.info(f"{sub}\t{mode=}\t{thresholding=}\t{threshold=}")

        if subj_to_pick_shared:
            subj_subdir = os.path.join("shared", f"subj_{(sub_i+1):02d}")
        else:
            subj_subdir = f"subj_{(sub_i+1):02d}"

        t_data = np.load(
            os.path.join(
                config.t_test_results_dir,
                subj_subdir,
                f"result_subj_{sub[4:]}.npy",
            )
        )

        logging.info(f"")

        if clipping_value is not None:
            t_data = np.clip(t_data, -clipping_value, clipping_value)
            max_value = clipping_value
            min_value = -clipping_value
        else:
            max_value = np.nanmax(t_data[:, 0]).item()
            min_value = np.nanmin(t_data[:, 0]).item()

        hemisphere_shapes = []

        for hemi_i, hemi in enumerate(hemis):

            if sub == "subj06" or sub == "subj08":
                maskdata_long_file = os.path.join(
                    mask_dir, sub, f"{hemi}.{sub}.nans.testrois.mgz"
                )
            else:
                maskdata_long_file = os.path.join(
                    mask_dir, sub, f"{hemi}.{sub}.testrois.mgz"
                )
            maskdata_long = nib.load(maskdata_long_file).get_fdata().squeeze()

            hemisphere_shapes.append(maskdata_long.shape[0])

            if clipping_value is None:

                data_out_file = os.path.join(
                    config.freesurfer_dir,
                    sub,
                    "label",
                    f"{hemi}.t_test_{mode}_{str(threshold).replace('.', '_')}_label_{label}_{shared_string}.mgz",
                )
            else:
                data_out_file = os.path.join(
                    config.freesurfer_dir,
                    sub,
                    "label",
                    f"{hemi}.t_test_{mode}_{str(threshold).replace('.', '_')}_label_{label}_clip_{clipping_value}_{shared_string}.mgz",
                )

            logging.info(f"Saving to {data_out_file}")

            data_out = np.zeros(maskdata_long.shape)

            for i in range(maskdata_long.shape[0]):
                # this needs to be commented out so full brain is taken
                # voxel = int(maskdata_long[i])
                # if voxel==0 or voxel > 15:

                #    continue
                # s += 1

                if hemi_i == 0:
                    index = i
                else:
                    index = i + hemisphere_shapes[0]

                if thresholding:
                    if np.abs(t_data[index][0]) < threshold:
                        continue

                if mode == "absolute":
                    data_out[i] = np.abs(t_data[index][0])
                else:
                    data_out[i] = t_data[index][0] + np.abs(min_value)

            logging.info(f"{data_out_file}")
            img = nib.Nifti1Image(np.expand_dims(data_out, axis=(1, 2)), np.eye(4))
            nib.loadsave.save(img, data_out_file)

        if mode == "signed":
            range_max = max(max_value, np.abs(min_value))
            val_range = [-range_max, range_max]
            adjusted_range = [0, 2 * range_max]

            logging.info(f"Data range: {val_range}")
            logging.info(f"Adjusted Data range: {adjusted_range}")
            logging.info(f"Maximal Value: {max_value}")
            logging.info(f"Minimal Value: {min_value}")

        else:
            logging.info(f"Maximal Value: {max_value}")
            logging.info(f"Minimal Value: {min_value}")

        print("-" * 50)


if __name__ == "__main__":
    config = load_config("config.yaml")

    # mutually_exclusive_t_test_export(config, 5.0, "face_animate", "animate_nonanimate")

    # quit()

    t_test_subdir = "subj_01"
    subj_to_pick_shared = False

    for labels_subdir in ["across"]:
        for mode in ["absolute", "signed"]:
            t_test_results_to_mgz(
                config,
                mode,
                0.0,
                labels_subdir,
                subj_to_pick_shared,
                thresholding=False,
                t_test_results_subdir=t_test_subdir,
                clipping_value=7,
            )

            t_test_results_to_mgz(
                config,
                mode,
                5.0,
                labels_subdir,
                subj_to_pick_shared,
                thresholding=True,
                t_test_results_subdir=t_test_subdir,
                clipping_value=7,
            )

            t_test_results_to_mgz(
                config,
                mode,
                4.0,
                labels_subdir,
                subj_to_pick_shared,
                thresholding=True,
                t_test_results_subdir=t_test_subdir,
                clipping_value=7,
            )

            t_test_results_to_mgz(
                config,
                mode,
                3.0,
                labels_subdir,
                subj_to_pick_shared,
                thresholding=True,
                t_test_results_subdir=t_test_subdir,
                clipping_value=7,
            )

            t_test_results_to_mgz(
                config,
                mode,
                2.0,
                labels_subdir,
                subj_to_pick_shared,
                thresholding=True,
                t_test_results_subdir=t_test_subdir,
                clipping_value=7,
            )
