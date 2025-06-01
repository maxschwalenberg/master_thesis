import os
import glob
import time
import numpy as np
from scipy.spatial.distance import pdist
from previousCode.nsddatapaper_rsa.utils.utils import mds
from t_testing.clean_roi_mask import modify_mask_with_ttest
import nibabel as nib
import pandas as pd
import logging

from utils.config import Configuration, load_config
from utils.utils import retrieve_stacked_betas
from scipy.stats import wasserstein_distance


"""
This file takes the betas, the mask and computes
the RDM based on the pdist scipy function 

The mask were created using a matlab script, available in the repo 
(credit to Luis for that)

The RDMs are then stored under the projects directory, and can be access later on 

Then, we used these RDM (or ones previously computed) to get the corresponding MDS 

"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Define custom metric wrapper
def emd(u, v):
    return wasserstein_distance(u, v)


def create_rdm(
    config: Configuration,
    list_subj,
    mask_value: int,
    set_to_take: str,
    t_test_threshold: float,
    mode="averaged",
    sample_to_pick: int = 0,
    randomization: bool = False,
):
    
    augment_shared_set = True

    assert mode in ["averaged", "single"]
    logging.info(f"Creating RDM for mode {mode} - T-Test THRESHOLD: {t_test_threshold}")
    logging.info(
        f"Using distance-metric={config.pipeline.step_4_rsa_analysis.distance_metric}"
    )
    time.sleep(0.25)
    logging.info(
        f"Using positive set: {os.path.join(set_to_take, config.dataset_creation.subset_animate_face_final)}"
    )

    if mode == "single":
        logging.info(f"Picking sample {sample_to_pick}")

    for i, sub in enumerate(list_subj):
        # Determine if using shared set or not
        pick_shared = set_to_take == "shared"

        # Modify and generate temporary t-test mask files
        modify_mask_with_ttest(
            config, t_test_threshold, pick_shared, sub, sub_filename="temporary_mask"
        )

        # Load left and right hemisphere masks
        mask_path_lh = os.path.join(
            config.directories.t_test_roi_dir,
            set_to_take,
            f"lh.subj{sub:02d}.temporary_mask.mgz",
        )
        mask_path_rh = os.path.join(
            config.directories.t_test_roi_dir,
            set_to_take,
            f"rh.subj{sub:02d}.temporary_mask.mgz",
        )
        logging.info(f"Loading mask from\n{mask_path_rh=}\n{mask_path_lh=}")

        mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()
        mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()

        # Load beta values and transpose for further analysis
        betas, image_ids, mds_mapping = retrieve_stacked_betas(
            config,
            sub,
            mode,
            sample_to_pick,
            subj_to_check=set_to_take,
            only_face_set=config.pipeline.step_4_rsa_analysis.only_face_set,
            randomization=randomization,
            augment_shared_set=augment_shared_set,
        )
        assert not np.isnan(betas).any()
        betas = np.transpose(betas)

        # For the combined (both) analysis, check that the beta dimensions match the concatenated mask
        combined_mask = np.concatenate((mask_lh, mask_rh)).astype(int)
        assert betas.shape[0] == combined_mask.shape[0], (
            f"Beta shape {betas.shape[0]} does not match combined mask shape "
            f"{combined_mask.shape[0]} for subject {sub}"
        )

        # Split beta values according to hemisphere (assuming left mask first, then right)
        n_lh = mask_lh.shape[0]
        betas_both = betas  # full concatenated beta matrix
        betas_lh = betas[:n_lh, :]
        betas_rh = betas[n_lh:, :]

        # Save metadata (e.g., image_ids) once per subject.
        rdm_dir_subject = os.path.join(
            config.directories.rdm_dir, set_to_take, f"subj_{sub:02d}"
        )
        os.makedirs(rdm_dir_subject, exist_ok=True)
        metadata_file = os.path.join(rdm_dir_subject, "metadata.npy")
        logging.info(f"{len(image_ids)=}")
        np.save(metadata_file, image_ids)

        # For MDS outputs, make sure the target directory exists.
        mds_dir_subject = os.path.join(
            config.directories.mds_dir, set_to_take, f"subj_{sub:02d}"
        )
        os.makedirs(mds_dir_subject, exist_ok=True)

        # Define the three hemispheric analyses.
        for hemisphere in ["both", "lh", "rh"]:
            if hemisphere == "both":
                current_mask = combined_mask
                current_betas = betas_both
            elif hemisphere == "lh":
                current_mask = mask_lh.astype(int)
                current_betas = betas_lh
            elif hemisphere == "rh":
                current_mask = mask_rh.astype(int)
                current_betas = betas_rh

            logging.info(f"Processing hemisphere '{hemisphere}' for subject {sub:02d}")

            # Create the voxel mask based on mask_value
            if isinstance(mask_value, int):
                masked_voxels = current_mask == mask_value
            elif isinstance(mask_value, list):
                masked_voxels = np.isin(current_mask, mask_value)
            else:
                raise ValueError("mask_value must be either an int or a list of ints")

            # Define output file paths
            if mode == "averaged":
                rdm_file = os.path.join(
                    rdm_dir_subject, f"mask_{mask_value}_{mode}_{hemisphere}_rdm.npy"
                )
                mds_file = os.path.join(
                    mds_dir_subject, f"mask_{mask_value}_{mode}_{hemisphere}_mds.npy"
                )
            elif mode == "single":
                rdm_file = os.path.join(
                    rdm_dir_subject,
                    f"mask_{mask_value}_{mode}_sample_{sample_to_pick}_{hemisphere}_rdm.npy",
                )
                mds_file = os.path.join(
                    mds_dir_subject,
                    f"mask_{mask_value}_{mode}_sample_{sample_to_pick}_{hemisphere}_mds.npy",
                )

            # Apply the mask to the beta values for this hemisphere
            masked_betas = current_betas[masked_voxels, :]

            if np.isnan(masked_betas).any():
                raise ValueError("Found NaNs in masked betas!")

            # Remove any voxels which contain any NaN values along the feature axis.
            good_vox = [np.sum(np.isnan(x)) == 0 for x in masked_betas]
            masked_betas = masked_betas[good_vox, :]

            if masked_betas.shape[0] == 0:
                logging.info(
                    f"All voxels have NaN values ... {mask_value=} for hemisphere '{hemisphere}'"
                )
                continue  # Skip to the next hemisphere analysis

            logging.info(
                f"Shape of masked betas for mask value {mask_value}, hemisphere '{hemisphere}': {masked_betas.shape}"
            )

            if np.isnan(masked_betas).any():
                masked_betas = masked_betas[~np.isnan(masked_betas).any(axis=1), :]

            # Transpose for distance computation (features as rows)
            X = masked_betas.T
            logging.info(f"Masked betas shape for distance computation: {X.shape}")

            # Determine which distance metric to use
            if config.pipeline.step_4_rsa_analysis.distance_metric != "wasserstein":
                metric_to_use = config.pipeline.step_4_rsa_analysis.distance_metric
            else:
                metric_to_use = emd  # assuming emd is defined/imported

            # Compute the RDM using pdist (e.g. from scipy.spatial.distance)
            rdm = pdist(X, metric=metric_to_use)
            logging.info(
                f"Computed RDM shape for hemisphere '{hemisphere}': {rdm.shape}"
            )

            if np.any(np.isnan(rdm)):
                raise ValueError("NaN values found in RDM")

            np.save(rdm_file, rdm)
            logging.info(f"Saved RDM to {rdm_file}")

            # Load the RDM back and run MDS (assuming mds() is defined/imported)
            rdm_loaded = np.load(rdm_file, allow_pickle=True).astype(np.float32)
            mds_out = mds(rdm_loaded).astype(np.float32)
            logging.info(
                f"MDS output shape for hemisphere '{hemisphere}': {mds_out.shape}"
            )
            np.save(mds_file, mds_out)
            logging.info(f"Saved MDS output to {mds_file}")

        # if np.isnan(betas).any():
        #       # SUBJ06 AND SUBJ08 Have NaNs for some  reason: need to remove and replace. Can't do before because of the masking
        #     # which means I need to change the masks
        #     # The hemishpere masks have been stack horizontally so we use that
        #     # so if a voxel is below the cutoff, its in LH. if its above, its in RH
        #     #
        #     # This was quite a time waster, other solution might be preferable
        #     #
        #     # ++ SUBJ06 and SUBJ08 do quite poorly: removing them entirely is an option, although
        #     # the amount of voxels that the code below removes is very low
        #     #
        #     # Also yes, this should be in load_betas.py

        #     if mode == 'train':
        #         betas_test_file = os.path.join(betas_dir, f'{sub}_betas_list_{targetspace}_test.npy')  # also do it for test
        #         betas_test = np.load(betas_test_file, allow_pickle=True).astype(np.float32)
        #         betas_test = betas_test[~np.isnan(betas).any(axis=1)]
        #         print(betas_test.shape)
        #         np.save(betas_test_file, betas_test)

        #         betas_mask_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy') # also do it for mask
        #         print('FIXING MASKS')
        #         betas_mask = np.load(betas_mask_file, allow_pickle=True)
        #         print(betas_mask.shape)
        #         betas_mask = betas_mask[~np.isnan(betas).any(axis=1)]
        #         betas_mask_file = os.path.join(mask_dir, sub, f'short.reduced.nans.{sub}.testrois.npy')
        #         np.save(betas_mask_file, betas_mask)

        #     mask_nans = np.ones(betas.shape[0], dtype = bool)
        #     mask_nans[np.where(np.isnan(betas).any(axis=1))[0]] = False
        #     np.save(os.path.join(betas_dir, f'{sub}_betas_nans_mask.npy'), mask_nans)

        #     maskdata_reduced_file = os.path.join(mask_dir, sub, f'short.reduced.{sub}.testrois.npy')
        #     maskdata_reduced = np.load(maskdata_reduced_file).astype(int)
        #     maskdata_reduced = maskdata_reduced[~np.isnan(betas).any(axis=1)]
        #     maskdata_reduced_file_new = os.path.join(mask_dir, sub, f'short.reduced.nans.{sub}.testrois.npy')  # also rewrite the mask (needed for model building)
        #     np.save(maskdata_reduced_file_new, maskdata_reduced)

        #     maskdata_lh_path = os.path.join(mask_dir, sub , f'lh.{sub}.testrois.mgz')
        #     maskdata_lh = nib.load(maskdata_lh_path).get_fdata().squeeze()
        #     maskdata_rh_path = os.path.join(mask_dir, sub , f'rh.{sub}.testrois.mgz')
        #     maskdata_rh = nib.load(maskdata_rh_path).get_fdata().squeeze()

        #     maskdata_lh[maskdata_lh > (max(rois_long.values()))] = 0  # Remove ROIs above TO-2. Apparently this is necessary to not break???
        #     maskdata_rh[maskdata_rh > (max(rois_long.values()))] = 0

        #     maskdata_lh_short_path = os.path.join(mask_dir, sub, f'lh.short.{sub}.testrois.npy')
        #     maskdata_lh_short = np.load(maskdata_lh_short_path).astype(int)
        #     maskdata_rh_short_path = os.path.join(mask_dir, sub, f'rh.short.{sub}.testrois.npy')
        #     maskdata_rh_short = np.load(maskdata_rh_short_path).astype(int)

        #     lh_indices = np.where((maskdata_lh >= 1) & (maskdata_lh <= 15))[0]
        #     rh_indices = np.where((maskdata_rh >= 1) & (maskdata_rh <= 15))[0]
        #     # I am storing the indexes of the mask corresponding to our ROIs so I can delete said indexes, without reformatting the whole mask

        #     indices_to_detele_lh = lh_indices[np.isnan(betas[:lh_indices.shape[0]]).any(axis=1)]
        #     indices_to_detele_rh = rh_indices[np.isnan(betas[lh_indices.shape[0]:]).any(axis=1)]
        #     # lh_indices length is also the number of voxels in LH, so we know which voxel is in which hemi

        #     maskdata_lh_short_new = maskdata_lh_short[~np.isnan(betas[:maskdata_lh_short.shape[0]]).any(axis=1)] # this makes more sense in my mind
        #     maskdata_rh_short_new = maskdata_rh_short[~np.isnan(betas[maskdata_lh_short.shape[0]:]).any(axis=1)]

        #     maskdata_lh_short_new_path = os.path.join(mask_dir, sub, f'lh.short.nans.{sub}.testrois.npy')
        #     maskdata_rh_short_new_path = os.path.join(mask_dir, sub, f'rh.short.nans.{sub}.testrois.npy')

        #     np.save(maskdata_lh_short_new_path, maskdata_lh_short_new)
        #     np.save(maskdata_rh_short_new_path, maskdata_rh_short_new)

        #     maskdata_lh[indices_to_detele_lh] = 0
        #     maskdata_rh[indices_to_detele_rh] = 0

        #     # I need to do that for distance/correlation calculations
        #     maskdata_lh_deleted = np.delete(maskdata_lh, indices_to_detele_lh)
        #     maskdata_rh_deleted = np.delete(maskdata_rh, indices_to_detele_rh)

        #     maskdata_lh_img = nib.Nifti2Image(maskdata_lh, affine=None)
        #     maskdata_rh_img = nib.Nifti2Image(maskdata_rh, affine=None)

        #     maskdata_lh_del_img = nib.Nifti2Image(maskdata_lh_deleted, affine=None)
        #     maskdata_rh_del_img = nib.Nifti2Image(maskdata_rh_deleted, affine=None)

        #     lh_path = os.path.join(mask_dir, sub, f'lh.{sub}.nans.testrois.mgz')
        #     rh_path = os.path.join(mask_dir, sub, f'rh.{sub}.nans.testrois.mgz')

        #     lh_path_del = os.path.join(mask_dir, sub, f'lh.{sub}.nans_del.testrois.mgz')
        #     rh_path_del = os.path.join(mask_dir, sub, f'rh.{sub}.nans_del.testrois.mgz')

        #     nib.save(maskdata_lh_img, lh_path)
        #     nib.save(maskdata_rh_img, rh_path)

        #     nib.save(maskdata_lh_del_img, lh_path_del)
        #     nib.save(maskdata_rh_del_img, rh_path_del)

        #     ### Also need to create a new white file for the meshes on the surface,
        #     ### which is used when computing correlation
        #     lh_white_path = os.path.join(label_dir, 'freesurfer', sub, 'surf', 'lh.white')
        #     rh_white_path = os.path.join(label_dir, 'freesurfer', sub, 'surf', 'rh.white')

        #     # I think this mess is not used at the end
        #     lh_white = nib.freesurfer.read_geometry(lh_white_path)
        #     rh_white = nib.freesurfer.read_geometry(rh_white_path)

        #     lh_coords_del = np.delete(lh_white[0], indices_to_detele_lh, axis=0)
        #     rh_coords_del = np.delete(rh_white[0], indices_to_detele_rh, axis=0)

        #     mask_faces_lh = np.isin(lh_white[1], indices_to_detele_lh)
        #     mask_faces_rh = np.isin(rh_white[1], indices_to_detele_rh)

        #     rows_excluded_lh = np.any(mask_faces_lh, axis=1)
        #     rows_excluded_rh = np.any(mask_faces_rh, axis=1)

        #     mask_white_lh = np.ones(lh_white[0].shape[0], dtype = bool)  ### Here we cannot just deleted the triangles that store deleted voxels's indexes: we also need to updated the other indexes: if I delete voxel 1, voxel 2 needs to become voxel 1, etc... because we do not store indexes in these file format. Another solution could be to store deleted indexes in the white[0] file as an arbitrary negative value but this is what I have now
        #     mask_white_rh = np.ones(rh_white[0].shape[0], dtype = bool)

        #     mask_white_lh[indices_to_detele_lh] = False
        #     mask_white_rh[indices_to_detele_rh] = False

        #     new_index_lh = np.cumsum(mask_white_lh) -1 # this updated our indexes and set the unwanted ones as -1. Bless the numpy gods for this one
        #     new_index_rh = np.cumsum(mask_white_rh) -1

        #     valid_faces_maks_lh  = ~np.isin(lh_white[1], indices_to_detele_lh).any(axis=1)
        #     valid_faces_maks_rh  = ~np.isin(rh_white[1], indices_to_detele_rh).any(axis=1)

        #     filter_faces_lh = lh_white[1][valid_faces_maks_lh]
        #     filter_faces_rh = rh_white[1][valid_faces_maks_rh]

        #     updated_faces_lh = new_index_lh[filter_faces_lh]
        #     updated_faces_rh = new_index_rh[filter_faces_rh]

        #     # nibabel.freesurfer.io.write_geometry(filepath, coords, faces, create_stamp=None, volume_info=None)

        #     lh_white_del_path = os.path.join(label_dir, 'freesurfer', sub, 'surf', 'lh.white_del')
        #     rh_white_del_path = os.path.join(label_dir, 'freesurfer', sub, 'surf', 'rh.white_del')

        #     nib.freesurfer.write_geometry(lh_white_del_path, lh_coords_del, updated_faces_lh)
        #     nib.freesurfer.write_geometry(rh_white_del_path, rh_coords_del, updated_faces_rh)

        #     betas = betas[~np.isnan(betas).any(axis=1), :]

        #     np.save(betas_file, betas)


# create_rdm(subjects_sessions)


if __name__ == "__main__":
    for subj_id in range(1, 9):
        set_to_take = f"subj_{subj_id:02d}"

        config = load_config("config.yaml")
        mask_values = list(config.analysis.rois_to_analyze.values())
        for mask_value in mask_values:
            try:
                n_samples = 3
                for sample_v in range(0, n_samples):
                    create_rdm(
                        config,
                        [subj_id],
                        mask_value,
                        set_to_take,
                        2.0,
                        mode="single",
                        sample_to_pick=sample_v,
                        randomization=True,
                    )

            except:
                n_samples = 2
                for sample_v in range(0, n_samples):
                    create_rdm(
                        config,
                        [subj_id],
                        mask_value,
                        set_to_take,
                        2.0,
                        mode="single",
                        sample_to_pick=sample_v,
                        randomization=True,
                    )
