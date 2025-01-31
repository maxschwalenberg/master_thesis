import os
import glob
import time
import numpy as np
from scipy.spatial.distance import pdist
from previousCode.nsddatapaper_rsa.utils.utils import mds
import nibabel as nib
import pandas as pd
import logging

from utils.config import Configuration, load_config
from utils.utils import retrieve_stacked_betas

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


def create_rdm(
    config: Configuration,
    list_subj,
    mask_value: int,
    mode="averaged",
    sample_to_pick: int = 0,
):
    assert mode in ["averaged", "single"]

    logging.info(f"Creating RDM for mode {mode}")
    logging.info(f"Using positive set: {config.nsd_positive_subset}")

    if mode == "single":
        logging.info(f"Picking sample {sample_to_pick}")

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

        mask_lh = nib.load(mask_path_lh).get_fdata().squeeze()
        mask_rh = nib.load(mask_path_rh).get_fdata().squeeze()

        mask = np.concatenate((mask_lh, mask_rh)).astype(int)
        betas, image_ids = retrieve_stacked_betas(config, sub, mode, sample_to_pick)

        betas = np.transpose(betas)

        logging.info(f"{betas.shape=}")

        # Create mask for value maskvalue
        if isinstance(mask_value, int):
            masked_voxels = mask == mask_value

        elif isinstance(mask_value, list):
            masked_voxels = np.isin(mask, mask_value)

        else:
            raise ValueError()

        rdm_dir = os.path.join(config.rdm_dir, f"subj_{sub:02d}")
        os.makedirs(rdm_dir, exist_ok=True)

        if mode == "averaged":
            rdm_file = os.path.join(rdm_dir, f"mask_{mask_value}_{mode}_rdm.npy")
        elif mode == "single":
            rdm_file = os.path.join(
                rdm_dir,
                f"mask_{mask_value}_{mode}_sample_{sample_to_pick}_rdm.npy",
            )

        mds_dir = os.path.join(config.mds_dir, f"subj_{sub:02d}")
        metadata_file = os.path.join(config.rdm_dir, f"subj_{sub:02d}", "metadata.npy")
        logging.info(f"{len(image_ids)=}")
        np.save(metadata_file, image_ids)

        os.makedirs(mds_dir, exist_ok=True)

        if mode == "averaged":
            mds_file = os.path.join(mds_dir, f"mask_{mask_value}_{mode}_mds.npy")
        elif mode == "single":
            mds_file = os.path.join(
                mds_dir,
                f"mask_{mask_value}_{mode}_sample_{sample_to_pick}_mds.npy",
            )

        logging.info(f"Saving to ...\nRDM ----> {rdm_file}\nMDS ----> {mds_file}")

        if True:
            # Apply mask20 to betas
            masked_betas = betas[masked_voxels, :]

            if np.isnan(masked_betas).any():
                raise ValueError("Found NaNs!!")

            # Remove any voxels with NaN values
            good_vox = [np.sum(np.isnan(x)) == 0 for x in masked_betas]
            masked_betas = masked_betas[good_vox, :]

            if masked_betas.shape[0] == 0:
                logging.info(f"All voxels have NaN values ... {mask_value=}")
                return
                raise ValueError(f"All voxels have NaN values ... {mask_value=}")

            logging.info(
                f"Shape of masked betas for value {mask_value}: {masked_betas.shape}"
            )

            if np.isnan(masked_betas).any():
                masked_betas = masked_betas[~np.isnan(masked_betas).any(axis=1), :]

            # Transpose for correlation distance computation
            X = masked_betas.T

            logging.info(f"Masked betas shape: {X.shape}")

            rdm = pdist(X, metric="correlation")
            logging.info(f"RDM: {rdm.shape}")

            if np.any(np.isnan(rdm)):
                raise ValueError("NaN values found in RDM")

            np.save(rdm_file, rdm)

        if True:

            rdm = np.load(rdm_file, allow_pickle=True).astype(np.float32)

            mds_out = mds(rdm).astype(np.float32)
            logging.info(f"MDS: {mds_out.shape}")

            np.save(mds_file, mds_out)

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
    config = load_config("config.yaml")
    mask_values = list(config.rois_to_analyze.values())
    for mask_value in mask_values:
        create_rdm(config, [1], mask_value, mode="averaged")
