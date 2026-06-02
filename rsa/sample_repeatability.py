"""
Sample Repeatability Analysis Module

This module analyzes the repeatability of neural representations across samples
by computing RDMs and MDS coordinates for multiple trials, then comparing
within-image distances against null distributions. It includes functionality
for generating comprehensive repeatability statistics and correlation analyses.
"""

# Standard library imports
import logging
import os
import pickle
from itertools import combinations
from typing import Iterable, List, Tuple

# Third-party imports
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.stats import pearsonr, spearmanr, wasserstein_distance, ttest_1samp

# Local imports
from rsa.create_rdm import create_rdm
from t_testing.clean_roi_mask import modify_mask_with_ttest
from utils.config import Configuration, load_config
from utils.utils import retrieve_stacked_betas
from previousCode.nsddatapaper_rsa.utils.utils import mds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# CONFIGURATION AND FILE PATHS
# =========================================================================

config = load_config("config.yaml")

sample_repeatability_distances_pickle = os.path.join(
    config.directories.output_dir,
    config.saved_results_paths.sample_repeatability_distances_pickle,
)
sample_repeatability_distances_excel = os.path.join(
    config.directories.output_dir,
    config.saved_results_paths.sample_repeatability_distances_excel,
)
sample_repeatability_correlations_true_excel = os.path.join(
    config.directories.output_dir,
    config.saved_results_paths.sample_repeatability_correlations_true_excel,
)
sample_repeatability_correlations_null_excel = os.path.join(
    config.directories.output_dir,
    config.saved_results_paths.sample_repeatability_correlations_null_excel,
)

# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================


def emd(u, v):
    """
    Custom metric wrapper for Earth Mover's Distance (Wasserstein distance).

    Args:
        u, v: Input arrays for distance computation

    Returns:
        float: Wasserstein distance between u and v
    """
    return wasserstein_distance(u, v)


# =========================================================================
# CORE RDM GENERATION FUNCTIONS
# =========================================================================


def create_rdm_with_all_trials(
    config: Configuration,
    subj_list: list[int],
    mask_values: list[int],
    set_to_take,
    t_test_threshold,
    randomization=False,
):
    """
    Generate RDMs using all trials per image for comprehensive repeatability analysis.

    For each subject, load all 3 trials per image (e.g., 150 images → 450 patterns),
    apply ROI mask, compute pairwise RDM (450×450) and MDS coordinates (450×2).

    Args:
        config: Configuration object containing analysis parameters
        subj_list: List of subject IDs to process
        mask_values: List of ROI mask values to analyze
        set_to_take: Dataset identifier
        t_test_threshold: Threshold for t-test based masking
        randomization: Whether to apply randomization

    Returns:
        dict: Results dictionary with structure:
            { subject_id : {
                mask_value: {
                    'rdm': 1D array of pairwise distances,
                    'mds': array (n_patterns, 2) of MDS coordinates,
                    'meta': list of (image_id, trial_idx) for each pattern
                }
            }}
    """
    logging.info(f"Creating RDMs with all trials for subjects: {subj_list}")
    results = {}

    for sub in tqdm(subj_list, desc="Subjects"):
        results[sub] = {}

        # Generate t-test mask
        pick_shared = set_to_take == "shared"
        modify_mask_with_ttest(
            config, t_test_threshold, pick_shared, sub, sub_filename="temporary_mask"
        )

        # Load and concatenate hemisphere masks
        roi_dir = config.directories.t_test_roi_dir
        mask_lh = (
            nib.load(
                os.path.join(
                    roi_dir, set_to_take, f"lh.subj{sub:02d}.temporary_mask.mgz"
                )
            )
            .get_fdata()
            .squeeze()
        )
        mask_rh = (
            nib.load(
                os.path.join(
                    roi_dir, set_to_take, f"rh.subj{sub:02d}.temporary_mask.mgz"
                )
            )
            .get_fdata()
            .squeeze()
        )
        combined_mask = np.concatenate((mask_lh, mask_rh)).astype(int)

        # Collect all patterns across trials
        all_patterns = []
        meta = []  # Store (image_id, trial_idx) metadata

        for trial_idx in range(3):
            try:
                betas, image_ids, _ = retrieve_stacked_betas(
                    config,
                    sub,
                    "single",  # Single trial mode
                    trial_idx,
                    subj_to_check=set_to_take,
                    only_face_set=config.pipeline.step_4_rsa_analysis.only_face_set,
                    randomization=randomization,
                    augment_shared_set=True,
                )
            except Exception as e:
                logging.warning(f"Error retrieving trial {trial_idx}: {e}")
                continue

            all_patterns.append(betas)
            meta += [(img, trial_idx) for img in image_ids]

        # Stack all patterns: (n_trials_total, n_voxels)
        X = np.vstack(all_patterns)
        assert len(meta) == X.shape[0], "Metadata length must match pattern count"

        # Process each ROI mask
        for mv in mask_values:
            # Build voxel selector for this ROI
            if isinstance(mv, int):
                vox_selector = combined_mask == mv
            else:
                vox_selector = np.isin(combined_mask, mv)
            vox_selector = vox_selector.flatten()

            n_vox = vox_selector.sum()
            if n_vox == 0:
                logging.info(f"Mask {mv}: no voxels → skipping")
                continue

            logging.info(f"Mask {mv}: {n_vox} voxels")

            # Apply mask and compute RDM + MDS
            X_masked = X[:, vox_selector]
            assert not np.isnan(X_masked).any(), "Masked data contains NaN values"

            # Use configured distance metric
            dm = config.pipeline.step_4_rsa_analysis.distance_metric
            metric_fn = emd if dm == "wasserstein" else dm
            rdm_vec = pdist(X_masked, metric=metric_fn)
            mds_coords = mds(rdm_vec).astype(np.float32)

            # Store results
            results[sub][mv] = {"rdm": rdm_vec, "mds": mds_coords, "meta": meta}

    return results


def create_all_rdms(
    config: Configuration,
    subj_id: int,
    n_samples: int = 3,
    n_randomizations: int = 1,
    t_thresh: float = 2.5,
    mask_values: list = list(range(1, 10)),
):
    """
    Generate RDMs for one subject across all configured ROIs and samples.

    Args:
        config: Configuration object
        subj_id: Subject ID to process
        n_samples: Number of samples per subject
        n_randomizations: Number of randomization iterations
        t_thresh: T-test threshold for masking
        mask_values: List of ROI mask values to process
    """
    logging.info(f"Generating RDMs for subject {subj_id:02d}")
    set_to_take = f"subj_{subj_id:02d}"

    while True:
        try:
            for sample_v in range(n_samples):
                for n_rand in range(n_randomizations):
                    create_rdm(
                        config,
                        [subj_id],
                        mask_values,
                        set_to_take,
                        t_thresh,
                        mode="single",
                        sample_to_pick=sample_v,
                        randomization=True,
                        augment_shared_set=True,
                        randomize_offset=n_rand,
                    )
            break
        except Exception as e:
            logging.error(f"Failed generating RDMs for subject {subj_id}, error: {e}")
            break


# =========================================================================
# DISTANCE ANALYSIS FUNCTIONS
# =========================================================================


def sample_repeatability_distances_data(
    config: Configuration,
    subjects: list[int],
    mask_values: list[int] = list(range(1, 10)),
    t_thresh: float = 2.5,
    n_perm: int = 10,
):
    """
    Analyze sample repeatability by comparing within-image distances to null distributions.

    This function computes true within-image distances and compares them against
    null distributions generated by permuting stimulus labels.

    Args:
        config: Configuration object
        subjects: List of subject IDs to analyze
        mask_values: List of ROI mask values
        t_thresh: T-test threshold for masking
        n_perm: Number of permutations for null distribution
    """
    logging.info("Starting sample repeatability distance analysis")

    n_perm = 5000  # Override for thorough analysis
    rng = np.random.RandomState(0)  # Fixed seed for reproducibility

    # Load or generate RDM data
    results_all = {}

    if os.path.exists(sample_repeatability_distances_pickle):
        logging.info(
            f"Loading existing data from {sample_repeatability_distances_pickle}"
        )
        with open(sample_repeatability_distances_pickle, "rb") as f:
            results_all = pickle.load(f)
    else:
        logging.info(
            f"Generating new data, will save to {sample_repeatability_distances_pickle}"
        )
        for subj in subjects:
            res = create_rdm_with_all_trials(
                config,
                subj_list=[subj],
                mask_values=mask_values,
                set_to_take=f"subj_{subj:02d}",
                t_test_threshold=t_thresh,
            )
            results_all.update(res)

        with open(sample_repeatability_distances_pickle, "wb") as f:
            pickle.dump(results_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Compute true within-image distances
    logging.info("Computing true within-image distances")
    true_recs = []
    pval_recs = []

    for subj in tqdm(subjects, desc="Processing subjects"):
        subj_res = results_all.get(subj, {})

        for mv in mask_values:
            for distance_space in ["rdm", "mds"]:
                if mv not in subj_res:
                    continue

                # Get distance matrix for current space
                if distance_space == "mds":
                    mds_coords = subj_res[mv]["mds"]
                    D_full = squareform(pdist(mds_coords, metric="euclidean"))
                elif distance_space == "rdm":
                    D_full = squareform(subj_res[mv]["rdm"])
                else:
                    raise ValueError(f"Unknown distance space: {distance_space}")

                meta = subj_res[mv]["meta"]
                labels = np.array([img for img, _ in meta])

                # Process each unique stimulus
                for stim in tqdm(np.unique(labels), leave=False):
                    idx = np.where(labels == stim)[0]
                    if idx.size < 2:
                        continue  # Need at least 2 repeats

                    # Compute observed within-stimulus distances
                    D_sub = D_full[np.ix_(idx, idx)]
                    iu = np.triu_indices(len(idx), k=1)
                    obs_dists = D_sub[iu]
                    obs_mean = obs_dists.mean()

                    # Record all true pairwise distances
                    for d in obs_dists:
                        true_recs.append(
                            {
                                "subject": subj,
                                "mask": mv,
                                "distance_space": distance_space,
                                "stimulus": stim,
                                "distance": d,
                                "type": "true_distribution",
                            }
                        )

                    # Generate null distribution of mean distances
                    perm_means = np.empty(n_perm)
                    for b in range(n_perm):
                        perm_labels = rng.permutation(labels)
                        perm_idx = np.where(perm_labels == stim)[0]
                        if perm_idx.size < 2:
                            raise RuntimeError(
                                "Permutation resulted in insufficient repeats"
                            )

                        Dp = D_full[np.ix_(perm_idx, perm_idx)]
                        iu_p = np.triu_indices(len(perm_idx), k=1)
                        perm_means[b] = Dp[iu_p].mean()

                    # Compute one-sided p-value: obs_mean ≤ perm_mean
                    t_, p_val = ttest_1samp(perm_means, obs_mean, alternative="greater")

                    pval_recs.append(
                        {
                            "subject": subj,
                            "mask": mv,
                            "distance_space": distance_space,
                            "stimulus": stim,
                            "obs_mean": obs_mean,
                            "p_value": p_val,
                        }
                    )

    logging.info(f"Generated {len(true_recs)} true distance measurements")

    # Build null distribution by permuting labels
    logging.info("Generating null distribution")
    null_recs = []

    for subj in tqdm(subjects, desc="Generating null distances"):
        subj_res = results_all.get(subj, {})

        for mv in mask_values:
            if mv not in subj_res:
                continue

            for distance_space in ["rdm", "mds"]:
                # Get distance matrix
                if distance_space == "mds":
                    mds_coords = subj_res[mv]["mds"]
                    D_full = squareform(pdist(mds_coords, metric="euclidean"))
                elif distance_space == "rdm":
                    D_full = squareform(subj_res[mv]["rdm"])
                else:
                    raise ValueError(f"Unknown distance space: {distance_space}")

                meta = subj_res[mv]["meta"]
                labels = np.array([img for img, _ in meta])

                iu = np.triu_indices_from(D_full, k=1)
                all_d = D_full[iu]  # All pairwise distances

                # Generate multiple permutations
                for _ in range(10):
                    perm_labels = rng.permutation(labels)
                    same_p = perm_labels[iu[0]] == perm_labels[iu[1]]
                    d_within_p = all_d[same_p]

                    for d in d_within_p:
                        null_recs.append(
                            {
                                "subject": subj,
                                "mask": mv,
                                "distance": d,
                                "type": "null_distribution",
                                "distance_space": distance_space,
                            }
                        )

    # Save results
    df_true = pd.DataFrame(true_recs)
    df_null = pd.DataFrame(null_recs)
    df_pval_recs = pd.DataFrame(pval_recs)

    df = pd.concat([df_true, df_null], ignore_index=True)
    df.to_excel(sample_repeatability_distances_excel, index=False)
    df_pval_recs.to_excel("data/output/pval_recs.xlsx", index=False)

    logging.info(
        f"Distance analysis complete. Results saved to {sample_repeatability_distances_excel}"
    )


# =========================================================================
# CORRELATION ANALYSIS FUNCTIONS
# =========================================================================


def analyze_within_subjects(
    config: Configuration,
    sets: Iterable[str],
    subjects: Iterable[int],
    n_masks: int,
    n_randomizations: int,
    mask_mode: str = "single",
    hemispheres: List[str] = ["both", "lh", "rh"],
    n_permutations: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze within-subject repeatability by correlating RDMs and MDS configurations.

    For each set/subject/mask/hemisphere combination:
    - Load up to 3 RDM samples, compute all pairwise Pearson correlations
    - Load up to 3 MDS samples, align via Procrustes, then correlate distances

    Args:
        config: Configuration object
        sets: Iterable of dataset identifiers
        subjects: Iterable of subject IDs
        n_masks: Number of ROI masks to analyze
        n_randomizations: Number of randomization iterations
        mask_mode: Mask processing mode ("single" or "averaged")
        hemispheres: List of hemispheres to analyze
        n_permutations: Number of permutations for null distributions

    Returns:
        tuple: (observed_correlations_df, null_correlations_df)
    """
    logging.info("Starting within-subject correlation analysis")

    records = []
    null_rec = []

    for set_path in sets:
        # Update configuration for current dataset
        config.directories.rdm_dir = f"data/rdm/{set_path}"
        config.directories.mds_dir = f"data/mds/{set_path}"

        for subj in tqdm(subjects, desc="Processing subjects"):
            subj_str = f"subj_{subj:02d}"

            for mask_value in range(1, n_masks + 1):
                for hemi in hemispheres:
                    # Process RDM samples
                    rdm_samples = {}
                    for randomize_offset in range(n_randomizations):
                        for idx in (0, 1, 2):
                            fn = (
                                f"mask_{mask_value}_{mask_mode}"
                                f"_sample_{idx}_{hemi}_rand_{randomize_offset}_rdm.npy"
                            )
                            p = os.path.join(
                                config.directories.rdm_dir, subj_str, subj_str, fn
                            )
                            if os.path.exists(p):
                                try:
                                    rdm_samples[idx] = np.load(p)
                                except Exception as e:
                                    logging.warning(f"Couldn't load {p}: {e}")

                        # Compute pairwise RDM correlations
                        if len(rdm_samples) >= 2:
                            for i, j in combinations(sorted(rdm_samples), 2):
                                x, y = rdm_samples[i], rdm_samples[j]

                                # Observed correlations
                                r_obs_pearson, p_obs_pearson = pearsonr(x, y)
                                r_obs_spearman, p_obs_spearman = spearmanr(x, y)

                                base = {
                                    "set": set_path,
                                    "subject": subj,
                                    "mask": mask_value,
                                    "hemi": hemi,
                                    "sample_i": i,
                                    "sample_j": j,
                                    "randomization": randomize_offset,
                                }

                                records += [
                                    {
                                        **base,
                                        "type": "RDM",
                                        "r": r_obs_pearson,
                                        "p": p_obs_pearson,
                                        "disparity": np.nan,
                                        "correlation_type": "pearson",
                                    },
                                    {
                                        **base,
                                        "type": "RDM",
                                        "r": r_obs_spearman,
                                        "p": p_obs_spearman,
                                        "disparity": np.nan,
                                        "correlation_type": "spearman",
                                    },
                                ]

                                # Generate null distributions
                                for perm in range(n_permutations):
                                    y_shuf = np.random.permutation(y)

                                    # Pearson null
                                    r_null_p, p_null_p = pearsonr(x, y_shuf)
                                    null_rec.append(
                                        {
                                            **base,
                                            "type": "RDM",
                                            "perm": perm,
                                            "r": r_null_p,
                                            "p": p_null_p,
                                            "disparity": np.nan,
                                            "correlation_type": "pearson",
                                        }
                                    )

                                    # Spearman null
                                    r_null_s, p_null_s = spearmanr(x, y_shuf)
                                    null_rec.append(
                                        {
                                            **base,
                                            "type": "RDM",
                                            "perm": perm,
                                            "r": r_null_s,
                                            "p": p_null_s,
                                            "disparity": np.nan,
                                            "correlation_type": "spearman",
                                        }
                                    )

                    # Process MDS samples
                    mds_samples = {}
                    for randomize_offset in range(n_randomizations):
                        for idx in (0, 1, 2):
                            fn = (
                                f"mask_{mask_value}_{mask_mode}"
                                f"_sample_{idx}_{hemi}_rand_{randomize_offset}_mds.npy"
                            )
                            p = os.path.join(
                                config.directories.mds_dir, subj_str, subj_str, fn
                            )
                            if os.path.exists(p):
                                try:
                                    mds_samples[idx] = np.load(p)
                                except Exception as e:
                                    logging.warning(f"Couldn't load {p}: {e}")
                            else:
                                logging.debug(f"File not found: {p}")

                        # Compute pairwise MDS correlations via Procrustes alignment
                        if len(mds_samples) >= 2:
                            for i, j in combinations(sorted(mds_samples), 2):
                                base = {
                                    "set": set_path,
                                    "subject": subj,
                                    "mask": mask_value,
                                    "hemi": hemi,
                                    "type": "MDS",
                                    "sample_i": i,
                                    "sample_j": j,
                                    "randomization": randomize_offset,
                                }

                                # Observed Procrustes alignment and distance correlation
                                m1, m2_aligned, disp_obs = procrustes(
                                    mds_samples[i], mds_samples[j]
                                )
                                d1 = squareform(pdist(m1))
                                d2 = squareform(pdist(m2_aligned))

                                # Pearson correlation
                                r_obs_p, p_obs_p = pearsonr(d1.ravel(), d2.ravel())
                                records.append(
                                    {
                                        **base,
                                        "r": r_obs_p,
                                        "p": p_obs_p,
                                        "disparity": disp_obs,
                                        "correlation_type": "pearson",
                                    }
                                )

                                # Spearman correlation
                                r_obs_s, p_obs_s = spearmanr(d1.ravel(), d2.ravel())
                                records.append(
                                    {
                                        **base,
                                        "r": r_obs_s,
                                        "p": p_obs_s,
                                        "disparity": disp_obs,
                                        "correlation_type": "spearman",
                                    }
                                )

                                # Generate null distributions by permuting labels
                                for perm in range(n_permutations):
                                    # Shuffle rows (labels) of sample j
                                    idx = np.random.permutation(mds_samples[j].shape[0])
                                    m2_shuf = mds_samples[j][idx]

                                    # Procrustes alignment on shuffled data
                                    m1n, m2n_aligned, disp_null = procrustes(
                                        mds_samples[i], m2_shuf
                                    )
                                    d1n = squareform(pdist(m1n))
                                    d2n = squareform(pdist(m2n_aligned))

                                    # Pearson null
                                    r_null_p, p_null_p = pearsonr(
                                        d1n.ravel(), d2n.ravel()
                                    )
                                    null_rec.append(
                                        {
                                            **base,
                                            "perm": perm,
                                            "r": r_null_p,
                                            "p": p_null_p,
                                            "disparity": disp_null,
                                            "correlation_type": "pearson",
                                        }
                                    )

                                    # Spearman null
                                    r_null_s, p_null_s = spearmanr(
                                        d1n.ravel(), d2n.ravel()
                                    )
                                    null_rec.append(
                                        {
                                            **base,
                                            "perm": perm,
                                            "r": r_null_s,
                                            "p": p_null_s,
                                            "disparity": disp_null,
                                            "correlation_type": "spearman",
                                        }
                                    )

    df_obs = pd.DataFrame.from_records(records)
    df_null = pd.DataFrame.from_records(null_rec)

    logging.info("Within-subject correlation analysis complete")
    return df_obs, df_null


# =========================================================================
# MAIN EXECUTION
# =========================================================================


def main():
    """Main execution function for sample repeatability analysis."""
    logging.info("Starting sample repeatability analysis pipeline")

    # Example usage (commented out for safety)
    pass

    # config = load_config("config.yaml")
    # subjects = list(range(1, 9))
    # n_randomizations = 4
    # n_masks = 9
    # mask_mode = "single"
    # t_threshold = 2.5

    # # Run distance analysis
    # sample_repeatability_distances_data(config, subjects, t_thresh=t_threshold)

    # # Configure for sample repeatability analysis
    # config.directories.mds_dir = "data/mds/sample_repeatability"
    # config.directories.rdm_dir = "data/rdm/sample_repeatability"
    # sets = ["sample_repeatability"]

    # # Generate RDMs for all subjects
    # for subj_id in subjects:
    #     logging.info(f"Generating RDMs for subject {subj_id:02d}")
    #     create_all_rdms(
    #         config,
    #         subj_id,
    #         n_samples=3,
    #         n_randomizations=n_randomizations,
    #         t_thresh=t_threshold,
    #     )

    # # Perform correlation analysis
    # logging.info("Running correlation analysis")
    # df_obs, df_null = analyze_within_subjects(
    #     config,
    #     sets=sets,
    #     subjects=subjects,
    #     n_masks=n_masks,
    #     n_randomizations=n_randomizations,
    #     mask_mode=mask_mode,
    # )

    # # Save results
    # df_obs.to_excel(sample_repeatability_correlations_true_excel, index=False)
    # df_null.to_excel(sample_repeatability_correlations_null_excel, index=False)
    # logging.info("Sample repeatability analysis pipeline complete")


if __name__ == "__main__":
    main()
