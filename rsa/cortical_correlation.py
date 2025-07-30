#!/usr/bin/env python3
"""
distances_analysis.py

Compute:
  1) MDS‐based voxel distances
  2) Geodesic (cortical) distances on the surface mesh
  3) Spearman correlations between the two distance matrices

Usage:
  distances_analysis.py               # run all 3 steps
  distances_analysis.py --mds         # only MDS distances
  distances_analysis.py --cortical    # only cortical distances
  distances_analysis.py --corr        # only correlation step
  distances_analysis.py --subjects 1 2 3   # restrict to specific subjects
"""

# Standard library imports
import os
import random
import logging

# Third-party imports
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.stats
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
from nibabel import freesurfer
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids  # sklearn ≥1.2; otherwise use sklearn_extra.cluster.KMedoids

# Local imports
from utils.config import load_config, Configuration
from utils.utils import retrieve_roi_mask_extended

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =========================================================================
# DISTANCE CALCULATION UTILITIES
# =========================================================================

def bhattacharyya_distance(mu1, S1, mu2, S2):
    """
    Compute Bhattacharyya distance between two Gaussians N(mu1, S1) and N(mu2, S2).
    
    Args:
        mu1, mu2: Mean vectors of the two Gaussians
        S1, S2: Covariance matrices of the two Gaussians
        
    Returns:
        float: Bhattacharyya distance
    """
    S_avg = (S1 + S2) / 2.0
    inv_Savg = np.linalg.inv(S_avg)
    diff = mu1 - mu2

    term1 = 0.125 * diff.T @ inv_Savg @ diff
    term2 = 0.5 * np.log(
        np.linalg.det(S_avg) /
        np.sqrt(np.linalg.det(S1) * np.linalg.det(S2))
    )
    return term1 + term2


def build_mesh_graph(vertices: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    """
    Build CSR adjacency matrix weighted by Euclidean length of edges.
    
    Args:
        vertices: Mesh vertices array
        faces: Mesh faces array
        
    Returns:
        sp.csr_matrix: Sparse adjacency matrix with edge weights
    """
    a, b, c = faces.T
    row = np.hstack([a, b, c, b, c, a])
    col = np.hstack([b, c, a, a, b, c])
    data = np.linalg.norm(vertices[row] - vertices[col], axis=1)
    N = vertices.shape[0]
    return sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()


# =========================================================================
# MDS DISTANCE COMPUTATION
# =========================================================================

def compute_mds_distances(
    config: Configuration,
    subjects: list[int],
    t_test_threshold: float,
    rois: list[int] = list(range(1, 10)),
    sigma_threshold: float = 5
) -> None:
    """
    Compute Euclidean MDS distances and Bhattacharyya distances
    between fitted 2D Gaussians (normalized, ignoring amp & offset).
    
    Args:
        config: Configuration object containing directory paths
        subjects: List of subject IDs to process
        t_test_threshold: Threshold for t-test ROI masking
        rois: List of ROI indices to process
        sigma_threshold: Threshold for filtering Gaussian fits by sigma value
    """
    for sub in tqdm(subjects, desc=f"Computing MDS distances ({config.directories.gaussian_fit_results_dir})"):
        subj_str = f"subj_{sub:02d}"
        fits_base = os.path.join(
            config.directories.gaussian_fit_results_dir,
            subj_str, f"subj{sub:02d}"
        )
        mask_dir = os.path.join(config.directories.t_test_roi_dir, subj_str)
        
        # Retrieve ROI mask for this subject
        mask, n_lh = retrieve_roi_mask_extended(config, sub, f"subj_{sub:02d}", True, t_test_threshold=t_test_threshold)
        mask_lh = mask[:n_lh]
        mask_rh = mask[n_lh:]
        mask_roi_values = np.unique(mask)

        for roi in rois:
            if roi not in mask_roi_values:
                continue

            # Load fitted Gaussian parameters for this ROI
            fits_file = os.path.join(
                fits_base, f"fitted_voxels_mask_{roi}.xlsx"
            )
            if not os.path.isfile(fits_file):
                logging.info(f"[MDS] subj{sub:02d}: ROI {roi} fits missing → skip")
                continue

            df = pd.read_excel(fits_file)
            
            # Validate required columns for 2D Gaussian fitting
            needed = {"original_index", "x0", "y0"}
            if not (needed <= set(df.columns)):
                logging.info(f"[MDS] subj{sub:02d}: ROI {roi} bad columns → skip")
                raise Exception(f"[MDS] subj{sub:02d}: ROI {roi} bad columns → skip")

            # Filter data: both hemispheres and reasonable sigma values
            df = df[df["mds_hemi"] == "both"]
            df = df[df["sigma"] < sigma_threshold]

            # Extract relevant data
            idx_all = df["original_index"].to_numpy(int)
            coords = df[["x0", "y0"]].to_numpy(float)
            hemis = np.where(idx_all < n_lh, "lh", "rh")

            # Create output directory
            out_subdir = os.path.join(
                config.directories.distances_dir,
                "mds_distances", subj_str
            )
            os.makedirs(out_subdir, exist_ok=True)

            # Precompute each voxel's mean (mu) and covariance (Sigma)
            mus = []
            Sigmas = []
            for _, row in df.iterrows():
                mu = np.array([row["x0"], row["y0"]])
                s = row["sigma"]
                S = np.eye(2) * (s * s)  # isotropic covariance
                mus.append(mu)
                Sigmas.append(S)
            
            mus = np.stack(mus, axis=0)      # shape (N_all, 2)
            Sigmas = np.stack(Sigmas, axis=0) # shape (N_all, 2, 2)

            # Process each hemisphere separately
            for hemi in ("lh", "rh"):
                mask = hemis == hemi
                if mask.sum() < 2:
                    logging.info(f"[MDS] subj{sub:02d} ROI {roi} {hemi}: <2 voxels → skip")
                    continue

                # Select data for this hemisphere
                sel_idx = idx_all[mask]
                sel_coords = coords[mask]
                sel_mus = mus[mask]       # shape (N, 2)
                sel_S = Sigmas[mask]      # shape (N, 2, 2)
                N = sel_coords.shape[0]

                # 1) Euclidean MDS distance (unchanged)
                D = squareform(pdist(sel_coords, "euclidean")).astype(np.float32)
                np.savez_compressed(
                    os.path.join(out_subdir, f"{hemi}.dists.roi_{roi}.npz"),
                    distances=D,
                    indices=sel_idx
                )

                # 2) Bhattacharyya distance on isotropic Gaussians — vectorized version
                # Extract the scalar sigmas from the isotropic covariance matrices
                sigma2 = sel_S[:, 0, 0]       # σ_i^2 for each voxel
                sigmas = np.sqrt(sigma2)      # σ_i

                # (a) pairwise squared Euclidean distance between means
                d2 = squareform(pdist(sel_mus, metric="sqeuclidean"))   # shape (N, N)

                # (b) build σ²-sum and σ-product matrices
                S_sum = sigma2[:, None] + sigma2[None, :]               # shape (N, N)
                S_prod = sigmas[:, None] * sigmas[None, :]              # shape (N, N)

                # (c) apply the closed-form Bhattacharyya distance for isotropic Gaussians
                B = d2 / (4.0 * S_sum) + np.log(S_sum / (2.0 * S_prod))

                # Zero out the diagonal
                np.fill_diagonal(B, 0.0)

                # Save the Bhattacharyya distance matrix
                np.savez_compressed(
                    os.path.join(out_subdir, f"{hemi}.bhatt.roi_{roi}.npz"),
                    distances=B.astype(np.float32),
                    indices=sel_idx
                )

                logging.info(
                    f"[MDS] subj{sub:02d} ROI {roi} {hemi}: "
                    f"saved MDS‐Euc {D.shape}, Bhatt {B.shape}"
                )

    logging.info("[MDS] Done.")


# =========================================================================
# CORTICAL DISTANCE COMPUTATION
# =========================================================================

def compute_cortical_distances(
    config: Configuration,
    subjects: list[int],
    rois: list[int] = list(range(1, 10)),
):
    """
    Equivalent of the MATLAB meshes_and_distances.m (GIBBON‐based Dijkstra) in Python.
    For each subject and each hemisphere and each ROI,
    compute geodesic distances among ROI vertices via SciPy's Dijkstra.
    
    Args:
        config: Configuration object containing directory paths
        subjects: List of subject IDs to process
        rois: List of ROI indices to process
    """
    hemis = ["lh", "rh"]
    logging.info(f"Processing ROIs: {rois}")

    for sub in subjects:
        subj_dir = f"subj_{sub:02d}"
        
        for hemi in hemis:
            # Get left hemisphere size for right hemisphere offset calculation
            if hemi == "rh":
                n_lh = (
                    nib.load(
                        os.path.join(
                            config.directories.t_test_roi_dir,
                            subj_dir,
                            f"lh.subj{sub:02d}.final.mgz",
                        )
                    )
                    .get_fdata()
                    .squeeze()
                    .astype(int)
                    .size
                )

            # 1) Load the ROI file for this hemisphere
            if sub in (6, 8):
                roi_mgz = os.path.join(
                    config.directories.t_test_roi_dir,
                    subj_dir,
                    f"{hemi}.subj{sub:02d}.final.mgz",
                )
            else:
                roi_mgz = os.path.join(
                    config.directories.t_test_roi_dir,
                    subj_dir,
                    f"{hemi}.subj{sub:02d}.final.mgz",
                )
            hhroivals = nib.load(roi_mgz).get_fdata().squeeze().astype(int)

            # 2) Load surface mesh and build adjacency graph
            surf_file = os.path.join(
                config.nsd_data.freesurfer_dir,
                f"subj{sub:02d}",
                "surf",
                f"{hemi}.white" if sub in (6, 8) else f"{hemi}.white",
            )
            vertices, faces = freesurfer.read_geometry(surf_file)
            graph = build_mesh_graph(vertices, faces)

            # 3) Process each ROI
            distances_dir = os.path.join(
                config.directories.distances_dir, "distances_cortical", subj_dir
            )
            os.makedirs(distances_dir, exist_ok=True)

            for roi_label in rois:
                out_file = os.path.join(
                    distances_dir, f"{hemi}.dists.roi_{roi_label}.npz"
                )
                
                # Skip if already computed
                if os.path.exists(out_file):
                    logging.info(
                        f"{subj_dir}:{hemi}.roi_{roi_label} → already done, skipping."
                    )
                    continue

                # Find vertices belonging to this ROI
                mask = np.isin(hhroivals, roi_label)
                roi_indices = np.flatnonzero(mask)
                k = len(roi_indices)
                
                if k == 0:
                    logging.info(f"{subj_dir}:{hemi}.roi_{roi_label} → no vertices, skipping.")
                    continue

                logging.info(
                    f"Computing distances for {subj_dir}:{hemi}.roi_{roi_label} (k={k}) …"
                )
                
                # Compute all-pairs shortest paths for ROI vertices
                d_roi = np.zeros((k, k), dtype=np.float32)
                for i, src in enumerate(
                    tqdm(roi_indices, desc=f"{hemi} ROI {roi_label}")
                ):
                    dist = dijkstra(csgraph=graph, directed=False, indices=src)
                    d_roi[:, i] = dist[roi_indices]

                # Symmetrize the distance matrix
                d_roi = np.minimum(d_roi, d_roi.T)

                # Adjust indices for right hemisphere (add left hemisphere size)
                if hemi == "rh":
                    roi_indices += n_lh
                    
                # Save distance matrix and indices
                np.savez_compressed(out_file, distances=d_roi, indices=roi_indices)
                logging.info(f"   → saved {k}×{k} to {out_file}")

    logging.info("All cortical distance computations done.")


# =========================================================================
# CORRELATION ANALYSIS
# =========================================================================

def spearman_trials(
    a: np.ndarray,
    b: np.ndarray,
    fraction: float = 1.0,
    trials: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Spearman rho & p‐value for each trial.
    
    Args:
        a, b: Input arrays for correlation
        fraction: Fraction of data to use per trial (for bootstrapping)
        trials: Number of trials to run
        shuffle: If True, shuffle `a` each trial to build a null distribution
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (rhos, pvals) arrays of shape (trials,)
    """
    if seed is not None:
        random.seed(seed)

    M = a.size
    # Number of pairs per trial (full if fraction=1)
    n = max(1, int(round(M * fraction)))
    idxs = list(range(M))

    rhos = np.empty(trials, dtype=float)
    pvals = np.empty(trials, dtype=float)

    for t in range(trials):
        # Maybe shuffle a
        if shuffle:
            a_t = a.copy()
            random.shuffle(a_t)
        else:
            a_t = a

        # Maybe subsample
        if fraction < 1.0:
            sel = random.sample(idxs, n)
            x, y = a_t[sel], b[sel]
        else:
            x, y = a_t, b

        r = scipy.stats.spearmanr(x, y)
        rho = r.correlation if hasattr(r, "correlation") else r.statistic
        rhos[t] = rho
        pvals[t] = r.pvalue

    return rhos, pvals


def compute_correlations(
    config: Configuration,
    subjects: list[int],
    hemis: list[str] = ["lh", "rh"],
    rois: list[int] = list(range(1, 10)),
    downsample_frac: float = 1.0 / (1.8**3),
    trials: int = 1000,
    n_seeds: int = 5,  # Currently unused
    k_nn: int = 100,   # Currently unused
) -> None:
    """
    Compute & save trial‐level Spearman correlations for full/bootstrap/null conditions.
    
    Args:
        config: Configuration object containing directory paths
        subjects: List of subject IDs to process
        hemis: List of hemispheres to process
        rois: List of ROI indices to process
        downsample_frac: Fraction of data to use in bootstrap/null trials
        trials: Number of bootstrap/null trials to run
        n_seeds: Number of clustering seeds (unused in current implementation)
        k_nn: K for nearest neighbor clustering (unused in current implementation)
    """
    # Set up input/output paths
    mds_base = os.path.join(config.directories.distances_dir, "mds_distances")
    cort_base = os.path.join(config.directories.distances_dir, "distances_cortical")

    out_long_pkl = os.path.join(
        config.directories.output_dir,
        config.saved_results_paths.spearman_correlations_pickle,
    )
    out_summary_xlsx = os.path.join(
        config.directories.output_dir,
        config.saved_results_paths.sample_repeatability_distances_excel,
    )

    long_rows = []

    # Process each subject/hemisphere/ROI combination
    for sub in tqdm(subjects, desc="Subjects"):
        sub_str = f"subj_{sub:02d}"
        
        for hemi in hemis:
            for roi in tqdm(rois, desc=f"ROIS – hemi={hemi}"):
                # Define file paths for distance matrices
                mds_euc_f = os.path.join(mds_base, sub_str, f"{hemi}.dists.roi_{roi}.npz")
                mds_bhatt_f = os.path.join(mds_base, sub_str, f"{hemi}.bhatt.roi_{roi}.npz")
                cort_f = os.path.join(cort_base, sub_str, f"{hemi}.dists.roi_{roi}.npz")

                # Skip if any required files are missing
                if not os.path.isfile(mds_euc_f) or not os.path.isfile(mds_bhatt_f) or not os.path.isfile(cort_f):
                    continue

                # Load distance matrices and indices
                md_euc = np.load(mds_euc_f)
                md_bhatt = np.load(mds_bhatt_f)
                cd = np.load(cort_f)
                
                ids_m, ids_m_bhatt, ids_c = md_euc["indices"], md_bhatt["indices"], cd["indices"]
                Dm, Dm_bhatt, Dc = md_euc["distances"], md_bhatt["distances"], cd["distances"]

                # Find overlapping indices between MDS and cortical distances
                mask_m = np.isin(ids_m, ids_c)          # Euclidean MDS mask
                mask_m_bhatt = np.isin(ids_m_bhatt, ids_c)  # Bhattacharyya MDS mask

                # Validate sufficient overlap
                if mask_m.sum() < 2 and mask_m_bhatt.sum() < 2:
                    continue
                elif mask_m.sum() >= 2 and mask_m_bhatt.sum() >= 2:
                    pass
                else:
                    raise Exception(f"[CORR] subj{sub:02d}: ROI {roi} bad indices → skip")

                # Create index mappings for cortical distances
                ic = {int(v): i for i, v in enumerate(ids_c)}
                sel_c = [ic[int(v)] for v in ids_m[mask_m]]
                sel_c_bhatt = [ic[int(v)] for v in ids_m_bhatt[mask_m_bhatt]]

                # Extract overlapping distance submatrices
                Dm2 = Dm[np.ix_(mask_m, mask_m)]
                Dm2_bhatt = Dm_bhatt[np.ix_(mask_m_bhatt, mask_m_bhatt)]
                Dc2 = Dc[np.ix_(sel_c, sel_c)]
                Dc2_bhatt = Dc[np.ix_(sel_c_bhatt, sel_c_bhatt)]

                # Flatten upper triangular parts for correlation analysis
                triu = np.triu_indices(Dm2.shape[0], k=1)
                triu_bhatt = np.triu_indices(Dm2_bhatt.shape[0], k=1)
                a_flat, b_flat = Dm2[triu], Dc2[triu]
                a_flat_bhatt, b_flat_bhatt = Dm2_bhatt[triu_bhatt], Dc2_bhatt[triu_bhatt]

                n_pairs = a_flat.size

                # === FULL CORRELATION (single trial, no shuffle/subsample) ===
                rhos, pvals = spearman_trials(
                    a_flat, b_flat, fraction=1.0, trials=1, shuffle=False, seed=0
                )
                rhos_bhatt, pvals_bhatt = spearman_trials(
                    a_flat_bhatt, b_flat_bhatt, fraction=1.0, trials=1, shuffle=False, seed=0
                )

                # Store Euclidean results
                for t, (rho, p) in enumerate(zip(rhos, pvals)):
                    long_rows.append({
                        "subject": sub_str,
                        "hemisphere": hemi,
                        "roi": roi,
                        "method": "full",
                        "trial": t,
                        "n_pairs": n_pairs,
                        "rho": rho,
                        "pval": p,
                        "distance_metric": "euclidean",
                    })

                # Store Bhattacharyya results
                for t, (rho, p) in enumerate(zip(rhos_bhatt, pvals_bhatt)):
                    long_rows.append({
                        "subject": sub_str,
                        "hemisphere": hemi,
                        "roi": roi,
                        "method": "full",
                        "trial": t,
                        "n_pairs": n_pairs,
                        "rho": rho,
                        "pval": p,
                        "distance_metric": "bhattacharyya",
                    })

                # === BOOTSTRAP DOWNSAMPLE ===
                rhos, pvals = spearman_trials(
                    a_flat, b_flat, fraction=downsample_frac, trials=trials, shuffle=False, seed=0
                )
                rhos_bhatt, pvals_bhatt = spearman_trials(
                    a_flat_bhatt, b_flat_bhatt, fraction=downsample_frac, trials=trials, shuffle=False, seed=0
                )

                # Store Euclidean bootstrap results
                for t, (rho, p) in enumerate(zip(rhos, pvals)):
                    long_rows.append({
                        "subject": sub_str,
                        "hemisphere": hemi,
                        "roi": roi,
                        "method": "bootstrap",
                        "trial": t,
                        "n_pairs": round(n_pairs * downsample_frac),
                        "rho": rho,
                        "pval": p,
                        "distance_metric": "euclidean",
                    })

                # Store Bhattacharyya bootstrap results
                for t, (rho, p) in enumerate(zip(rhos_bhatt, pvals_bhatt)):
                    long_rows.append({
                        "subject": sub_str,
                        "hemisphere": hemi,
                        "roi": roi,
                        "method": "bootstrap",
                        "trial": t,
                        "n_pairs": round(n_pairs * downsample_frac),
                        "rho": rho,
                        "pval": p,
                        "distance_metric": "bhattacharyya",
                    })

                # === NULL DISTRIBUTION (bootstrap + shuffle) ===
                rhos, pvals = spearman_trials(
                    a_flat, b_flat, fraction=downsample_frac, trials=trials, shuffle=True, seed=0
                )
                rhos_bhatt, pvals_bhatt = spearman_trials(
                    a_flat_bhatt, b_flat_bhatt, fraction=downsample_frac, trials=trials, shuffle=True, seed=0
                )

                # Store Euclidean null results
                for t, (rho, p) in enumerate(zip(rhos, pvals)):
                    long_rows.append({
                        "subject": sub_str,
                        "hemisphere": hemi,
                        "roi": roi,
                        "method": "null",
                        "trial": t,
                        "n_pairs": round(n_pairs * downsample_frac),
                        "rho": rho,
                        "pval": p,
                        "distance_metric": "euclidean",
                    })

                # Store Bhattacharyya null results
                for t, (rho, p) in enumerate(zip(rhos_bhatt, pvals_bhatt)):
                    long_rows.append({
                        "subject": sub_str,
                        "hemisphere": hemi,
                        "roi": roi,
                        "method": "null",
                        "trial": t,
                        "n_pairs": round(n_pairs * downsample_frac),
                        "rho": rho,
                        "pval": p,
                        "distance_metric": "bhattacharyya",
                    })

                # Save incrementally after each ROI (pickle is fast)
                pd.DataFrame(long_rows).to_pickle(out_long_pkl)

    # Save final results
    pd.DataFrame(long_rows).to_pickle(out_long_pkl)
    logging.info(f"[CORR] trial‐level data → {out_long_pkl}")


# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    """Main function to run all distance analysis steps."""
    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    cfg = load_config("config.yaml")

    # Run all three analysis steps
    logging.info("Starting cortical distance computation...")
    compute_cortical_distances(cfg, subjects)

    logging.info("Starting MDS distance computation...")
    compute_mds_distances(cfg, subjects)

    logging.info("Starting correlation analysis...")
    compute_correlations(cfg, subjects, trials=100)

    logging.info("All analyses complete.")


if __name__ == "__main__":
    main()