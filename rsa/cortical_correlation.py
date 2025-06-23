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
import os
import random

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import dijkstra
import scipy.sparse as sp
import scipy.stats
from nibabel import freesurfer
from tqdm import tqdm

from utils.config import load_config, Configuration

from gaussian.fit_gaussian import fit_gaussian_params


def compute_mds_distances(
    config: Configuration,
    subjects: list[int],
    rois: list[int] = list(range(1, 10)),
) -> None:
    """Compute pairwise Euclidean distances in MDS‐space for each ROI & hemisphere."""
    for sub in subjects:
        subj_str = f"subj_{sub:02d}"
        fits_base = os.path.join(
            config.directories.gaussian_fit_results_dir, subj_str, f"subj{sub:02d}"
        )
        mask_dir = os.path.join(config.directories.t_test_roi_dir, subj_str)
        try:
            mask_lh = (
                nib.load(os.path.join(mask_dir, f"lh.subj{sub:02d}.testrois.mgz"))
                .get_fdata()
                .astype(int)
            )
            mask_rh = (
                nib.load(os.path.join(mask_dir, f"rh.subj{sub:02d}.testrois.mgz"))
                .get_fdata()
                .astype(int)
            )
        except FileNotFoundError:
            print(f"[MDS] subj{sub:02d}: missing mask → skipping.")
            continue
        n_lh, n_rh = mask_lh.shape[0], mask_rh.shape[0]

        for roi in rois:
            fits_file = os.path.join(fits_base, f"fitted_voxels_mask_{roi}.xlsx")
            if not os.path.isfile(fits_file):
                print(f"[MDS] subj{sub:02d}: ROI {roi} fits missing → skip")
                continue

            df = pd.read_excel(fits_file)
            if not {"original_index", "x0", "y0"}.issubset(df.columns):
                print(f"[MDS] subj{sub:02d}: ROI {roi} bad columns → skip")
                continue

            df = df[df["mds_hemi"] == "both"]

            idx_all = df["original_index"].to_numpy(int)
            coords = df[["x0", "y0"]].to_numpy(float)
            hemis = np.where(idx_all < n_lh, "lh", "rh")

            out_subdir = os.path.join(
                config.directories.distances_dir, "mds_distances", subj_str
            )
            os.makedirs(out_subdir, exist_ok=True)

            for hemi in ("lh", "rh"):
                mask = hemis == hemi
                if np.sum(mask) < 2:
                    print(f"[MDS] subj{sub:02d} ROI {roi} {hemi}: <2 voxels → skip")
                    continue

                sel_idx = idx_all[mask]
                sel_coords = coords[mask]
                D = squareform(pdist(sel_coords, "euclidean")).astype(np.float32)

                out_file = os.path.join(out_subdir, f"{hemi}.dists.roi_{roi}.npz")
                np.savez_compressed(out_file, distances=D, indices=sel_idx)
                print(f"[MDS] subj{sub:02d} ROI {roi} {hemi}: saved {D.shape}")

    print("[MDS] Done.")


def build_mesh_graph(vertices: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    """CSR adjacency matrix weighted by Euclidean length of edges."""
    a, b, c = faces.T
    row = np.hstack([a, b, c, b, c, a])
    col = np.hstack([b, c, a, a, b, c])
    data = np.linalg.norm(vertices[row] - vertices[col], axis=1)
    N = vertices.shape[0]
    return sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()


# def compute_cortical_distances(
#     config: Configuration,
#     subjects: list[int],
#     rois: list[int] = list(range(1, 10)),
#     # freesurfer_dir: str | None = None
# ) -> None:
#     """Geodesic distances on the white‐matter surface mesh via Dijkstra."""
#     freesurfer_dir = config.nsd_data.freesurfer_dir  # or hard‐code if needed

#     for sub in subjects:
#         subj_str = f"subj_{sub:02d}"
#         mask_dir = os.path.join(config.directories.t_test_roi_dir, subj_str)

#         try:
#             mask_lh = nib.load(os.path.join(mask_dir, f"lh.subj{sub:02d}.testrois.mgz")).get_fdata()
#             mask_rh = nib.load(os.path.join(mask_dir, f"rh.subj{sub:02d}.testrois.mgz")).get_fdata()
#         except FileNotFoundError:
#             print(f"[CORT] subj{sub:02d}: missing ROI masks → skip")
#             continue

#         n_lh, n_rh = mask_lh.shape[0], mask_rh.shape[0]
#         graph_cache: dict[str, sp.csr_matrix] = {}

#         for hemi in ('lh','rh'):
#             # lazy‐load graph per hemisphere
#             if hemi not in graph_cache:
#                 white_fn = os.path.join(freesurfer_dir, subj_str, 'surf', f"{hemi}.white")
#                 verts, faces = freesurfer.read_geometry(white_fn)
#                 graph_cache[hemi] = build_mesh_graph(verts, faces)

#             graph = graph_cache[hemi]
#             n_tot = n_lh if hemi=='lh' else n_rh

#             for roi in rois:
#                 fits = os.path.join(
#                     config.directories.gaussian_fit_results_dir,
#                     subj_str, subj_str, f"fitted_voxels_mask_{roi}.xlsx"
#                 )
#                 if not os.path.isfile(fits):
#                     continue
#                 df = pd.read_excel(fits)
#                 if 'original_index' not in df.columns:
#                     raise RuntimeError(f"Bad fits file: {fits}")

#                 df['hemi'] = df['original_index'].apply(lambda i: 'lh' if i<n_lh else 'rh')
#                 hemi_df = df[df['hemi']==hemi]
#                 if hemi_df.empty:
#                     continue

#                 global_idx = hemi_df['original_index'].to_numpy(int)
#                 local_idx = global_idx if hemi=='lh' else global_idx - n_lh
#                 k = local_idx.size
#                 if k < 2:
#                     continue

#                 out_dir = os.path.join(config.directories.distances_dir, "distances_cortical", subj_str)
#                 os.makedirs(out_dir, exist_ok=True)
#                 out_file = os.path.join(out_dir, f"{hemi}.dists.roi_{roi}.npz")
#                 if os.path.exists(out_file):
#                     print(f"[CORT] {subj_str} {hemi} ROI {roi} exists → skip")
#                     continue

#                 D = np.zeros((k,k), np.float32)
#                 for j, src in enumerate(tqdm(local_idx, desc=f"{subj_str} {hemi} ROI{roi}")):
#                     d = dijkstra(graph, directed=False, indices=src)
#                     D[:,j] = d[local_idx]
#                 D = np.minimum(D, D.T)
#                 np.savez_compressed(out_file, distances=D, indices=local_idx)
#                 print(f"[CORT] {subj_str} {hemi} ROI {roi}: saved {D.shape}")

#     print("[CORT] Done.")


def compute_cortical_distances(
    config: Configuration,
    subjects: list[int],
    rois: list[int] = list(range(1, 10)),
):
    """
    Equivalent of the MATLAB meshes_and_distances.m (GIBBON‐based Dijkstra) in Python.
    For each subject and each hemisphere and each ROI,
    compute geodesic distances among ROI vertices via SciPy's Dijkstra.
    """
    # roi_names = [
    #     'V1', 'V2v', 'V2d', 'V3v', 'V3d',
    #     'hV4', 'VO-1', 'VO-2', 'PHC-1', 'PHC-2',
    #     'LO-1', 'LO-2', 'TO-1', 'TO-2'
    # ]
    hemis = ["lh", "rh"]

    for sub in subjects:
        subj_dir = f"subj_{sub:02d}"
        for hemi in hemis:
            if hemi == "rh":
                n_lh = (
                    nib.load(
                        os.path.join(
                            config.directories.t_test_roi_dir,
                            subj_dir,
                            f"lh.subj{sub:02d}.testrois.mgz",
                        )
                    )
                    .get_fdata()
                    .squeeze()
                    .astype(int)
                    .size
                )

            # --- 1) pick the correct testrois file on disk ---
            if sub in (6, 8):
                roi_mgz = os.path.join(
                    config.directories.t_test_roi_dir,
                    subj_dir,
                    f"{hemi}.subj{sub:02d}.testrois.mgz",
                )
            else:
                roi_mgz = os.path.join(
                    config.directories.t_test_roi_dir,
                    subj_dir,
                    f"{hemi}.subj{sub:02d}.testrois.mgz",
                )
            hhroivals = nib.load(roi_mgz).get_fdata().squeeze().astype(int)

            # --- 2) load surface and build graph once per hemi ---
            surf_file = os.path.join(
                config.nsd_data.freesurfer_dir,
                f"subj{sub:02d}",
                "surf",
                f"{hemi}.white_del" if sub in (6, 8) else f"{hemi}.white",
            )
            vertices, faces = freesurfer.read_geometry(surf_file)
            graph = build_mesh_graph(vertices, faces)

            # --- 3) per‐ROI processing ---
            distances_dir = os.path.join(
                config.directories.distances_dir, "distances_cortical", subj_dir
            )
            os.makedirs(distances_dir, exist_ok=True)

            for roi_label in rois:
                # name it by the first label if you like, or join them
                out_file = os.path.join(
                    distances_dir, f"{hemi}.dists.roi_{roi_label}.npz"
                )
                if os.path.exists(out_file):
                    print(
                        f"{subj_dir}:{hemi}.roi_{roi_label} → already done, skipping."
                    )
                    continue

                # which vertices belong to this ROI?
                mask = np.isin(hhroivals, roi_label)
                roi_indices = np.flatnonzero(mask)
                k = len(roi_indices)
                if k == 0:
                    print(f"{subj_dir}:{hemi}.roi_{roi_label} → no vertices, skipping.")
                    continue

                print(
                    f"Computing distances for {subj_dir}:{hemi}.roi_{roi_label} (k={k}) …"
                )
                d_roi = np.zeros((k, k), dtype=np.float32)
                for i, src in enumerate(
                    tqdm(roi_indices, desc=f"{hemi} ROI {roi_label}")
                ):
                    dist = dijkstra(csgraph=graph, directed=False, indices=src)
                    d_roi[:, i] = dist[roi_indices]

                # symmetrize
                d_roi = np.minimum(d_roi, d_roi.T)

                if hemi == "rh":
                    roi_indices += n_lh
                # save
                np.savez_compressed(out_file, distances=d_roi, indices=roi_indices)
                print(f"   → saved {k}×{k} to {out_file}")

    print("All done.")


# def spearman_downsample_median(
#     a: np.ndarray,
#     b: np.ndarray,
#     fraction: float = 0.17,
#     trials: int = 1000,
#     seed: int | None = None,
# ) -> tuple[float, float]:
#     """Repeated subsampling for robust Spearman estimate."""
#     if seed is not None:
#         random.seed(seed)
#     M = a.size
#     n = max(1, int(round(M * fraction)))
#     rhos, pvals = [], []
#     idxs = list(range(M))
#     for _ in range(trials):
#         s = random.sample(idxs, n)
#         r = scipy.stats.spearmanr(a[s], b[s])
#         rhos.append(r.correlation if hasattr(r, "correlation") else r.statistic)
#         pvals.append(r.pvalue)
#     return float(np.median(rhos)), float(np.median(pvals))


# def compute_correlations(
#     config: Configuration,
#     subjects: list[int],
#     hemis: list[str] = ["lh", "rh"],
#     rois: list[int] = list(range(1, 10)),
#     downsample_frac: float = 1.0 / (1.8**3),
#     trials: int = 1000,
#     alpha: float = 0.05,
# ) -> None:
#     """Load MDS vs cortical .npz pairs, compute Spearman, save & plot."""
#     # derive dirs
#     mds_base = os.path.join(config.directories.distances_dir, "mds_distances")
#     cort_base = os.path.join(config.directories.distances_dir, "distances_cortical")
#     out_excel = os.path.join(
#         config.directories.distances_dir, "roi_spearman_distances.xlsx"
#     )
#     out_pkl = os.path.join(
#         config.directories.distances_dir, "roi_spearman_distances.pkl"
#     )

#     rows: list[dict] = []
#     for sub in tqdm(subjects, desc="Subjects"):
#         sub_str = f"subj_{sub:02d}"
#         for hemi in hemis:
#             for roi in tqdm(rois, desc=f"ROIS – hemi={hemi}"):
#                 mds_f = os.path.join(mds_base, sub_str, f"{hemi}.dists.roi_{roi}.npz")
#                 cort_f = os.path.join(cort_base, sub_str, f"{hemi}.dists.roi_{roi}.npz")
#                 if not os.path.isfile(mds_f) or not os.path.isfile(cort_f):
#                     continue

#                 # load
#                 md = np.load(mds_f)
#                 cd = np.load(cort_f)
#                 ids_m, ids_c = md["indices"], cd["indices"]
#                 Dm, Dc = md["distances"], cd["distances"]

#                 # restrict to ids in MDS (preserves MDS order)
#                 mask_m = np.isin(ids_m, ids_c)
#                 if mask_m.sum() < 2:
#                     continue
#                 ids_m_sub = ids_m[mask_m]

#                 # find matching cortical positions
#                 ic = {int(v): i for i, v in enumerate(ids_c)}
#                 sel_c = [ic[int(v)] for v in ids_m_sub]

#                 # sub‐set distance matrices
#                 Dm2 = Dm[np.ix_(mask_m, mask_m)]
#                 Dc2 = Dc[np.ix_(sel_c, sel_c)]

#                 # sanity check: same ordered indices
#                 ids_c_sub = ids_c[sel_c]
#                 if not np.array_equal(ids_m_sub, ids_c_sub):
#                     raise ValueError(
#                         f"Index mismatch for {sub_str}, {hemi}, ROI {roi}: "
#                         "MDS and cortical indices do not align after filtering."
#                     )

#                 # flatten upper triangle
#                 triu = np.triu_indices(Dm2.shape[0], k=1)
#                 a_flat, b_flat = Dm2[triu], Dc2[triu]

#                 # compute spearman (with down‐sampling)
#                 rho, p = spearman_downsample_median(
#                     a_flat, b_flat, fraction=downsample_frac, trials=trials, seed=0
#                 )

#                 rows.append(
#                     {
#                         "subject": sub_str,
#                         "hemisphere": hemi,
#                         "roi": roi,
#                         "n_pairs": int(a_flat.size),
#                         "spearman_rho": rho,
#                         "spearman_pval": p,
#                     }
#                 )

#                 # ————————————————
#                 # incremental save
#                 df = pd.DataFrame(rows)
#                 df.to_excel(out_excel, index=False)
#                 df.to_pickle(out_pkl)
#                 # ————————————————

#     print(f"[CORR] Saved final results → {out_excel}")

#     # OPTIONALLY: add your plotting code here (or call a separate module)
#     # but beware that in scripts it's often better to save figures than show() them.


import os
import random
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm


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
    If shuffle=True, shuffle `a` each trial to build a null distribution.
    If fraction<1, subsample that fraction each trial (bootstrap).
    Returns:
        rhos:   shape (trials,)
        pvals:  shape (trials,)
    """
    if seed is not None:
        random.seed(seed)

    M = a.size
    # number of pairs per trial (full if fraction=1)
    n = max(1, int(round(M * fraction)))
    idxs = list(range(M))

    print(f"{fraction=}\t{n}/{M}")

    rhos = np.empty(trials, dtype=float)
    pvals = np.empty(trials, dtype=float)

    for t in range(trials):
        # maybe shuffle a
        if shuffle:
            a_t = a.copy()
            random.shuffle(a_t)
        else:
            a_t = a

        # maybe subsample
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
) -> None:
    """Compute & save trial‐level Spearman for full/bootstrap/null."""
    mds_base = os.path.join(config.directories.distances_dir, "mds_distances")
    cort_base = os.path.join(config.directories.distances_dir, "distances_cortical")

    out_long_pkl = os.path.join(
        config.directories.distances_dir, "roi_spearman_distances_trials.pkl"
    )
    out_summary_xlsx = os.path.join(
        config.directories.distances_dir, "roi_spearman_distances_summary.xlsx"
    )

    long_rows = []

    for sub in tqdm(subjects, desc="Subjects"):
        sub_str = f"subj_{sub:02d}"
        for hemi in hemis:
            for roi in tqdm(rois, desc=f"ROIS – hemi={hemi}"):
                mds_f = os.path.join(mds_base, sub_str, f"{hemi}.dists.roi_{roi}.npz")
                cort_f = os.path.join(cort_base, sub_str, f"{hemi}.dists.roi_{roi}.npz")
                if not os.path.isfile(mds_f) or not os.path.isfile(cort_f):
                    continue

                # load + align indices
                md = np.load(mds_f)
                cd = np.load(cort_f)
                ids_m, ids_c = md["indices"], cd["indices"]
                Dm, Dc = md["distances"], cd["distances"]
                mask_m = np.isin(ids_m, ids_c)
                if mask_m.sum() < 2:
                    continue
                ic = {int(v): i for i, v in enumerate(ids_c)}
                sel_c = [ic[int(v)] for v in ids_m[mask_m]]
                Dm2 = Dm[np.ix_(mask_m, mask_m)]
                Dc2 = Dc[np.ix_(sel_c, sel_c)]
                triu = np.triu_indices(Dm2.shape[0], k=1)
                a_flat, b_flat = Dm2[triu], Dc2[triu]
                n_pairs = a_flat.size

                # --- full (single trial, no shuffle/subsample) ---
                rhos, pvals = spearman_trials(
                    a_flat, b_flat, fraction=1.0, trials=1, shuffle=False, seed=0
                )
                for t, (rho, p) in enumerate(zip(rhos, pvals)):
                    long_rows.append(
                        {
                            "subject": sub_str,
                            "hemisphere": hemi,
                            "roi": roi,
                            "method": "full",
                            "trial": t,
                            "n_pairs": n_pairs,
                            "rho": rho,
                            "pval": p,
                        }
                    )

                # --- bootstrap downsample ---
                rhos, pvals = spearman_trials(
                    a_flat,
                    b_flat,
                    fraction=downsample_frac,
                    trials=trials,
                    shuffle=False,
                    seed=0,
                )
                for t, (rho, p) in enumerate(zip(rhos, pvals)):
                    long_rows.append(
                        {
                            "subject": sub_str,
                            "hemisphere": hemi,
                            "roi": roi,
                            "method": "bootstrap",
                            "trial": t,
                            "n_pairs": round(n_pairs*downsample_frac),
                            "rho": rho,
                            "pval": p,
                        }
                    )

                # --- null (bootstrap + shuffle) ---
                rhos, pvals = spearman_trials(
                    a_flat,
                    b_flat,
                    fraction=downsample_frac,
                    trials=trials,
                    shuffle=True,
                    seed=0,
                )
                for t, (rho, p) in enumerate(zip(rhos, pvals)):
                    long_rows.append(
                        {
                            "subject": sub_str,
                            "hemisphere": hemi,
                            "roi": roi,
                            "method": "null",
                            "trial": t,
                            "n_pairs": round(n_pairs*downsample_frac),
                            "rho": rho,
                            "pval": p,
                        }
                    )

                # save incrementally after each ROI (pickle is fast)
                pd.DataFrame(long_rows).to_pickle(out_long_pkl)

    """# at the end, also write a tiny summary (medians per method)
    df_long = pd.DataFrame(long_rows)
    summary = (
        df_long
        .groupby(["subject","hemisphere","roi","method"])
        .agg(
            median_rho   = ("rho", "median"),
            median_pval  = ("pval","median"),
            n_pairs      = ("n_pairs","first"),
            n_trials     = ("trial","count")
        )
        .reset_index()
    )
    summary.to_excel(out_summary_xlsx, index=False)"""

    print(f"[CORR] trial‐level data → {out_long_pkl}")
    # print(f"[CORR] summary metrics   → {out_summary_xlsx}")


def main():
    subjects = [1, 2, 3, 4, 5, 7]
    cfg = load_config("config.yaml")

    # if run_mds:
    compute_mds_distances(cfg, subjects)

    # if run_cortical:
    compute_cortical_distances(cfg, subjects)

    # if run_corr:
    compute_correlations(cfg, subjects, trials=100)


def run_over_night():
    config = load_config("config.yaml")

    from rsa.create_rdm import create_rdm
    from t_testing.clean_roi_mask import modify_mask_with_ttest

    subjects = [1, 2, 3, 4]

    mask_values = list(config.analysis.rois_to_analyze.values())
    for subject in subjects:
        if subject == "shared":
            set_to_take = "shared"
            subject_list = range(1, 8 + 1)

        else:
            set_to_take = f"subj_{subject:02d}"
            subject_list = [subject]

        for sub in subject_list:
            if set_to_take == "shared":
                shared_bool = True
            else:
                shared_bool = False

            modify_mask_with_ttest(config, 3.0, shared_bool, sub)

        for mask_value in mask_values:
            create_rdm(
                config,
                subject_list,
                mask_value,
                set_to_take,
                3.0,
                mode="averaged",
                randomization=True,
            )

    for subject in subjects:
        set_to_take = f"subj_{subject:02d}"
        subject_list = [subject]

        fit_gaussian_params(config, subject_list, set_to_take)

    compute_mds_distances(config, subjects)
    compute_correlations(config, subjects, trials=100)


if __name__ == "__main__":
    main()

    # run_over_night()
