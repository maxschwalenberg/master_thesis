import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, kendalltau
from utils.config import load_config, Configuration
from rsa.create_rdm import create_rdm
from matplotlib.lines import Line2D


# MDS distances
# ------------------------
import os
import numpy as np
import nibabel as nib
from scipy.spatial.distance import pdist
from tqdm import tqdm

from t_testing.clean_roi_mask import modify_mask_with_ttest
import nibabel as nib
import pandas as pd
import logging

from utils.config import Configuration, load_config
from utils.utils import retrieve_stacked_betas
from previousCode.nsddatapaper_rsa.utils.utils import mds

from scipy.stats import wasserstein_distance


# Define custom metric wrapper
def emd(u, v):
    return wasserstein_distance(u, v)


def create_rdm_with_all_trials(
    config: Configuration,
    subj_list: list[int],
    mask_values: list[int],
    set_to_take,
    t_test_threshold,
    randomization=False,
):
    """
    For each subject, load all 3 trials per image (so 150 images ⇒ 450 patterns),
    apply your ROI mask, compute the pairwise RDM (450×450) and MDS coords (450×2).

    Returns
    -------
    results : dict
        { sub : {
            'rdm'      : 1D array of length 450*449/2,
            'mds'      : array (450,2),
            'meta'     : list of (image_id, trial_idx) for each of the 450 rows
          }
        }
    """
    results = {}

    for sub in tqdm(subj_list, desc="Subjects"):
        results[sub] = {}

        # 1) Make t-test mask
        pick_shared = set_to_take == "shared"
        modify_mask_with_ttest(
            config, t_test_threshold, pick_shared, sub, sub_filename="temporary_mask"
        )

        # 2) Load and concatenate LH & RH mask arrays
        roi_dir = config.directories.t_test_roi_dir
        # data/t_test_roi/subj_02/lh.subj02.temporary_mask.mgz
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

        all_patterns = []
        meta = []  # to store (image_id, trial_idx)
        for trial_idx in range(3):
            try:
                betas, image_ids, _ = retrieve_stacked_betas(
                    config,
                    sub,
                    "single",  # single ⇒ one trial at a time
                    trial_idx,
                    subj_to_check=set_to_take,
                    only_face_set=config.pipeline.step_4_rsa_analysis.only_face_set,
                    randomization=randomization,
                )
            except Exception as e:
                print(f"[Warning] error retrieving trial {trial_idx}: {e}")
                continue

            # betas: (n_images, V) → we want (n_images, V) (↔ no change, but sanity‐check)
            # if it were (V, n_images) you'd do betas = betas.T
            all_patterns.append(betas)
            meta += [(img, trial_idx) for img in image_ids]

        # Final stack: (n_trials_total, V)
        X = np.vstack(all_patterns)
        assert len(meta) == X.shape[0]

        # ── 2) Loop ROIs and compute RDM + MDS ─────────────────────────────────────────
        for mv in mask_values:
            # build mask for this ROI
            if isinstance(mv, int):
                vox_selector = combined_mask == mv
            else:
                vox_selector = np.isin(combined_mask, mv)
            vox_selector = vox_selector.flatten()

            n_vox = vox_selector.sum()
            if n_vox == 0:
                print(f"[Info] Mask {mv}: no voxels → skipping")
                continue

            print(f"[Info] Mask {mv}: {n_vox} voxels")

            # apply mask
            X_masked = X[:, vox_selector]
            assert not np.isnan(X_masked).any()

            # compute RDM + MDS
            dm = config.pipeline.step_4_rsa_analysis.distance_metric
            metric_fn = emd if dm == "wasserstein" else dm
            rdm_vec = pdist(X_masked, metric=metric_fn)
            mds_coords = mds(rdm_vec).astype(np.float32)

            # store
            results[sub][mv] = {"rdm": rdm_vec, "mds": mds_coords, "meta": meta}

    return results


import numpy as np
import matplotlib.pyplot as plt


def plot_triangle_difference():
    # ─── 1) PARAMETERS ─────────────────────────────────────────────────────────────
    subjects = list(range(1, 9))  # subjects 1–8
    subjects = [1, 2]
    mask_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # ROIs
    t_thresh = 2.5

    # ─── 2) LOAD + COMPUTE ALL RDM/MDS ────────────────────────────────────────────
    config = load_config("config.yaml")
    # results_all will be { subj: {mv: { 'mds':…, 'meta':… }, … }, … }
    results_all = {}

    

    rdm_results_path = os.path.join("data/sample_repeatability_dict.pkl")

    import pickle

    if os.path.exists(rdm_results_path):
        with open(rdm_results_path, "rb") as f:
            results_all = pickle.load(f)
        
    else:
        for subj in subjects:
            res = create_rdm_with_all_trials(
                config,
                subj_list=[subj],
                mask_values=mask_values,
                set_to_take=f"subj_{subj:02d}",
                t_test_threshold=t_thresh,
            )
            results_all.update(res)
        with open(rdm_results_path, "wb") as f:
        # protocol=pickle.HIGHEST_PROTOCOL picks the fastest/best format
            pickle.dump(results_all, f, protocol=pickle.HIGHEST_PROTOCOL)

    quit()
    # ─── 3) GATHER ALL DISTANCES ──────────────────────────────────────────────────
    all_dists = []
    for subj in subjects:
        subj_res = results_all.get(subj, {})
        for mv in mask_values:
            if mv not in subj_res:
                # force an empty slot
                all_dists.append({"mask": mv, "distance": np.nan})
                continue

            mds_coords = subj_res[mv]["mds"]
            meta = subj_res[mv]["meta"]

            # for each unique image in this subj/ROI
            for img in sorted({img_id for img_id, _ in meta}):
                idxs = [i for i, (img_id, _) in enumerate(meta) if img_id == img]
                pts = mds_coords[idxs]
                if len(pts) < 2:
                    continue

                # record every pairwise distance
                for i, j in combinations(range(len(pts)), 2):
                    d = np.linalg.norm(pts[i] - pts[j])
                    all_dists.append({"mask": mv, "distance": d})

    # make a DataFrame
    df = pd.DataFrame(all_dists)

    # ─── 4) PLOT ONE BIG VIOLIN PER ROI ────────────────────────────────────────────
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.violinplot(
        x="mask",
        y="distance",
        data=df,
        order=mask_values,
        inner=None,  # no quartiles inside the violin
        cut=0,  # no tails beyond the data
        palette="viridis",
        bw=0.2,
    )

    # overlay boxplots for IQR + median
    sns.boxplot(
        x="mask",
        y="distance",
        data=df,
        order=mask_values,
        width=0.2,
        boxprops={"facecolor": "none", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        medianprops={"color": "red", "linewidth": 2},
        showfliers=False,
    )

    # overlay mean as red 'X'
    sns.pointplot(
        x="mask",
        y="distance",
        data=df,
        order=mask_values,
        estimator=np.mean,
        errorbar=None,
        join=False,
        color="red",
        marker="X",
        markersize=8,
    )

    # zoom into central IQR region
    q1, q3 = df["distance"].quantile([0.25, 0.75])
    iqr = q3 - q1
    plt.ylim(max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)

    plt.xlabel("Mask value (ROI)")
    plt.ylabel("Pairwise distance")
    plt.title("Within‐Image Pairwise Distances (all subjects) by ROI")
    plt.legend(
        [Line2D([0], [0], marker="X", color="red", linestyle="none")],
        ["Mean"],
        loc="upper right",
    )
    plt.savefig(f"data/output/sample_repeat_mds.png")
    # plt.show()


def create_all_rdms(config: Configuration, subj_id, n_samples=3, n_randomizations=1):
    """Generate RDMs for one subject across all configured ROIs."""
    set_to_take = f"subj_{subj_id:02d}"
    mask_values = list(config.analysis.rois_to_analyze.values())

    while True:
        try:
            for sample_v in range(n_samples):
                for n_rand in range(n_randomizations):
                    create_rdm(
                        config,
                        [subj_id],
                        mask_values,
                        set_to_take,
                        3.0,
                        mode="single",
                        sample_to_pick=sample_v,
                        randomization=True,
                        augment_shared_set=True,
                        randomize_offset=n_rand,
                    )
            break
        except Exception as e:
            # if first run fails, cut back to n_samples-1 and retry once
            print(e)
            print(
                f"  → failed generating RDMs for subj={subj_id}, mask={mask_values} sample={sample_v}"
            )
            break


from itertools import combinations
from scipy.spatial.distance import pdist


import sys
import os
import logging
from itertools import combinations
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr


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
    For each set / subject / mask / hemi:
      - load up to 3 RDM samples, compute all pairwise Pearson(RDM_i, RDM_j)
      - load up to 3 MDS samples, align via Procrustes, then Pearson(dist_i, dist_j)

    Returns a DataFrame with columns:
      ['set','subject','mask','hemi','type','sample_i','sample_j','r','p','disparity']
    """
    records = []
    null_rec = []

    for set_path in sets:
        # re‐point your config each loop
        config.directories.rdm_dir = f"data/rdm/{set_path}"
        config.directories.mds_dir = f"data/mds/{set_path}"

        for subj in subjects:
            subj_str = f"subj_{subj:02d}"

            for mask_value in range(1, n_masks + 1):
                for hemi in hemispheres:
                    # —— RDM samples ——
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
                                    logging.info(f"Couldn’t load {p}: {e}")

                        # pairwise RDM–RDM
                        if len(rdm_samples) >= 2:
                            for i, j in combinations(sorted(rdm_samples), 2):
                                x, y = rdm_samples[i], rdm_samples[j]
                                # — observed —
                                r_obs, p_obs = pearsonr(x, y)
                                base = {
                                    "set": set_path,
                                    "subject": subj,
                                    "mask": mask_value,
                                    "hemi": hemi,
                                    "sample_i": i,
                                    "sample_j": j,
                                    "randomization": randomize_offset,
                                }
                                records.append(
                                    {
                                        **base,
                                        "type": "RDM",
                                        "r": r_obs,
                                        "p": p_obs,
                                        "disparity": np.nan,
                                    }
                                )

                                # — nulls —
                                for perm in range(n_permutations):
                                    y_shuf = np.random.permutation(y)
                                    r_null, p_null = pearsonr(x, y_shuf)
                                    null_rec.append(
                                        {
                                            **base,
                                            "type": "RDM",
                                            "perm": perm,
                                            "r": r_null,
                                            "p": p_null,
                                            "disparity": np.nan,
                                        }
                                    )

                    # —— MDS samples ——
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
                            # data/mds/sample_repeatability/subj_08/subj_08/mask_6_single_sample_2_rh_mds.npy
                            # data/mds/sample_repeatability/subj_01/subj_01/mask_1_single_sample_0_both_mds.npy
                            if os.path.exists(p):
                                try:
                                    mds_samples[idx] = np.load(p)
                                except Exception as e:
                                    logging.warning(f"Couldn’t load {p}: {e}")
                            else:
                                print(f"Not found {p=}")

                        # pairwise MDS–MDS via Procrustes + Pearson(distances)
                        if len(mds_samples) >= 2:
                            for i, j in combinations(sorted(mds_samples), 2):
                                m1, m2, disp = procrustes(
                                    mds_samples[i], mds_samples[j]
                                )
                                d1 = squareform(pdist(m1))
                                d2 = squareform(pdist(m2))
                                r, pval = pearsonr(d1.ravel(), d2.ravel())

                                records.append(
                                    {
                                        "set": set_path,
                                        "subject": subj,
                                        "mask": mask_value,
                                        "hemi": hemi,
                                        "type": "MDS",
                                        "sample_i": i,
                                        "sample_j": j,
                                        "r": r,
                                        "p": pval,
                                        "randomization": randomize_offset,
                                        "disparity": disp,
                                    }
                                )

    df_obs = pd.DataFrame.from_records(records)
    df_null = pd.DataFrame.from_records(null_rec)
    return df_obs, df_null
    return pd.DataFrame.from_records(records)


def plot_rdm_correlations_from_excel(
    path: str,
    plot_title: str,
    sheet_name: str = None,
    excel_kwargs: dict = None,
    figsize: tuple = (14, 6),
    violin_kwargs: dict = None,
    rdm_bool: bool = True,
    y_lim: list[float] = None,
) -> plt.Figure:
    """
    Load an Excel (or CSV) of RDM correlations and plot
    a grouped violin of Pearson r by mask_value & hemisphere.

    Parameters
    ----------
    path : str
        Path to the Excel or CSV file.
    sheet_name : str, optional
        If reading Excel, the sheet name. Passed to pd.read_excel.
        If None and path ends in .csv, will call pd.read_csv.
    excel_kwargs : dict, optional
        Additional kwargs for pd.read_excel or pd.read_csv.
    figsize : tuple, optional
        Figure size in inches.
    violin_kwargs : dict, optional
        Additional kwargs for sns.violinplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    excel_kwargs = excel_kwargs or {}
    violin_kwargs = violin_kwargs or {}

    # 1) Load the data
    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, sheet_name=sheet_name, **excel_kwargs)
    else:
        raise FileNotFoundError(f"Please use excel file.")

    if rdm_bool:
        print(f"Using RDM entries!")
        df = df[df["type"] == "RDM"]
    else:
        print(f"Using MDS entries!")
        df = df[df["type"] == "MDS"]

    # 2) Quick sanity check for required columns
    required = {"mask", "r", "hemi"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    # 3) Create the violin plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=df,
        x="mask",
        y="r",
        hue="hemi",
        inner="box",
        scale="width",
        ax=ax,
        **violin_kwargs,
    )
    ax.set_title(plot_title)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel("Mask Value")
    ax.set_ylabel("Pearson Correlation (r)")
    ax.legend(title="Hemisphere")
    ax.grid(True)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # plot_triangle_difference()

    df_results_path = f"rdm_mds_correlations.xlsx"
    df_results_path_null = f"rdm_mds_correlations_null.xlsx"

    # 1) generate all your RDMs
    config = load_config("config.yaml")

    plot_triangle_difference()


    config.directories.mds_dir = "data/mds/sample_repeatability"
    config.directories.rdm_dir = "data/rdm/sample_repeatability"

    create_rdms = False

    subjects = list(range(1, 9))
    # subjects = [1,2,3,4,5,6,7,8]

    n_randomizations = 4

    # e.g. subjects 1–8
    for subj_id in subjects:
        if create_rdms:
            print(f"\n=== Generating RDMs for subject {subj_id:02d} ===")
            create_all_rdms(
                config, subj_id, n_samples=3, n_randomizations=n_randomizations
            )

    # quit()
    # 2) analyze correlations RDM ↔ MDS
    # after you’ve generated all RDMs…
    # 2) parameters
    sets = ["sample_repeatability"]
    subjects = range(1, 9)
    n_masks = 9
    mask_mode = "single"

    # 3) call the function
    if not os.path.exists(df_results_path) or not os.path.exists(df_results_path_null):
        print("\n\n=== Correlating RDMs with MDS embeddings ===")

        df_obs, df_null = analyze_within_subjects(
            config,
            sets=sets,
            subjects=subjects,
            n_masks=n_masks,
            n_randomizations=n_randomizations,
            mask_mode=mask_mode,
        )
        df_obs.to_excel(df_results_path)
        df_null.to_excel(df_results_path_null)
    else:
        print("\n\n=== Skipping: Correlating RDMs with MDS embeddings ===")

        df_obs = pd.read_excel(df_results_path)
        df_null = pd.read_excel(df_results_path_null)

    print("\nSaved all correlations to rdm_mds_correlations.xlsx")

    fig_mds = plot_rdm_correlations_from_excel(
        df_results_path,
        "MDS Correlation Distribution by Mask and Hemisphere",
        sheet_name="Sheet1",
        rdm_bool=False,
    )
    fig_mds.savefig(f"data/output/mds_correlation.png")
    # plt.show()
    fig_rdm = plot_rdm_correlations_from_excel(
        df_results_path,
        "RDM Correlation Distribution by Mask and Hemisphere",
        sheet_name="Sheet1",
        rdm_bool=True,
    )
    fig_rdm.savefig(f"data/output/rdm_correlation.png")

    fig_rdm_null = plot_rdm_correlations_from_excel(
        df_results_path_null,
        "RDM Correlation Null-Distribution by Mask and Hemisphere",
        sheet_name="Sheet1",
        rdm_bool=True,
    )
    fig_rdm_null.savefig(f"data/output/rdm_null_correlation.png")
    # plt.show()

    # to save:
    # fig.savefig("rdm_violin_plot.png", dpi=300)

