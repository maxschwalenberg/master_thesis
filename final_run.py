import os
from utils.config import Configuration, load_config
from rsa.sample_repeatability import (
    sample_repeatability_distances_data,
    create_all_rdms,
    analyze_within_subjects,
)
from rsa.permutation_analysis import run_mantel_test_for_subject_mask
from gaussian.fit_gaussian import fit_gaussian_params
from utils.utils import retrieve_roi_mask
from rsa.cortical_correlation import (
    compute_cortical_distances,
    compute_mds_distances,
    compute_correlations,
)
from tqdm import tqdm
import numpy as np
import pandas as pd


# TODO


def run():
    config = load_config("config.yaml")
    subjects = list(range(1, 9))
    n_randomizations = 4
    n_masks = 9
    mask_mode = "single"

    t_threshhold = 3
    sigma_threshold = 5

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

    mask_values = [1, 2, 3, 4, 5, 6, 7, 9]

    # 1. sample repeatability
    # approach (2)

    # sample_repeatability_distances_data(config, subjects, t_thresh=t_threshhold, mask_values=mask_values)

    # approach (1)

    config.directories.mds_dir = "data/mds/final_run_rsa"
    config.directories.rdm_dir = "data/rdm/final_run_rsa"
    sets = ["final_run_rsa"]

    # e.g. subjects 1–8
    # for subj_id in subjects:
    #     print(f"\n=== Generating RDMs for subject {subj_id:02d} ===")
    #     create_all_rdms(
    #         config,
    #         subj_id,
    #         mask_values=mask_values,
    #         n_samples=3,
    #         n_randomizations=n_randomizations,
    #         t_thresh=t_threshhold,
    #     )

    # print("\n\n=== Correlating RDMs and MDSs ===")

    # df_obs, df_null = analyze_within_subjects(
    #     config,
    #     sets=sets,
    #     subjects=subjects,
    #     n_masks=n_masks,
    #     n_randomizations=n_randomizations,
    #     mask_mode=mask_mode,
    # )
    # df_obs.to_excel(sample_repeatability_correlations_true_excel)
    # df_null.to_excel(sample_repeatability_correlations_null_excel)

    # 2. permutation encoding analysis
    # results_path = os.path.join(config.directories.output_dir, config.saved_results_paths.permutation_analysis_excel)

    # results = []
    # for subj in range(1, 9):
    #     subj_mask = retrieve_roi_mask(config, subj, f"subj_{subj:02d}", True, t_threshhold)

    #     mask_values = np.unique(subj_mask).tolist()
    #     mask_values.remove(0)

    #     try:
    #         mask_values.remove(8)
    #     except:
    #         print(f"Mask 8 not existent")

    #     for mask in tqdm(mask_values):
    #         for feature_selection in ["centers", "sizes", "ages", "genders"]:
    #             r = run_mantel_test_for_subject_mask(
    #                 subj, mask,"final_run_rsa", t_threshhold, B=2000, feature_selection=feature_selection, config=config
    #             )
    #             results += r

    #     df_results = pd.DataFrame(results)
    #     df_results.to_excel(results_path)

    # df_results = pd.DataFrame(results)
    # df_results.to_excel(results_path)

    # # 3. voxel_fits
    # # positively
    # config.directories.gaussian_fit_results_dir = (
    #     "data/gaussian_results/final_run"
    # )

    # for subj in subjects:
    #     fit_gaussian_params(
    #         config,
    #         [subj],
    #         f"subj_{subj:02d}",
    #         use_negative_slope=False,
    #         use_mixed_slope=False,
    #         t_test_threshold=t_threshhold
    #     )

    subjects = [1, 2, 3, 4, 5, 6, 7, 8]
    # mask_values = [2,3,4,5,6,7,9]
    # subjects = [6, 8]

    compute_cortical_distances(config, subjects, rois=mask_values)

    # if run_mds:
    config.directories.gaussian_fit_results_dir = "data/gaussian_results/final_run"
    mask_values = [1, 2, 3, 4, 5, 6, 7, 9]

    compute_mds_distances(
        config,
        subjects,
        t_threshhold,
        rois=mask_values,
        sigma_threshold=sigma_threshold,
    )

    # if run_corr:
    compute_correlations(config, subjects, trials=100, rois=mask_values)


if __name__ == "__main__":
    run()
