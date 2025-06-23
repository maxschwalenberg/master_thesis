import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from rsa.permutation_analysis import run_mantel_test
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

def run_mantel_test_for_subject_mask(
    subject_i: int,
    mask: int,
    B: int = 2000,
    feature_selection: str = "centers"
) -> dict:
    """
    Loads X and metadata for (subject_i, mask), then delegates to run_mantel_test.
    """
    rand = 0

    # 1) load your precomputed distance matrix & metadata
    X = np.load(
        f"data/mds/final/"
        f"subj_{subject_i:02d}/subj_{subject_i:02d}/"
        f"mask_{mask}_averaged_both_rand_{rand}_mds.npy"
    )

    D_X = np.load(
        f"data/rdm/final/"
        f"subj_{subject_i:02d}/subj_{subject_i:02d}/"
        f"mask_{mask}_averaged_both_rand_{rand}_rdm.npy"
    )

    metadata = np.load(
        f"data/rdm/final/"
        f"subj_{subject_i:02d}/subj_{subject_i:02d}/"
        "metadata.npy"
    )
    # 2) call the new, more flexible function
    res = run_mantel_test(
        D_X,
        X,
        subject_i,
        metadata,
        B=B,
        feature_selection=feature_selection
    )

    # 3) re-attach mask & effect_size
    return [{
        "type" : "MDS",
        "subject":     res["MDS"]["subject"],
        "mask":        mask,
        "r_obs":       res["MDS"]["r_obs"],
        "p_value":     res["MDS"]["p_value"],
        "feature":     res["MDS"]["feature"]
    },{
        "type" : "RDM",
        "subject":     res["RDM"]["subject"],
        "mask":        mask,
        "r_obs":       res["RDM"]["r_obs"],
        "p_value":     res["RDM"]["p_value"],
        "feature":     res["RDM"]["feature"]
    }]

    # except Exception as e:
    #     print(f"[ERROR] subj={subject_i:02d}, mask={mask} → {e}")
    #     return {
    #         "subject":     subject_i,
    #         "mask":        mask,
    #         "r_obs":       np.nan,
    #         "p_value":     np.nan,
    #         "feature":     feature_selection
    #     }


# -------------------------------------------------------------------
# Run it for all subjects & masks, collect into a DataFrame:

from utils.utils import retrieve_roi_mask
from utils.config import load_config


config = load_config("config.yaml")
results = []
for subj in range(1, 9):
    subj_mask = retrieve_roi_mask(config, subj, f"subj_{subj:02d}", False)

    mask_values = np.unique(subj_mask).tolist()
    mask_values.remove(0)

    for mask in tqdm(mask_values):
        for feature_selection in ["centers", "sizes", "ages", "genders"]:
            r = run_mantel_test_for_subject_mask(subj, mask, B=2000, feature_selection=feature_selection)
            print(r)
            results += r
            # print(f"Subject {r['subject']:02d}, Mask {r['mask']} → "
            #     f"r = {r['r_obs']:.3f}, p = {r['p_value']:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_excel(f"correlation_res.xlsx")

df_results = pd.DataFrame(results)
df_results.to_excel(f"correlation_res.xlsx")
