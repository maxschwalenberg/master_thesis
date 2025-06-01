import os
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

def mantel_test_feature(X: np.ndarray, feature: np.ndarray, B: int = 2000, metric: str = 'euclidean') -> dict:
    X = np.asarray(X)
    feature = np.asarray(feature)

    assert X.shape[0] == feature.shape[0], "X and feature must have same length"

    if feature.ndim == 1:
        feature_vals = feature.reshape(-1, 1)
    else:
        feature_vals = feature

    n = X.shape[0]

    D_X = squareform(pdist(X, metric=metric))
    D_F = squareform(pdist(feature_vals, metric=metric))

    iu = np.triu_indices(n, k=1)
    d_vec = D_X[iu]
    f_vec = D_F[iu]

    r_obs, _ = spearmanr(d_vec, f_vec)

    r_perm = np.empty(B)
    for b in range(B):
        perm = np.random.permutation(n)
        D_F_perm = D_F[perm][:, perm]
        f_p = D_F_perm[iu]
        r_perm[b], _ = spearmanr(d_vec, f_p)

    p_value = (np.sum(np.abs(r_perm) >= abs(r_obs)) + 1) / (B + 1)

    return {
        'r_obs': r_obs,
        'p_value': p_value,
        'r_perm': r_perm
    }

def run_mantel_test(X, subject_i, metadata, B: int = 2000, feature_selection: str = "centers") -> dict:
    # print("Running mantel test with feature selection:", feature_selection)
    try:
        # metadata = np.load(f"data/rdm/rdm_stan_data_face_only/subj_{subject_i:02d}/subj_{subject_i:02d}/metadata.npy")
        animate_labels = pd.read_excel(f"data/labels/subj_{subject_i:02d}/animate_nonface/animate_non_face_final.xlsx")
        face_labels = pd.read_excel(f"data/labels/subj_{subject_i:02d}/faces/faces_final.xlsx")


        face_det_path = f"data/labels/subj_{subject_i:02d}/face_detection_result.json"
        with open(face_det_path, "r") as f:
            face_detections = json.load(f)

        filtered = [x for x in face_detections if x["file_name"][:-4] in metadata]
        for x in filtered:
            x["cocoId"] = x["file_name"][:-4]

        centers = []
        eye_coordinates = []
        feature_array_age = []
        sizes = []
        genders = []

        for entry in metadata:
            # find the matching detection+metadata record
            match = next(x for x in filtered if x["cocoId"] == entry)
            assert len(match["detection"]) == 1, "expected exactly one bbox"
            
            # --- bbox center & size ---
            x1, y1, x2, y2 = match["detection"][0]
            # center point
            center = np.array([ (x1 + x2) / 2, (y1 + y2) / 2 ])
            centers.append(center)
            # width & height
            width  = x2 - x1
            height = y2 - y1
            sizes.append(np.array([width, height]))
            
            # --- demographic feature ---
            feature_array_age.append(match["age"])
            genders.append(match["gender"])
            
            # --- eye landmarks ---
            # assuming match["landmarks"] is a list of (x,y) tuples or lists
            lm = match["landmarks"][0]
            left_eye  = np.array(lm[38])
            right_eye = np.array(lm[88])
            eye_coordinates.append(
                [left_eye, right_eye]
            )

        if feature_selection == "centers":
            features = np.array(centers)
        elif feature_selection == "sizes":
            features = np.array(sizes)
        elif feature_selection == "eye":
            features = np.array(eye_coordinates)
        elif feature_selection == "ages":
            features = np.array(feature_array_age)
        elif feature_selection == "genders":
            features = np.array(genders)
        

        # # features = centers_np
        # features = centers_np




        result = mantel_test_feature(X, features, B=B)

        return {
            "subject": subject_i,
            "r_obs": result["r_obs"],
            "p_value": result["p_value"],
            "feature": feature_selection
        }

    except Exception as e:
        print(e)
        # print(f"Subject {subject_i:02d}, Mask {mask} failed: {e}")
        return {
            "subject": subject_i,
            # "mask": mask,
            "r_obs": np.nan,
            "p_value": np.nan,
            "feature": feature_selection
        }