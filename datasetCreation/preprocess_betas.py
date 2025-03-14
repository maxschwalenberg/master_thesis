import os
import numpy as np
import pandas as pd
import nibabel as nib
import h5py as h5
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.config import load_config, Configuration

config = load_config("config.yaml")

# Konfiguration: Anpassbare Parameter
SUBJECTS = range(1, 9)  # Subjects 1 bis 8
SESSIONS = range(1, 41)  # Sessions 1 bis 40
PERCENTILE = 60  # Perzentil für die globale Normalisierung

# Verzeichnisse (Anpassen nach Bedarf)
freesurfer_dir = config.freesurfer_dir  # Pfad zur FreeSurfer-Datenbank
tsv_dir = config.nsddata_responses_tsv_dir  # Pfad zu den TSV-Dateien mit Behavior-Daten
betas_dir_base = config.nsddata_betas_dir  # Basisverzeichnis für fMRI-Daten


def load_v1_rois(subject):
    """Lädt die V1-ROI-Daten für einen bestimmten Subject."""
    lh_path = os.path.join(
        freesurfer_dir,
        f"subj{subject:02d}",
        "label",
        "customrois",
        f"lh.subj{subject:02d}.testrois.mgz",
    )
    rh_path = os.path.join(
        freesurfer_dir,
        f"subj{subject:02d}",
        "label",
        "customrois",
        f"rh.subj{subject:02d}.testrois.mgz",
    )

    lh_img = nib.load(lh_path).get_fdata()
    rh_img = nib.load(rh_path).get_fdata()

    v1_rois = np.concatenate((np.squeeze(lh_img), np.squeeze(rh_img)))
    v1_indices = np.where((v1_rois == 1) | (v1_rois == 2))[0]

    return v1_indices


def load_betas(subject, session):
    """Lädt die Beta-Werte für einen bestimmten Subject und eine bestimmte Session."""
    betas_dir = os.path.join(
        betas_dir_base,
        "ppdata",
        f"subj{subject:02d}",
        "nativesurface",
        "betas_fithrf_GLMdenoise_RR",
    )
    beta_lh = os.path.join(betas_dir, f"lh.betas_session{session:02d}.hdf5")
    beta_rh = os.path.join(betas_dir, f"rh.betas_session{session:02d}.hdf5")

    with h5.File(beta_lh, "r") as f:
        betas_lh = f["betas"][:]

    with h5.File(beta_rh, "r") as f:
        betas_rh = f["betas"][:]

    return np.concatenate((betas_lh, betas_rh), axis=1)


def normalize_betas(betas, betas_v1, percentile):
    """Führt eine lokale und globale Normalisierung der Beta-Werte durch."""
    # Lokale Z-Normalisierung pro Sample (vektorisiert)
    means = np.mean(betas, axis=1, keepdims=True)
    stds = np.std(betas, axis=1, keepdims=True)
    normalized_samples = (betas - means) / stds

    # Globale Skalierung mit Perzentil
    global_amplitudes = np.percentile(np.abs(betas_v1), percentile, axis=1)
    global_amplitude_mean = np.mean(global_amplitudes)

    # Skalierung anwenden (vektorisierte Berechnung)
    scaling_factors = global_amplitudes / global_amplitude_mean
    scaled_samples = normalized_samples * scaling_factors[:, np.newaxis]

    return scaled_samples


def process_subject(subject):
    """Hauptfunktion zur Verarbeitung eines Subjects über alle Sessions."""
    print(f"Processing Subject {subject}...")

    # V1 ROIs laden
    v1_indices = load_v1_rois(subject)

    target_dir = os.path.join(
        "/media/harveylab/STORAGE1_NA/NSD/full_brain_max", f"subj{subject:02d}"
    )
    os.makedirs(target_dir, exist_ok=True)

    for session in tqdm(SESSIONS):
        target_file_path = os.path.join(
            target_dir, f"normalized_betas_session_{session}.npy"
        )
        if os.path.exists(target_file_path):
            continue

        # Betas laden
        betas = load_betas(subject, session)
        betas_v1 = betas[:, v1_indices]

        # Normalisierung berechnen
        normalized_betas = normalize_betas(betas, betas_v1, PERCENTILE)

        np.save(target_file_path, normalized_betas)

        # Hier könnte eine Speicherung oder weitere Analyse erfolgen
        # np.save(f"normalized_betas_subj{subject}_sess{session}.npy", normalized_betas)


if __name__ == "__main__":
    for subject in SUBJECTS:
        process_subject(subject)
