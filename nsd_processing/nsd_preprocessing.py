import os
import numpy as np
import nibabel as nib
import h5py as h5
import psutil
from tqdm import tqdm
from utils.config import load_config, Configuration
from utils.utils import subjects_list_unifier, logging_message

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Laden der Konfiguration
config = load_config("config.yaml")

# Globale Parameter
SUBJECTS = range(1, 2)  # Subjects 1 bis 8
SESSIONS = range(1, 41)  # Sessions 1 bis 40
PERCENTILE = 60  # Perzentil für globale Normalisierung

# Verzeichnisse
data_dirs = {
    "freesurfer": config.nsd_data.freesurfer_dir,
    "tsv": config.nsd_data.nsddata_responses_tsv_dir,
    "betas": config.nsd_data.nsddata_betas_dir,
    "target": config.nsd_data.full_brain_data_dir,
}


def get_memory_usage():
    """Gibt die aktuelle RAM-Auslastung zurück."""
    process = psutil.Process(os.getpid())
    mem_info = psutil.virtual_memory()
    return {
        "RAM genutzt (MB)": process.memory_info().rss / 1e6,
        "RAM verfügbar (MB)": mem_info.available / 1e6,
        "RAM gesamt (MB)": mem_info.total / 1e6,
        "RAM genutzt (%)": mem_info.percent,
    }


def load_v1_rois(subject):
    """Lädt die V1-ROI-Daten für einen bestimmten Subject."""
    lh_path = os.path.join(
        data_dirs["freesurfer"],
        f"subj{subject:02d}",
        "label",
        "customrois",
        f"lh.subj{subject:02d}.testrois.mgz",
    )
    rh_path = os.path.join(
        data_dirs["freesurfer"],
        f"subj{subject:02d}",
        "label",
        "customrois",
        f"rh.subj{subject:02d}.testrois.mgz",
    )
    lh_img = nib.load(lh_path).get_fdata()
    rh_img = nib.load(rh_path).get_fdata()
    v1_rois = np.concatenate((np.squeeze(lh_img), np.squeeze(rh_img)))
    return np.where((v1_rois == 1) | (v1_rois == 2))[0]


def load_betas(subject, session):
    """Lädt die Beta-Werte für einen bestimmten Subject und eine bestimmte Session."""
    betas_dir = os.path.join(
        data_dirs["betas"],
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
    """Führt eine lokale und globale Normalisierung durch."""
    means, stds = np.mean(betas, axis=1, keepdims=True), np.std(
        betas, axis=1, keepdims=True
    )
    normalized_samples = (betas - means) / stds
    global_amplitudes = np.percentile(np.abs(betas_v1), percentile, axis=1)
    scaling_factors = global_amplitudes / np.mean(global_amplitudes)
    return normalized_samples * scaling_factors[:, np.newaxis]


def process_subject(subject):
    """Verarbeitet einen Subject über alle Sessions."""
    print(f"Processing Subject {subject}...")
    v1_indices = load_v1_rois(subject)
    target_dir = os.path.join(data_dirs["target"], f"subj{subject:02d}")
    os.makedirs(target_dir, exist_ok=True)
    for session in tqdm(SESSIONS):
        target_file_path = os.path.join(
            target_dir, f"normalized_betas_session_{session}.npy"
        )
        if os.path.exists(target_file_path):
            continue
        betas = load_betas(subject, session)
        betas_v1 = betas[:, v1_indices]
        normalized_betas = normalize_betas(betas, betas_v1, PERCENTILE)
        np.save(target_file_path, normalized_betas)


def concatenate_subject_sessions(subject):
    """Fasst alle normalisierten Beta-Werte für einen Subject zusammen und speichert sie kompatibel mit np.load()."""
    print(f"Kombiniere Sessions für Subject {subject}...")

    target_dir = os.path.join(data_dirs["target"], f"subj{subject:02d}")
    concatenated_file_path = os.path.join(
        target_dir, "normalized_betas_all_sessions.npy"
    )

    first_session_file = os.path.join(target_dir, "normalized_betas_session_1.npy")
    if not os.path.exists(first_session_file):
        raise FileNotFoundError(f"Fehlende Datei: {first_session_file}")

    # Erste Datei laden, um die Dimensionen zu bestimmen
    first_betas = np.load(
        first_session_file, mmap_mode="r"
    )  # Vermeidung von RAM-Überlastung
    n_samples, n_voxels = first_betas.shape

    # `np.memmap` als temporäre Datei nutzen
    temp_memmap_file = concatenated_file_path + ".tmp"
    concatenated_betas = np.memmap(
        temp_memmap_file,
        dtype=np.float32,
        mode="w+",
        shape=(n_samples * len(SESSIONS), n_voxels),
    )

    start_idx = 0
    for session in tqdm(SESSIONS, desc=f"Subject {subject}: Processing Sessions"):
        session_file_path = os.path.join(
            target_dir, f"normalized_betas_session_{session}.npy"
        )

        if os.path.exists(session_file_path):
            betas = np.load(
                session_file_path, mmap_mode="r"
            )  # Vermeidet hohe RAM-Nutzung
            end_idx = start_idx + betas.shape[0]
            concatenated_betas[start_idx:end_idx] = betas
            start_idx = end_idx
        else:
            print(f"Warnung: {session_file_path} nicht gefunden. Überspringe...")

    # Sicherstellen, dass die Datei auf die Festplatte geschrieben wird
    concatenated_betas.flush()

    # **Jetzt die `.memmap`-Datei als reguläre `.npy` speichern**
    final_array = np.array(
        concatenated_betas
    )  # Memmap in reguläres NumPy-Array konvertieren
    np.save(concatenated_file_path, final_array)  # Endgültige Speicherung als `.npy`

    # **Temporäre `memmap`-Datei löschen**
    os.remove(temp_memmap_file)

    print(f"Gespeicherte Datei: {concatenated_file_path} im kompatiblen `.npy`-Format")


def extract_nsd_data(config: Configuration):
    subject_list = subjects_list_unifier(
        config.pipeline.step_1_preprocessing.subjects, True
    )

    if config.pipeline.step_1_preprocessing.extract_nsd_data:
        logging.info(logging_message(1, "Starting *NSD preprocessing and extraction*"))
        for subject in subject_list:
            process_subject(subject)
            concatenate_subject_sessions(subject)
    else:
        logging.info(logging_message(1, "Skipping *NSD preprocessing and extraction*"))


if __name__ == "__main__":
    for subject in SUBJECTS:
        extract_nsd_data(subject)
