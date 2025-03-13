# Master Thesis Maximilian Schwalenberg

This is the repository that contains all the code that was developed during my masters thesis with Dr. Ben Harvey.

**Title: Modelling relationships between neural responses and cortical organisation**

I would like to thank Stan and Luis, whose previous work and existing code was fundamental for the success of this project.

# 1. Steps

Important to note: All of the steps explained below run on the config file. It takes care of paths and file naming. It is designed to run on the remote machine in Ben's office.

Running on your machine requires the modification of paths to adjust to your own machine.

At a later point in the readme the setup of the config is explained in greater detail.

## **1.0 NSD Data Processing**

### **Overview**

This step processes the NSD (Natural Scenes Dataset) to prepare it for further analysis. The preprocessing involves structuring the data in a usable format and ensuring consistency across all participants.

**Script:** `nsd_processing/nsd_preprocessing.py`

**Execution:**

```bash
python nsd_processing/nsd_preprocessing.py
```

## **1.1 Labeling Processing**

### **Overview**

The labeling processing step consists of multiple stages that filter, detect, and validate image samples based on their category and the presence of faces. This process ensures the dataset is clean and structured properly for downstream tasks.

---

### **Pipeline Steps**

#### **Step 1: Data Filtering and Download**

**Script:** `datasetCreation/create_data_split.py`

This script performs the following actions:

- Filters out **animate** and **non-animate** images based on COCO dataset labels.
- Downloads the required images based on the filtered categories.

**Execution:**

```python
if __name__ == "__main__":
    config = load_config("config.yaml")
    download_data(config)  # Downloads required images
    create_data_split(config)  # Splits data into animate and non-animate subsets
```

---

#### **Step 2: Face Detection**

**Script:** `datasetCreation/face_detection.py`

This script identifies images that contain faces within the animate subset.

**Execution:**

```python
if __name__ == "__main__":
    config = load_config("config.yaml")
    generate_face_detection_results(config)  # Detects faces in animate images
```

---

#### **Step 3: Generating Face and Non-Face Sets**

**Script:** `datasetCreation/generate_faces_set.py`

This script processes the results of face detection to create:

1. A **positive set** (images with detected faces).
2. An **animate-non-face set** (images without faces).

**Execution:**

```python
if __name__ == "__main__":
    config = load_config("config.yaml")
    generate_animate_non_face(config)  # Generates non-face animate set
    generate_positive_set(config)  # Generates positive face set
```

---

#### **Step 4: Removing NaNs & Checking Missing Samples**

**Script:** `datasetCreation/check_labelling.py`

This script ensures:

- Removal of NaN values from labeling data.
- Detection of missing samples across participants.
- Differentiation between **shared sets** and **unique subject data**.

🚨 **Current Issue:**  
The script does not yet differentiate between the shared set and unique participant sets. This needs improvement.

**Execution:**

```python
if __name__ == "__main__":
    config = load_config("config.yaml")

    assert len(config.nsd_samples_subjects_to_check) == 1

    subj_to_pick = (
        "shared"
        if config.nsd_samples_subjects_to_check[0] == "shared"
        else f"subj_{int(config.nsd_samples_subjects_to_check[0]):02d}"
    )

    if subj_to_pick == "shared":
        check_labels(config)
    else:
        missing_subjects_json_path = os.path.join(
            config.excel_files_target_dir, subj_to_pick, "missing_subjects.json"
        )
        with open(missing_subjects_json_path, "w") as f:
            json.dump({}, f)

    adjust_labelled_data(config, subj_to_pick)

    correct_sets_for_nans_and_missing_samples(config, subj_to_pick)
```

---

#### **Step 5: Manual Labeling Verification**

**Script:** `datasetCreation/manually_label_images.py`

After automatic labeling and cleanup, the final step is to manually verify and adjust the labels.

**Execution:**

```bash
streamlit run datasetCreation/manually_label_images.py
```

- This launches a **Streamlit app** for manual labeling.
- Ensures the final dataset is **correct and properly annotated** before proceeding.

---

## **Dataset Filenames at Each Stage**

| Step       | Filename (Config Key)                                                      | Description                                         |
| ---------- | -------------------------------------------------------------------------- | --------------------------------------------------- |
| **Step 1** | `subset_non_animate`, `subset_animate`                                     | Initial data split into animate/non-animate images. |
| **Step 2** | `subset_animate_face_unchecked`, `subset_animate_non_face_unchecked`       | Face detection results (unchecked).                 |
| **Step 3** | `subset_animate_face_nans_removed`, `subset_animate_non_face_nans_removed` | Cleaned datasets (NaNs removed).                    |
| **Step 4** | `subset_animate_face_labelled`, `subset_animate_non_face_labelled`         | Checked and labeled datasets.                       |
| **Step 5** | `subset_animate_face_final`, `subset_animate_non_face_final`               | Final cleaned and validated dataset.                |

---

## **1.2 Prepare Beta Values**

Stan generated full brain beta-files for each participant across all sessions.

They are normalized in a certain kind of way. At this point, I just take them as is.
The preprocessing/normalizing might need to be questioned at a later stage.

Each file is very large (>50GB). When the neural responses are processed for single images, it is very unpractical to load all this data in the RAM (impossible) and work with it.

A script to extract (only!) the responses belonging to the actual sets is [here: load_betas.py](load_betas.py)
This script takes a very long time to run but is already heavily optimized (around 4h). It only needs to be run once for each final set of data.

## **1.3 T-Testing**

Having extracted all the required beta values from the participants for all the specified samples, t-values are now generated using the two distinct sets.

This script generates the t-testing results.
[t_testing.py](t_testing.py)

They are later used in Matlab for further analysis.

## **1.4 Mask-Creation**

Create mask based on the results of the t-tests.
For this, go into Matlab, load the t-testing results and create regions of interests around areas of high statistical significance.

Now, multiple ROIs should be available, used for further analysis.

## **1.5 Create Representational Sampling Space**

Creating RDMs among the ROIs - create dissimilarity matrices.
From this, map into 2D space using MDS.

[Code: rsa/create_rdm.py](rsa/create_rdm.py)

## **1.6 Gaussian Fitting**

For each voxel in the ROIs, fit a Gaussian to the MDS space. (Hypothesizing that single voxels/neurons can capture single features in the MDS space)

[Code: fit_gaussian.py](fit_gaussian.py)

---

## **Running the Full Processing Pipeline**

To execute all steps sequentially:

```bash
python datasetCreation/create_data_split.py
python datasetCreation/face_detection.py
python datasetCreation/generate_faces_set.py
python datasetCreation/check_labelling.py
streamlit run datasetCreation/manually_label_images.py


python load_betas.py
python t_testing.py
```

This ensures the dataset is fully processed, checked, and manually verified before further usage.

---

**✅ Documentation complete!** 🚀
