# Master Thesis: Modelling Relationships Between Neural Responses and Cortical Organisation

**Author:** Maximilian Schwalenberg  
**Supervisor:** Dr. Ben Harvey  
**Title:** Modelling relationships between neural responses and cortical organisation

## Acknowledgments

I would like to thank Stan and Luis, whose previous work and existing code was fundamental for the success of this project.

## Repository Overview

This repository contains all the code developed during my master's thesis research. The project investigates how neural responses in the visual cortex relate to cortical organization using the Natural Scenes Dataset (NSD). The codebase is comprehensive and follows a multi-step pipeline from data preprocessing to final analysis.

**Note:** This repository contains extensive code and is therefore quite complex. I have attempted to maintain organization and follow basic software engineering guidelines, though some violations may exist due to the dynamic nature of the project and evolving requirements. For any questions, please don't hesitate to contact me:

- **Email:** maximilian.schwalenberg@protonmail.com
- **LinkedIn:** [Max Schwalenberg](https://www.linkedin.com/in/max-schwalenberg-60436b200/)

## System Requirements

- Python 3.8+
- MATLAB (for ROI creation and mask generation)
- Streamlit (for manual labeling verification)
- Required Python packages (see requirements.txt)
- Significant storage space (>500GB recommended for full NSD dataset)
- High-performance computing resources recommended

## Configuration

**Important:** All steps run using a central configuration file (`config.yaml`) that manages paths and file naming. The current setup is designed for the remote machine in Ben's office. Running on your own machine requires modifying paths in the config file to match your local environment.

Refer to the config setup section below for detailed configuration instructions.

---

# Processing Pipeline

## 0.0 Getting the Data

### Natural Scenes Dataset (NSD) Download

The Natural Scenes Dataset must be downloaded before beginning analysis. The NSD contains fMRI responses to natural images from multiple participants across multiple sessions.

**Download Instructions:**
1. Visit the [NSD Data Portal](http://naturalscenesdataset.org/)
2. Register for data access
3. Download the following components:
   - Beta files for all participants
   - Stimulus images
   - COCO annotations
   - Participant metadata

**Required Data Structure:**
```
data/
├── nsd_betas/
│   ├── subj01/
│   ├── subj02/
│   └── ...
├── nsd_stimuli/
├── coco_annotations/
└── metadata/
```

---

## 1.0 NSD Data Processing

### Overview

This step processes the NSD to prepare it for further analysis. The preprocessing involves structuring the data in a usable format, ensuring consistency across all participants, and preparing the neural response data for subsequent analyses.


**Script:** `nsd_processing/nsd_preprocessing.py`

**Execution:**
```bash
python nsd_processing/nsd_preprocessing.py
```

---

## 1.1 Labeling Processing

### Overview

The labeling processing step consists of multiple stages that filter, detect, and validate image samples based on their category and the presence of faces. This process uses COCO dataset labels to categorize images and ensures the dataset is clean and properly structured for downstream analyses. The pipeline distinguishes between animate (containing living beings) and non-animate objects, with special attention to face detection within animate images.

### Pipeline Steps

#### Step 1: Data Filtering and Download

**Script:** `datasetCreation/create_data_split.py`

This script performs the following actions:
- Filters images into **animate** and **non-animate** categories based on COCO dataset labels
- Animate categories include: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Non-animate categories include: vehicle, furniture, electronic, sports equipment, and kitchen items
- Downloads the required images based on the filtered categories
- Creates initial data splits for subsequent processing

**Execution:**
```python
if __name__ == "__main__":
    config = load_config("config.yaml")
    download_data(config)  # Downloads required images
    create_data_split(config)  # Splits data into animate and non-animate subsets
```

#### Step 2: Face Detection

**Script:** `datasetCreation/face_detection.py`

This script identifies images that contain faces within the animate subset using computer vision techniques. Face detection is crucial for understanding neural responses to social stimuli.

**Key Features:**
- Uses robust face detection algorithms
- Processes only animate images for efficiency
- Handles multiple faces per image

**Execution:**
```python
if __name__ == "__main__":
    config = load_config("config.yaml")
    generate_face_detection_results(config)  # Detects faces in animate images
```

#### Step 3: Generating Face and Non-Face Sets

**Script:** `datasetCreation/generate_faces_set.py`

This script processes the results of face detection to create:
1. A **positive set** (images with detected faces)
2. An **animate-non-face set** (animate images without faces, e.g., animals, body parts)

This separation allows for targeted analysis of face-selective neural responses versus general animate object responses.

**Execution:**
```python
if __name__ == "__main__":
    config = load_config("config.yaml")
    generate_animate_non_face(config)  # Generates non-face animate set
    generate_positive_set(config)  # Generates positive face set
```

#### Step 4: Removing NaNs & Checking Missing Samples

**Script:** `datasetCreation/check_labelling.py`

This script ensures data quality by:
- Removing NaN values from labeling data
- Detecting missing samples across participants
- Differentiating between **shared sets** (images viewed by all participants) and **unique subject data**
- Validating data integrity across the pipeline

**Quality Checks:**
- Cross-participant data consistency
- Missing data identification and handling
- Label validation and correction

#### Step 5: Manual Labeling Verification

**Script:** `vis.py`

After automatic labeling and cleanup, the final step involves manual verification and adjustment of labels to ensure accuracy.

**Execution:**
```bash
streamlit run vis.py
```

**Features:**
- Launches a **Streamlit app** for interactive manual labeling
- Allows visualization of images with their assigned labels
- Enables correction of misclassified images
- Ensures the final dataset is **correct and properly annotated** before proceeding

**Note:** The subject string and set being checked need to be adjusted manually in the code before running the verification app.

### Dataset Filenames at Each Stage

| Step | Filename (Config Key) | Description |
|------|----------------------|-------------|
| **Step 1** | `subset_non_animate`, `subset_animate` | Initial data split into animate/non-animate images |
| **Step 2** | `subset_animate_face_unchecked`, `subset_animate_non_face_unchecked` | Face detection results (unchecked) |
| **Step 3** | `subset_animate_face_nans_removed`, `subset_animate_non_face_nans_removed` | Cleaned datasets (NaNs removed) |
| **Step 4** | `subset_animate_face_labelled`, `subset_animate_non_face_labelled` | Checked and labeled datasets |
| **Step 5** | `subset_animate_face_final`, `subset_animate_non_face_final` | Final cleaned and validated dataset |


**Important**:

This is handled a bit bad but was done as to ensure a further manual validation step by myself - the final file (step 5) is created by generating a copy of the file in step 4 and just renaming it properly (as defined in config file). 
It forced me to manually accept the final labelled version which ensured correctness - as from now on the analses are run mostly automatically and this step was critical. 

---

## 1.2 Prepare Beta Values

### Overview

Stan generated full brain beta-files for each participant across all sessions. These files contain the neural response estimates (beta values) for each voxel in response to each stimulus. The beta values are normalized using a specific method, which we accept as-is for this analysis (though the preprocessing/normalization approach may need to be questioned in future work).

### Challenge and Solution

Each beta file is very large (>50GB), making it impractical or impossible to load all data into RAM simultaneously. To address this, we extract only the neural responses corresponding to our final image sets.

**Script:** `datasetCreation/load_betas.py`

**Key Features:**
- Extracts beta values only for images in our final datasets
- Heavily optimized for memory efficiency
- Runtime: approximately 4 hours (machine-dependent)
- Needs to be run only once for each final set of data
- Significantly reduces data storage and processing requirements

**Process:**
1. Loads beta files in chunks to avoid memory overflow
2. Matches image IDs with corresponding beta values
3. Extracts and saves only relevant neural responses
4. Creates manageable datasets for downstream analysis

---

## 1.3 T-Testing

### Overview

Having extracted all required beta values from participants for specified samples, t-values are now generated using statistical comparisons between the two distinct sets (face vs. non-face animate images). This analysis identifies voxels that show significantly different responses between conditions.

**Script:** `t_testing/t_testing.py`

**Statistical Analysis:**
- Performs voxel-wise t-tests comparing face and non-face conditions
- Accounts for multiple comparisons using appropriate corrections
- Generates statistical maps showing regions of significant difference
- Outputs t-values, p-values, and effect sizes

**Results:**
- T-testing results are saved for further analysis
- These statistical maps are later imported into MATLAB for ROI creation
- Identifies candidate regions for detailed analysis

---

## 1.4 Mask Creation

### Overview

Based on t-testing results, create regions of interest (ROIs) around areas of high statistical significance. This step transitions from Python to MATLAB for specialized neuroimaging analysis tools.

**Process:**
1. Load t-testing results in MATLAB
2. Identify regions with significant t-values
3. Create spatial clusters of significant voxels
4. Define anatomical ROIs around areas of interest
5. Validate ROIs against known cortical anatomy

**Expected Output:**
- ROI masks for subsequent analyses



---

## 1.5 RSA (Representational Similarity Analysis)

### Sample Repeatability

**Script:** `rsa/sample_repeatability.py`

This analysis examines the consistency of neural representations across repeated presentations of the same stimuli. Sample repeatability is crucial for validating the reliability of neural response measurements.


### Permutation Analysis

**Script:** `rsa/permutation_analysis.py`

This analysis uses permutation testing (mantel test) to establish statistical significance of representational patterns.


---

## 1.6 Gaussian Fitting

### Overview

For each voxel in the ROIs, fit a Gaussian function to the multidimensional scaling (MDS) space. This analysis tests the hypothesis that individual voxels/neurons can capture single features in the representational space, with responses following a Gaussian distribution around preferred stimulus features.

**Script:** `gaussian/fit_gaussian.py`

**Methodology:**
- Perform MDS on neural response patterns
- Fit 2D Gaussian functions to voxel response profiles in MDS space
- Extract Gaussian parameters: center, width, orientation, amplitude
- Assess goodness of fit for each voxel

**Parameters Extracted:**
- **Center:** Preferred position in stimulus space
- **Width:** Selectivity/tuning breadth
- **Orientation:** Preferred stimulus dimensions
- **Amplitude:** Response magnitude

**Applications:**
- Understanding feature selectivity of individual voxels
- Mapping cortical organization in stimulus space
- Comparing neural tuning properties across regions

---

## 1.7 Cortical Correlation

**Script:** `rsa/cortical_correlation.py`

### Overview

This analysis investigates the relationship between positions in MDS representational space (derived from Gaussian fits) and physical positions on the cortical surface. This addresses a fundamental question about the spatial organization of feature representations in visual cortex.

**Key Analyses:**
- Correlate MDS coordinates with cortical surface coordinates
- Test for systematic organization of feature preferences
- Examine gradients in representational space across cortex
- Compare organization patterns across different ROIs

**Methodological Approach:**
- Extract cortical coordinates for each voxel
- Map Gaussian centers from MDS space to cortical positions
- Calculate spatial autocorrelation in feature preferences
- Statistical testing for significant organization patterns

**Expected Findings:**
- Gradual changes in feature preferences across cortical surface
- Clustering of similar feature preferences
- Relationship between cortical distance and representational distance

---

## 1.8 Analysis on Neural Networks

**Script:** `rsa/neural_network.py`

### Overview

Apply similar representational analysis methodology to layers of artificial neural networks to compare biological and artificial visual processing. This comparative approach helps understand computational principles underlying visual representation.

---

## Running the Full Processing Pipeline

### Two-Phase Execution

The pipeline requires two distinct phases due to the manual ROI creation step in MATLAB:

Use `main.py` with step-by-step execution by adjusting the `execute` flags in `config.yaml`. Note that even if all steps are set to `execute=True`, the pipeline won't run completely due to the required manual ROI creation step.


Refer to `final_run.py` to get a complete overview of **all** the performed analyses and methods.

### Configuration Management

The `config.yaml` file controls:
- File paths for all input/output data
- Analysis parameters
- Step execution flags
- Computational settings (memory limits, parallel processing)

---

## Results and Visualization

### Comprehensive Results Analysis

**Notebook:** `all_results.ipynb`

This Jupyter notebook creates all results plots and figures seen in the thesis. It assumes all previous processing steps have been completed successfully.


---

## File Organization

```
project_root/
├── config.yaml                    # Main configuration file
├── main.py                        # Primary pipeline executor
├── final_run.py                   # Post-ROI analysis runner
├── vis.py                         # Manual labeling verification app
├── requirements.txt               # Python dependencies
├── nsd_processing/
│   └── nsd_preprocessing.py
├── datasetCreation/
│   ├── create_data_split.py
│   ├── face_detection.py
│   ├── generate_faces_set.py
│   ├── check_labelling.py
│   └── load_betas.py
├── t_testing/
│   └── t_testing.py
├── rsa/
│   ├── sample_repeatability.py
│   ├── permutation_analysis.py
│   ├── cortical_correlation.py
│   └── neural_network.py
├── gaussian/
│   └── fit_gaussian.py
├── utils/
│   └── ....
├── data/
└── all_results.ipynb

```

**✅ Documentation complete!** 🚀

For additional questions or support, please contact maximilian.schwalenberg@protonmail.com