# Master Thesis Maximilian Schwalenberg

This is the repository that contains all the code that was developed during my masters thesis with Dr. Ben Harvey.

**Title: Modelling relationships between neural responses and cortical organisation**

I would like to thank Stan and Luis, whose previous work and existing code was fundamental for the success of this project.

# 1. Steps

Important to note: All of the steps explained below run on the config file. It takes care of paths and file naming. It is designed to run on the remote machine in Ben's office.

Running on your machine requires the modification of paths to adjust to your own machine.

At a later point in the readme the setup of the config is explained in greater detail.

## 1.1 Dataset creation

Assuming no data is downloaded yet, following preparing steps need to be taken.

- Download all coco images that are used in the NSD dataset (only images that all participants have seen). [Code: dataset-creation/create_data_split.py](dataset-creation/create_data_split.py)

- Generate a positive and a negative set of those images (whose responses should ultimately be compared). In this case sets are created that contain faces (positive) and only objects (negative). This is done in several steps:
  - run all the images through a face detector and save the results in a .json file
    [Code: dataset-creation/face_detection.py](dataset-creation/face_detection.py)
  - automatically create positive/negative set by running analysis on face detection result [Code: dataset-creation/generate_faces_set.py](dataset-creation/generate_faces_set.py)
  - the automatically created sets are far from perfect (especially the faces-set) so a script was developed to quickly manually correct the set [Code: dataset-creation/manually_label_images.py](dataset-creation/manually_label_images.py) You have to refer to the code and change a few paths for the according set. (MAYBE **TODO** TO MAKE IT EASIER)

## 1.2 Prepare Beta values

Stan generated full brain beta-files for each participant across all sessions.

They are normalized in a certain kind of way. At this point, I just take them as is.
The preprocessing/normalizing might need to be questioned at a later stage.

Each file is very large (>50GB). When the neural responses are processed for single images, it is very unpractical to load all this data in the RAM (impossible) and work with it.

A script to extract (only!) the responses belonging to the actual sets is [here: load_betas.py](load_betas.py)
This script takes a very long time to run but is already heavily optimized (around 4h). It only needs to be run once for each final set of data.

## 1.3 T-Testing

Having extracted all the required beta values from the participants for all the specified samples, t-values are now generated using the two distinct sets.

This script generates the t-testing results.
[t_testing.py](t_testing.py)

They are later used in matlab for further analysis.

## 1.4 Mask-Creation

Create mask based on the results of the t-tests.

## 1.5

RSA
