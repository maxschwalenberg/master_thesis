import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# from utils.utils import *

from utils.utils import retrieve_stacked_betas#, retrieve_stacked_betas_test
from utils.config import Configuration, load_config

# from utils.split_condition import split_conditions
# from nsddatapaper_rsa.utils.nsd_get_data import get_conditions, get_betas


def compute_noise_ceilling(config: Configuration, shared_set: bool, subject: int):
    """
    Computes the noise ceilling per voxel
    """
    subject_to_pick = "shared" if shared_set else f"subj_{subject:02d}"

    # shape --> (n_images, n_voxels)
    betas_train, _, _ = retrieve_stacked_betas(
        config, subject, "averaged", 0, subj_to_check=subject_to_pick
    )

    betas_test, _, _ = retrieve_stacked_betas(
        config, subject, "averaged", 0, subj_to_check=subject_to_pick, test=True
    )
    # betas_test = retrieve_stacked_betas_test(
    #     config, subject, subj_to_check=subject_to_pick
    # )

    results = []
    noise_ceilling_all_vox = np.zeros(betas_train.shape[1])

    # loop through each voxel (and n == n_dim_images)
    for row_i in tqdm(range(betas_train.shape[1])):
        current_row = np.array([betas_train[:, row_i]])

        new_row, _ = rescale(current_row, betas_test[:, row_i])

        rss = np.sum((new_row - betas_test[:, row_i]) ** 2)

        variance = np.sum((betas_test[:, row_i] - np.mean(betas_test[:, row_i])) ** 2)
        ve_voxels = 1 - rss / variance  # noise ceilling of all voxels in current roi

        noise_ceilling_all_vox[row_i] = ve_voxels

    # np.save(noise_ceilling_file, noise_ceilling_all_vox)


def rescale(train, test):
    """
    Input
    -------
    train: 1D array of length n, to rescale
    test : 1D array of length n, target to rescale to

    Output
    --------
    new_train: 1D array of length n, rescale train array

    Rescale the train array to the test array. Linear rescale using pseudoinverse
    """
    if len(train.shape) == 1:
        train = np.array([train])  # add a dimension for concat
    # print(train.shape)
    train_ones = np.concatenate((train, np.ones((train.shape[0], train.shape[1])))).T
    scale = np.linalg.pinv(train_ones) @ test.T

    new_train = train_ones @ scale  # make
    return new_train.T.squeeze(), scale


if __name__ == "__main__":
    config = load_config("config.yaml")
    compute_noise_ceilling(config, False, 1)
