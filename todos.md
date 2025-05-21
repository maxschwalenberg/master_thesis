- &check; integrate correct visualization in matlab (hotcoolmap oder so)

- &check; thresholding (done in python script)
- &check; crop ROI

- &check; new negative set (animate but non-face)

- &cross; create RSA using ROI

  - extract only needed betas &check;
  - create RDM/MDS &check;
  - assign each point original image &check;
  - sanity check &cross;

- fix bug: &check;

  - creating t-test results for new set (animate-non-face) results in error for third subject (maybe not all files are existent? then issue with load_betas script probably)
  - File "/media/Working/master_thesis/t_testing.py", line 93, in set_t_testing
    file_path = files[0]
    IndexError: list index out of range

- &check; plot with img instead of points (cropped face instead of original image)

- &check; see how repeatable compare dissimilarity matrices

- gaussian function fitting per voxel (how well variance in test set explained)
  - input : positions of MDS
  - fit for every voxel the function
  - training set: averaged 2 samples if 3 in total

## Potential Issues

- in the RDM creation: I am neglecting NaNs more or less

### 31.01.25

- Surfclust
- afni

- sanity checks

  - gaussian fitting -> search within polar space
  - use gaussian predictions for test data

- generate common plot of response amplitudes and fits

- analyze ROI 7 (FFA)

- include intercept fitting
- remove inhibitory ROI for now

- separating brain areas by responses of binary classes as opposed to using prf visual field maps

- new subject results
- variance map
- fitting results
- MDS space visualizations (images; voxel fits)

- same colormap 2d view (done)
- rescaled test variance explained (either both rescaled or both NOT rescaled)
- plots for variances explaineded and sigmas
- create rois and sets across participants

- allow rescaling of fitted gaussian

improved and more conclusive statistical testing: https://www.sciencedirect.com/science/article/pii/S089662732100845X

rescaling on V1

- retrieve_roi_mask .... add parametrization to t_test_value

- compare metrics (test variance, unscaled noise ceiling) to no rescaled data (stans)
- compare metrics for scaling based on significant voxels as opposed to V1

- model performance non rescaled include in script

labeled data:
1, 2, 3, 4, 5

labeled ROIs:
1, 4

- violin plot
- permutation analysis of pairwise MDS distance and bounding box distances

- check permutations with other masks as well

---

- compare position of face to other features (viewing angle? - roll/yaw of face, age, size of bounding box, distance between eyes, nose size, .... anatomical features)

- results writeup of all the done analyses

- if multiple significant features: GLM of these features in MDS space?

- compare results with RSA in neural network
