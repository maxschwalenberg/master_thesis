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
