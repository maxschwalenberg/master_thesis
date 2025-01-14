- &check; integrate correct visualization in matlab (hotcoolmap oder so)

- &check; thresholding (done in python script)
- &check; crop ROI

- &check; new negative set (animate but non-face)

- &cross; create RSA using ROI

  - extract only needed betas &check;
  - create RDM/MDS &check;
  - assign each point original image &check;
  - sanity check &cross;

- fix bug:

  - creating t-test results for new set (animate-non-face) results in error for third subject (maybe not all files are existent? then issue with load_betas script probably)
  - File "/media/Working/master_thesis/t_testing.py", line 93, in set_t_testing
    file_path = files[0]
    IndexError: list index out of range

- plot with img instead of points (cropped face instead of original image)

- see how repeatable compare dissimilarity matrices

- gaussian function fitting per voxel (how well variance in test set explained)
  - input : positions of MDS
  - fit for every voxel the function

## Potential Issues

- in the RDM creation: I am neglecting NaNs more or less
