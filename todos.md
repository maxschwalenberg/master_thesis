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

I am working with the Natural Scenes Dataset of fmri data. I am performing various experiments. It is important that I am able to have nice preprocessed data. However, I have some problems - participants having seen an image leads to a neural response which is captured using fmri (bold). Wouldnt it be expected that the same image results in a similar stimulus?

My analysis so far is not really complying with this. How can I align responses to the same stimuli to one another?

so in essence I need insights into the following:

- how well does fmri reflect the neural responses? what are the disadvantages and what noisiness/false responses is there to be excpeted
- how can neural responses (fmri) be aligned to each other? what is the state of the art or how are people doing it generally?
