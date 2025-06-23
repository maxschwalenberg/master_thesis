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


--------------
- voxel wise fitting variance explained by fitting on the betas
--> should this be done a single time? or could I mix up training/test sets and do permutations of the fitting, evaluate, 
and draw conclusions from that
- mds space cortical distance relationship
















- in utils: for t-testing ... filter out negative set animate_face
need to test function in utils and integrate into t-testing script; 
needs testing and relabelling


- test `create_rdm` shared set augmentation (happens in retrieve_stacked_betas; also test retrieve_stacked_betas_test)
--> should be working rn



---------------------------

- var test rescaling version with only offset change (amplitude fixed)



goal: want to find out if there are brain areas that ??? (dont exactly know anymore what ben said)
- challenges:
  rdm high dim ; only gonna work if there is a relationship between train/test -> RDM repeatability correlation between sets
  if there is still relationship : check for cortical surface correlation
  this is the precondition for any possible findings

- outline betas preprocessing and difference of repeatability result using different preprocessing techniques / no preprocessing technique








- idea: refactor beta loading ... extract the betas unprocessed and save the necessary stats per session in an additional file so the betas can be recomputed/processed/standardized on the fly





- bug: only one beta for subject 5,6,8 (judging by MDS/RDM results)
 --> why is the result so different compared to the graphic i have in my thesis already






 -----------------------------------
 - TODO: parameterize rsa/gaussian fit randomize order
- for p-value plot ... don't use 0.00....xxx but use < 0.05 instead


- make sure all ROIs are drawn correctly
- name ROIs differently

- regenerate the shared face detection results ... for some reason its excactly doubled



- permutation testing
  - RDM (shuffle rows of train set; keep test set the same)
  - triangle approach as well -> compare images distances not just to the own distances but all others as well
  - spearman correlation also ... shuffle cortical distances or mds distances and then repeat the correlation

- generate separate MDS spaces for the two hemispheres!!! 
----> filter mask 9 for ROI values thresholds!!

debug results by visualizing