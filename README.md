# working_memory

## Experiment:
1. load X (i.e. 2,3,4,5,6,...): in a trial, subject sees a sequence of scene images (including natural environments), and each image is present 1.4 seconds. 
2. after the last image in the sequence, there is a 6 seconds blank screen, with no cross. Namely the delay period.
3. after 6 seconds, a probe is present. response recorded.
4. a scramble of the probe is present. 

## Analysis:
The analysis present here mainly focuses on the delay period (6 seconds long). We present load 2 and load 5 conditions to subjects, and load 2 and load 5 have different number of trials (80 vs 40). The goal is to fit a linear classifier (SVM) at individual time samples and test the classifier at the same time sample. Features are the channels and the rows are the trials. 

### sampling time 
fit and test the classifier at a given time sample, and we sample the time every 50 time samples. Thus, for 6000 ms long epochs, we will have 123 time samples, and thus 123 X cv-folds classifiers are fit and tested. 
The decoded topomap of the brain waves could be done on 123 time samples. In such case, we lose some (lots of) information by doing so.

### vectorization
Sub-sample the epochs by a 50 ms and we vectorize the sub-sampled data (n_trials X n_channels X 50) along the last 2 dimensions ==> n_trials X 3050. 
