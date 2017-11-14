# working_memory

## Experiment:
1. load X (i.e. 2,3,4,5,6,...): in a trial, subject sees a sequence of scene images (including natural environments), and each image is present 1.4 seconds. 
2. after the last image in the sequence, there is a 6 seconds blank screen, with no cross. Namely the delay period.
3. after 6 seconds, a probe is present. response recorded.
4. a scramble of the probe is present. 

## Analysis:
The analysis present here mainly focuses on the delay period (6 seconds long). We present load 2 and load 5 conditions to subjects, and load 2 and load 5 have different number of trials (80 vs 40). The goal is to fit a linear classifier (SVM) at individual time samples and test the classifier at the same time sample. Features are the channels and the rows are the trials. 
