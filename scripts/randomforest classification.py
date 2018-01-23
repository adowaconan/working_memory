# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:25:08 2018

@author: ning
"""

import mne
import os
import matplotlib.pyplot as plt
import re
from glob import glob
import numpy as np
from tqdm import tqdm


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mne.decoding import Vectorizer
from sklearn.model_selection import StratifiedKFold
working_dir = 'D:\\working_memory\\data_probe_train_test\\'

"""probe"""
files_probe = glob(os.path.join(working_dir,'probe/*.fif'))
files_delay = glob(os.path.join(working_dir,'delay/*.fif'))

def make_clf(estimator,vec=True):
    clf = []
    if vec:
        clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf
cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=12345)
interval = 50 # 50 ms window
clfs = {}
scores = {}
for probe,delay in zip(files_probe,files_delay):
    epochs_probe = mne.read_epochs(probe)
    epochs_delay = mne.read_epochs(delay)
    sub,load,day = re.findall('\d+',probe)
    title = 'sub%s, day%s,load2'%(sub,day)
    clfs['sub%s, day%s,load2'%(sub,day)] = []
    scores['sub%s, day%s,load2'%(sub,day)] = []
    estimator = RandomForestClassifier(n_estimators=100,random_state=12345,class_weight='balanced')
    train_data = epochs_probe.get_data()[:,:,50:]
    train_labels = epochs_probe.events[:,-1]
    test_data = epochs_delay.get_data()[:,:,50:]
    test_labels = epochs_delay.events[:,-1]
    chunk_idx = np.arange(0,train_data.shape[-1],interval)
    chunks = np.vstack((chunk_idx[:-1],chunk_idx[1:])).T
    for chunk in tqdm(chunks,desc='chunks'):
        start,stop = chunk
        temp_train = train_data[:,:,start:stop] 
#        print(temp_train.shape)
        temp_clfs = []
        temp_scores = []
        for train,_ in cv.split(temp_train,train_labels):
            temp_train_split = temp_train[train]
            temp_train_labels = train_labels[train]
            clf = make_clf(estimator)
            clf.fit(temp_train_split,temp_train_labels)
            temp_clfs.append(clf)
            
            
        clfs['sub%s, day%s,load2'%(sub,day)].append(temp_clfs)



























