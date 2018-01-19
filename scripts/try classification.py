# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:37:23 2018

@author: ning
"""

########### machine learning #####################
import mne
import os
from glob import glob
from mne.decoding import LinearModel,Vectorizer,SlidingEstimator,get_coef,cross_val_multiscore,GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

data_dir = 'D:\\working_memory\\data_probe_train_test\\probe'
test_data_dir = 'D:\\working_memory\\data_probe_train_test\\delay'
files_probe = glob(os.path.join(data_dir,'*.fif'))
files_delay = glob(os.path.join(test_data_dir,'*.fif'))

for f,d in zip(files_probe,files_delay):
    print(f,d)

n = 0

epochs = mne.read_epochs(files_probe[n])
epochs_test = mne.read_epochs(files_delay[n])
epochs_test.set_channel_types({'LOc':'eog','ROc':'eog'})
epochs_test.pick_types(meg=False,eeg=True,eog=False)
ch_names = epochs.ch_names
cv = KFold(n_splits=3,shuffle=True,random_state=12345)# 4 folds cross validation
train_data = epochs.get_data()
train_labels = epochs.events[:,-1]
test_data = epochs_test.get_data()
test_labels = epochs_test.events[:,-1]

def make_clf(estimator):
    """
    Takes no argument, and return a pipeline of linear classifier, containing a scaler and a linear estimator
    """
    clf = []
    clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    estimator = estimator
#    estimator = LinearModel(estimator) # extra step for decoding patterns
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf 
estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced') 
clf = make_clf(estimator)
td = SlidingEstimator(clf,scoring='roc_auc',)

td.fit(train_data,train_labels)
pred_ = td.predict(test_data[:,:,:2001])


































































#data = epochs.get_data()
#labels = epochs.events[:,-1]
#def make_clf():
#    """
#    Takes no argument, and return a pipeline of linear classifier, containing a scaler and a linear estimator
#    """
#    clf = []
##    clf.append(('vectorizer',Vectorizer()))
#    clf.append(('scaler',StandardScaler()))
#    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
##    estimator = LinearModel(estimator) # extra step for decoding patterns
#    clf.append(('estimator',estimator))
#    clf = Pipeline(clf)
#    return clf  
#clf = make_clf()
#td = SlidingEstimator(clf,scoring='roc_auc',)
#predictions = []
#for train,test in cv.split(data,labels):
#    td = SlidingEstimator(clf,scoring='roc_auc',)
#    td.fit(data[train],labels[train])
#    pred_ = td.predict(data[test])
#    predictions.append([pred_,labels[test]])
#
#scores = []
#for pred_,test_labels in predictions:
#    pred_,test_labels
#    scores_ = [metrics.roc_auc_score(test_labels,p) for p in pred_.T]
#    scores.append(scores_)
#scores = np.array(scores)
#
#scores_mean = scores.mean(0)
#scores_std = scores.std(0)
#times = np.linspace(0,2000,num=scores.shape[1])
#
#fig,ax  = plt.subplots(figsize=(12,8))
#ax.plot(times,scores_mean,color='black',alpha=1.,label='decoding mean')
#ax.fill_between(times,scores_mean-scores_std,scores_mean+scores_std,color='red',alpha=0.3,label='decoding std')
#ax.axhline(0.5,linestyle='--',color='blue',alpha=0.6,label='reference')
#ax.set(xlabel='time (ms)',ylabel='ROC AUC',xlim=(0,2000),ylim=(0.3,.8),title="sub%s_load%s_day%s"%(sub,condition[1],day))
#ax.legend()
#
############ pattern decoding ##############
#def make_clf():
#    """
#    Takes no argument, and return a pipeline of linear classifier, containing a scaler and a linear estimator
#    """
#    clf = []
#    clf.append(('vectorizer',Vectorizer()))
#    clf.append(('scaler',StandardScaler()))
#    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
#    estimator = LinearModel(estimator) # extra step for decoding patterns
#    clf.append(('estimator',estimator))
#    clf = Pipeline(clf)
#    return clf  
#clf = make_clf()
#
#patterns = []
#for train,test in cv.split(data,labels):
#    clf.fit(data[train],labels[train])
#    patterns_ = get_coef(clf,attr='patterns_',inverse_transform=True)
#    patterns.append(patterns_)
#patterns = np.array(patterns)
#patterns_mean = patterns.mean(0)
#info = epochs.info
#evoke = mne.EvokedArray(patterns_mean,info,)
#epochs['non target probe'].average().plot_joint(title='non target probe')
#epochs['target probe'].average().plot_joint(title='target probe')
#evoke.plot_joint(title='pattern difference between target and non target probes')
#
#
#
#
#pred_ = clf.predict(data[test])
#print(metrics.classification_report(labels[test],pred_))

