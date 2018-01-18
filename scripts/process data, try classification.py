# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:20:49 2018

@author: ning
"""

import os
from glob import glob
import re
import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

working_dir = 'D:\\working_memory\\'
data_dir = 'D:\\working_memory\\data_probe_train_test'
result_dir = 'D:\\working_memory\\probe_train_test'
evt_dir = os.path.join(working_dir+'signal detection')

for directory in [data_dir,result_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
condition = 'l2_';n = 2
files_vhdr = glob(working_dir+'*%s*.vhdr'%condition)
files_evt = glob(os.path.join(evt_dir,'*%s*'%condition))
#sub,_,day = re.findall('\d+',files_vhdr[n])
#subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
#events = pd.read_csv(subject_evt,sep='\t')
#events.columns = ['tms','code','TriNo','RT','Recode','Comnt']


"""probe condition"""
data_dir_probe = os.path.join(data_dir,'probe')
if not os.path.exists(data_dir_probe):
    os.makedirs(data_dir_probe)
    
for n in range(len(files_vhdr)):
    sub,_,day = re.findall('\d+',files_vhdr[n])
    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
    events = pd.read_csv(subject_evt,sep='\t')
    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
    if events.Recode.sum() != 0:
        recode = events.Recode
    if 27 < int(sub) <30:
        events['Recode'] = events['TriNo'].apply(lambda x:int(str(x)[-1]) if x > 80 else 0)
        events_probe = events[events.Recode != 0]
        recode = events.Recode
    elif int(sub) > 29:
        events['Recode'] = recode
    else:
        events_probe = events[events.Recode != 0]
    label_dict = {4:0,3:0,2:1,1:1}
    events_probe['labels'] = events_probe.Recode.map(label_dict)
    events_probe = events_probe.dropna()
    events_ = events_probe[['tms','code','labels']].values.astype(int)
    events_[:,1] = 0


    print(files_vhdr[n],subject_evt)
    raw = mne.io.read_raw_brainvision(files_vhdr[n],montage='standard_1020',
                                         eog=('LOc','ROc','Aux1'),preload=True)
    raw.set_channel_types({'Aux1':'stim'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False)
    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
    reject={'eog':120e-5,'eeg':50e-5}
    event_id = {'non target probe':0,'target probe':1}
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=2,baseline=(None,0),picks=picks,preload=True,reject=reject,)
    ##### fit ICA ##############################
    noise_cov = mne.compute_covariance(epochs,tmin=0,tmax=2,)#n_jobs=2)
    n_components = .99  
    method = 'extended-infomax'  
    decim = 3  
    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
                               method=method,random_state=12345,max_iter=3000,)
    picks = mne.pick_types(epochs.info,meg=False,eeg=True,eog=False)
    ica.fit(epochs,picks=picks,decim=decim,reject=reject)
    ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'],eog_criterion=0.2,)
    epochs = ica.apply(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
#    epochs.resample(128) # so that I could decode patterns
    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))
##### plot average over trials############
#epochs['non target probe'].average().plot_joint(title='non target probe')
#epochs['target probe'].average().plot_joint(title='target probe')

"""delay condition"""
#data_dir_probe = os.path.join(data_dir,'delay')
#if not os.path.exists(data_dir_probe):
#    os.makedirs(data_dir_probe)
#
#for n in range(len(files_vhdr)):
#    sub,_,day = re.findall('\d+',files_vhdr[n])
#    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
#    events = pd.read_csv(subject_evt,sep='\t')
#    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
#    events_delay = events[events['TriNo'] == 71]
#    events_delay['Recode'] = 1
#    #### make the time label for 2000 - 4000 ms
#    events_delay_1 = events_delay.copy()
#    events_delay_1['tms'] = events_delay_1['tms'] + 2000
#    events_delay_1['Recode'] = 2
#    ### make the time label for 4000 - 6000 ms
#    events_delay_2 = events_delay.copy()
#    events_delay_2['tms'] = events_delay_2['tms'] + 4000
#    events_delay_2['Recode'] = 3
#    
#    events_delay = pd.concat([events_delay,events_delay_1,events_delay_2])
#    events_delay = events_delay.sort_values('tms')
#    events_ = events_delay[['tms','RT','Recode']].values.astype(int)
#    
#    print(files_vhdr[n],subject_evt)
#    raw = mne.io.read_raw_brainvision(files_vhdr[n],montage='standard_1020',
#                                         eog=('LOc','ROc','Aux1'),preload=True)
#    raw.set_channel_types({'Aux1':'stim'})
#    raw.set_eeg_reference().apply_proj()
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False)
#    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
#    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
#    reject={'eog':120e-5,'eeg':50e-5}
#    event_id={'0-2000ms':1,'2000-4000ms':2,'4000-6000ms':3}
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
#    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=2,baseline=(None,0),picks=picks,preload=True,reject=reject,)
#    ##### fit ICA ##############################
#    noise_cov = mne.compute_covariance(epochs,tmin=0,tmax=2,)#n_jobs=2)
#    n_components = .99  
#    method = 'extended-infomax'  
#    decim = 3  
#    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
#                               method=method,random_state=12345,max_iter=3000,)
#    picks = mne.pick_types(epochs.info,meg=False,eeg=True,eog=False)
#    ica.fit(epochs,picks=picks,decim=decim,reject=reject)
#    ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'],eog_criterion=0.2,)
#    epochs = ica.apply(epochs)
#    epochs.pick_types(meg=False,eeg=True,eog=False)
##    epochs.resample(128) # so that I could decode patterns
#    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))

########### machine learning #####################
from mne.decoding import LinearModel,Vectorizer,SlidingEstimator,get_coef,cross_val_multiscore,GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import metrics

ch_names = epochs.ch_names
cv = KFold(n_splits=3,shuffle=True,random_state=12345)# 4 folds cross validation
data = epochs.get_data()
labels = epochs.events[:,-1]
def make_clf():
    """
    Takes no argument, and return a pipeline of linear classifier, containing a scaler and a linear estimator
    """
    clf = []
#    clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
#    estimator = LinearModel(estimator) # extra step for decoding patterns
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf  
clf = make_clf()
td = SlidingEstimator(clf,scoring='roc_auc',)
predictions = []
for train,test in cv.split(data,labels):
    td = SlidingEstimator(clf,scoring='roc_auc',)
    td.fit(data[train],labels[train])
    pred_ = td.predict(data[test])
    predictions.append([pred_,labels[test]])

scores = []
for pred_,test_labels in predictions:
    pred_,test_labels
    scores_ = [metrics.roc_auc_score(test_labels,p) for p in pred_.T]
    scores.append(scores_)
scores = np.array(scores)

scores_mean = scores.mean(0)
scores_std = scores.std(0)
times = np.linspace(0,2000,num=scores.shape[1])

fig,ax  = plt.subplots(figsize=(12,8))
ax.plot(times,scores_mean,color='black',alpha=1.,label='decoding mean')
ax.fill_between(times,scores_mean-scores_std,scores_mean+scores_std,color='red',alpha=0.3,label='decoding std')
ax.axhline(0.5,linestyle='--',color='blue',alpha=0.6,label='reference')
ax.set(xlabel='time (ms)',ylabel='ROC AUC',xlim=(0,2000),ylim=(0.3,.8),title="sub%s_load%s_day%s"%(sub,condition[1],day))
ax.legend()

########### pattern decoding ##############
def make_clf():
    """
    Takes no argument, and return a pipeline of linear classifier, containing a scaler and a linear estimator
    """
    clf = []
    clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
    estimator = LinearModel(estimator) # extra step for decoding patterns
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf  
clf = make_clf()

patterns = []
for train,test in cv.split(data,labels):
    clf.fit(data[train],labels[train])
    patterns_ = get_coef(clf,attr='patterns_',inverse_transform=True)
    patterns.append(patterns_)
patterns = np.array(patterns)
patterns_mean = patterns.mean(0)
info = epochs.info
evoke = mne.EvokedArray(patterns_mean,info,)
epochs['non target probe'].average().plot_joint(title='non target probe')
epochs['target probe'].average().plot_joint(title='target probe')
evoke.plot_joint(title='pattern difference between target and non target probes')




pred_ = clf.predict(data[test])
print(metrics.classification_report(labels[test],pred_))






























