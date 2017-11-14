# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:24:24 2017

@author: ning
"""

import os
import mne
from matplotlib import pyplot as plt
import numpy as np
import re
import pandas as pd
from mne.decoding import LinearModel,Vectorizer,SlidingEstimator,get_coef,cross_val_multiscore,GeneralizingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import KFold
def exclude_rows(x):
    if re.compile('stimulus', re.IGNORECASE).search(x):
        return False
    else:
        return True
def choose_one_event(x,key = 'delay'):
    if re.compile(key,re.IGNORECASE).search(x):
        return True
    else:
        return False
def delay_onset(x,key='71'):
    if re.compile(key,re.IGNORECASE).search(x):
        return True
    else:
        return False
    
os.chdir('D:\\working_memory\\')
subs = np.arange(11,33)
subs = np.setdiff1d(subs,np.array([24,27]))
#for sub in subs:
#    files = [f for f in os.listdir() if ('suj%d_'%(sub) in f) and ('vhdr' in f)]
#    _,_,day_ = re.findall('\d+',files[0])
#    if os.path.exists('suj%d_wml2_day%s-epo.fif'%(sub,day_)):
#        preprocessing = False
#    else:
#        preprocessing = True
#    
#    if preprocessing:
#        for eeg in files:
#            _,load,day = re.findall('\d+',eeg)
#            montage = mne.channels.read_montage('standard_1020')
#            raw = mne.io.read_raw_brainvision(eeg,eog=('LOc','ROc'),preload=True)
#            raw.set_channel_types({'Aux1':'stim','STI 014':'stim'})
#            raw.set_montage(montage)
#            raw.pick_types(meg=False,eeg=True,eog=True,stim=False)
#            raw.set_eeg_reference()
#            raw.apply_proj()
#            picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False)
#            raw.filter(1,40,picks=picks,fir_design='firwin')    
#            raw.notch_filter(np.arange(60,241,60),fir_design='firwin')
#            noise_cov = mne.compute_raw_covariance(raw,n_jobs=4)
#            
#            events_file = 'suj%d_wml%s_day%s.evt'%(sub,load,day)
#            events = pd.read_csv(events_file,sep='\t')
#            events.columns = ['TMS','code','TriNo','Comnt']
#            row_idx = events['Comnt'].apply(exclude_rows)
#            events = events[row_idx]
#            event_rows = events['Comnt'].apply(choose_one_event)
#            events = events[event_rows]
#            events_delay_onset_rows = events['Comnt'].apply(delay_onset)
#            events = events[events_delay_onset_rows]
#            events['TMS'] = events['TMS'] / 1e3
#            event_id = {'load_%s'%(load):int(load)}
#            events = events[['TMS','code','TriNo']].values.astype(int)
#            events[:,-1] = int(load)
#            
#            reject = {'eeg':100e-6,'eog':300e-6}
#            epochs = mne.Epochs(raw,events,event_id,tmin=-0.1,tmax=6,proj=True,baseline=(-0.1,0),reject=None,detrend=1,preload=True)
#            ica = mne.preprocessing.ICA(n_components=.95,n_pca_components=.95,
#                                        noise_cov=noise_cov,random_state=12345,method='extended-infomax',max_iter=int(3e3),)
#            ica.fit(raw,reject=reject,decim=3)
#            ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'])
#            epochs = ica.apply(epochs,exclude=ica.exclude,)
#            epochs.save(eeg[:-5]+'-epo.fif',)
#    
#for sub in subs:    
#    epoch_fif = [f for f in os.listdir() if ('-epo.fif' in f) and ('suj%d_'%(sub) in f)]    
#    epochs_1 = mne.read_epochs(epoch_fif[0])
#    epochs_2 = mne.read_epochs(epoch_fif[1])
#    epochs = mne.concatenate_epochs([epochs_1,epochs_2])
#    epochs.event_id = {'load_2':2,'load_5':5}
#    epochs.pick_types(eeg=True,eog=False)
#    
#    labels = np.array(epochs.events[:,-1] == 2,dtype=int)
#    data = epochs.get_data()
#    times = epochs.times
#    info = epochs.info
#    del epochs
#    #del raw
#    
#    
#    clf = []
#    #clf.append(('vectorizer',Vectorizer()))
#    clf.append(('scaler',StandardScaler()))
#    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
#    #estimator = LinearModel(estimator)
#    clf.append(('estimator',estimator))
#    clf = Pipeline(clf)
#    cv = KFold(n_splits=4,shuffle=True,random_state=12345)
#    
#    td = SlidingEstimator(clf,scoring='roc_auc',n_jobs=4)
#    scores = cross_val_multiscore(td,data,labels,cv=cv,)
#    scores_mean = np.mean(scores,axis=0)
#    scores_std = np.std(scores,axis=1)
#    plt.close('all')
#    fig,ax = plt.subplots(figsize=(20,6))
#    time_picks = times[::50]
#    scores_picks = scores_mean[::50]
#    scores_se = (scores_std/2)[::50]
#    ax.plot(time_picks,scores_picks,label='scores')
#    ax.fill_between(time_picks,scores_picks-scores_se,scores_picks+scores_se,color='red',alpha=0.5)
#    ax.axhline(0.5,color='k',linestyle='--',label='chance')
#    ax.axvline(0,color='k',linestyle='--')
#    ax.set(xlabel='Time (Sec)',ylabel='ROC AUC',xlim=(-0.1,6),ylim=(0.35,1.),title='subject_%d_load2load5_decoding_scores'%sub)
#    ax.legend()
#    fig.savefig('results/subject_%d_load2load5_decoding_scores.png'%sub,dpi=300)
#    plt.close('all')
#    
#    coef = []
#    clf = []
#    #clf.append(('vectorizer',Vectorizer()))
#    clf.append(('scaler',StandardScaler()))
#    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
#    estimator = LinearModel(estimator)
#    clf.append(('estimator',estimator))
#    clf = Pipeline(clf)
#    for train,test in cv.split(data):
#        td = SlidingEstimator(clf,scoring='roc_auc',n_jobs=4)
#        td.fit(data[train],labels[train])
#        coef_ = get_coef(td,'patterns_',inverse_transform=True)
#        coef.append(coef_)
#    evoked = mne.EvokedArray(np.mean(coef,axis=0),info,tmin=times[0])
#    evoked.save('subject_%d_load2load5_patterns-evo.fif'%sub,)
#    del evoked
#    #evoked.plot_joint(title='patterns')


for sub in subs:    
    epoch_fif = [f for f in os.listdir() if ('-epo.fif' in f) and ('suj%d_'%(sub) in f)]    
    epochs_1 = mne.read_epochs(epoch_fif[0])
    epochs_2 = mne.read_epochs(epoch_fif[1])
    epochs = mne.concatenate_epochs([epochs_1,epochs_2])
    epochs.event_id = {'load_2':2,'load_5':5}
    epochs.pick_types(eeg=True,eog=False)
    
    labels = np.array(epochs.events[:,-1] == 2,dtype=int)
    data = epochs.get_data()
    times = epochs.times
    info = epochs.info
    del epochs
    
    clf = []
    #clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
    #estimator = LinearModel(estimator)
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    cv = KFold(n_splits=4,shuffle=True,random_state=12345)
    
    time_gen = GeneralizingEstimator(clf,scoring='roc_auc',)
    scores = cross_val_multiscore(time_gen,data,labels,cv=cv,n_jobs=-1,verbose=2)
    scores_mean = np.mean(scores,axis=0)
    scores_std = np.std(scores,axis=1)
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(scores_mean,interpolation='lanczos',origin='lower',
                  cmap='coolwarm',vmin=0.,vmax=1.)
    ax.set(xlabel='Testing Time (sec)',ylabel='Training Time (sec)',
          title='Temporal Generalization: subject %d, load2 vs load5'%sub)
    ax.axvline(0,color='k')
    ax.axhline(0,color='k')
    plt.colorbar(im,ax=ax)
    fig.savefig('D:\\working_memory\\working_memory\\results\\subject_%d_load2load5_generalization_scores.png'%sub,dpi=300)
