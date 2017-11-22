# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:02:51 2017

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
"""Below are a few helper functions I no longer use"""
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
# I use this function
def make_clf():
    """
    Takes no argument, and return a pipeline of linear classifier, containing a scaler and a linear estimator
    """
    clf = []
    clf.append(('vectorizer',Vectorizer())) # vectorize the 3D matrix along the last 2 dimensions
    clf.append(('scaler',StandardScaler()))
    estimator = SVC(kernel='linear',max_iter=int(-1),random_state=12345,class_weight='balanced')
    estimator = LinearModel(estimator) # extra step for decoding patterns
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf    
os.chdir('D:\\working_memory\\')
subs = np.arange(11,33)
subs = np.setdiff1d(subs,np.array([24,27]))


from tqdm import tqdm
from sklearn import metrics
import pickle
n_ = 50 # moving time window
c = int(n_/2) # center the position indeces
for sub in subs:    
    epoch_fif = [f for f in os.listdir() if ('-epo.fif' in f) and ('suj%d_'%(sub) in f)]    
    epochs_1 = mne.read_epochs(epoch_fif[0])
    epochs_2 = mne.read_epochs(epoch_fif[1])
    epochs = mne.concatenate_epochs([epochs_1,epochs_2])
    epochs.event_id = {'load_2':2,'load_5':5}
    epochs.pick_types(eeg=True,eog=False)
    
    labels = np.array(epochs.events[:,-1] == 2,dtype=int)
    data = epochs.get_data()
    info = epochs.info
    times = epochs.times * info['sfreq']
    ch_names = epochs.ch_names
    del epochs
    
    cv = KFold(n_splits=4,shuffle=True,random_state=12345)
    size = len(times[c::n_])
    scores=np.zeros((size,size,4))
    position_index = np.arange(times[0],times[-2],n_) + (n_ * 2 )
    position_index = position_index.astype(int)[:-1]
    clfs = []
    print('cv diagonal\n')
    for ii,idx_train in tqdm(enumerate(position_index),desc='diag loop'):
        scores_ = []
        clfs_ = []
        for train_,test_ in cv.split(data):
            try: # if not the last window, the shape of the training matrix is regular
                clf = make_clf()
                position_range_idx = np.arange(idx_train-c,idx_train+c)
                clf.fit(data[train_][:][...,position_range_idx],labels[train_])
                clfs_.append(clf)
                scores_.append(metrics.roc_auc_score(labels[test_],
                                        clf.predict(data[test_][:][...,position_range_idx])))
            except:# if is the last window, the shape of the training matrix is not regular
                clf = make_clf()
#                position_range_idx = np.arange(idx_train-c,idx_train+c)
                clf.fit(data[train_][:][...,idx_train-c:],labels[train_])
                clfs_.append(clf)
                scores_.append(metrics.roc_auc_score(labels[test_],
                                        clf.predict(data[test_][:][...,idx_train-c:])))
        scores[ii,ii,:] = scores_
        clfs.append(clfs_)
    
    print('cv different time samples\n')
    for ii,idx_train in tqdm(enumerate(position_index),desc='off diag loop'):
        for kk,idx_test in enumerate(position_index):
            if idx_train != idx_test:
                scores_ = []
                for jj,(train_,test_) in enumerate(cv.split(data)):
                    try:
                        clf = clfs[ii][jj]
                        position_range_idx = np.arange(idx_test-c,idx_test+c)
                        scores_.append(metrics.roc_auc_score(labels[test_],
                                                clf.predict(data[test_][:][...,position_range_idx])))
                    except:
                        clf = clfs[ii][jj]
#                        position_range_idx = np.arange(idx_test-c,idx_test+c)
                        scores_.append(metrics.roc_auc_score(labels[test_],
                                                clf.predict(data[test_][:][...,idx_test-c:])))
                    
                scores[ii,kk,:] = scores_
    pickle.dump(scores,open('D:\\working_memory\\subject%d_vectorization.p'%sub,'wb'))
#    scores = pickle.load(open('D:\\working_memory\\subject%d.p'%sub,'rb'))
    """
    Figure 1: time generalization plot. 
    
    A grid of training and testing time. The color indicates the decoding score.
    A high score means the trained model (wherever it was trained) can predict load 2 and load 5 labels using just the signals
    In other words, the signals might similar to the where the model was trained.
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(scores.mean(-1),interpolation=None,origin='lower',
                  cmap='winter',vmin=0.5,vmax=.8,extent=[-100,6000,-100,6000])
    ax.set(xlabel='Testing Time (ms)',ylabel='Training Time (ms)',
          title='Temporal Generalization: subject %d, load2 vs load5'%sub)
    
    ax.axvline(0,color='k')
    ax.axhline(0,color='k')
    cbar=plt.colorbar(im,ax=ax)
    cbar.ax.set_title('ROC AUC scores')
    fig.savefig('D:\\working_memory\\working_memory\\results\\vectorization\\subject_%d_load2load5_generalization_scores.png'%sub,dpi=300)
    """
    Figure 2: time decoding scores
    
    Train and test at the same time.
    Higher score means the linear classifier is able to distinguish load 2 and load 5 using only the signals.
    """
    scores_mean = np.mean(scores.diagonal(),axis=0)
    scores_std = np.std(scores.diagonal(),axis=0)
    plt.close('all')
    fig,ax = plt.subplots(figsize=(20,6))
    time_picks = times[c::n_] 
    scores_picks = scores_mean
    scores_se = (scores_std/2)
    ax.plot(time_picks,scores_picks,label='scores')
    ax.fill_between(time_picks,scores_picks-scores_se,scores_picks+scores_se,color='red',alpha=0.5)
    ax.axhline(0.5,color='k',linestyle='--',label='chance')
    ax.axvline(0,color='k',linestyle='--')
    ax.set(xlabel='Time (ms)',ylabel='ROC AUC',xlim=(-100,5900),ylim=(0.35,1.),title='subject_%d_load2load5_decoding_scores'%sub)
    ax.set(xticks=np.arange(-100,6000,400))
    ax.legend()
    fig.savefig('working_memory\\results/vectorization/subject_%d_load2load5_decoding_scores.png'%sub,dpi=300)


    coef = []
    for clfs_ in clfs:
        coef_ = [get_coef(clf,'patterns_',inverse_transform=True) for clf in clfs_]
        coef.append(coef_)
    coef = np.array(coef)
    coef = np.swapaxes(coef,0,2)
    coef = coef.reshape(61,4,-1)
    evoked = mne.EvokedArray(np.mean(coef,axis=1),info,tmin=times[0])
    evoked.times = times[:-101]
    evoked.save('D:\\working_memory\\vectorization\\subject_%d_load2load5_patterns-evo.fif'%sub,)
    del evoked
import matplotlib
font = {'weight':'bold',
        'size':16}
matplotlib.rc('font',**font)
os.chdir('D:\\working_memory\\')
subs = np.arange(11,33)
subs = np.setdiff1d(subs,np.array([24,27]))
save_dir = 'D:\\working_memory\\working_memory\\results\\'
for sub in subs:
    evoked = mne.read_evokeds('vectorization/subject_%d_load2load5_patterns-evo.fif'%sub,)
    evoked = evoked[0]
    evoked.times = np.arange(-100,6001)[:-51]

    plt.close('all')
    fig,ax = plt.subplots(figsize=(12,15))
    fig=mne.viz.plot_evoked_image(evoked,axes=ax)
    fig.axes[0].set(yticks=np.arange(len(evoked.ch_names)),yticklabels=evoked.ch_names)

    fig.savefig(save_dir+'vectorization\\subject_%d_load2load5_difference_image.png'%sub,dpi=300)
    """
    Figure 3: decoding patterns
    
    Train and test at the same time.
    Higher flatuation means bigger difference in terms of amplitude between load 2 and load 5 at the given time.
    """
    evoked.times = np.arange(-0.1,6.001,0.001)[:-51]
    a,b=-10,10
    fig = mne.viz.plot_evoked_joint(evoked,
                                    topomap_args={'scaling_time':1e3,'size':1,
                                                 'vmin':a,'vmax':b},)
    fig.savefig(save_dir+'vectorization\\subject_%d_load2load5_patterns.png'%sub,dpi=300)