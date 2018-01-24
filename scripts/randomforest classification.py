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
import pickle


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mne.decoding import Vectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
working_dir = 'D:\\working_memory\\data_probe_train_test\\'
saving_dir = 'D:\\working_memory\\working_memory\\results\\train_probe_test_encode'

"""directories"""
files_probe = glob(os.path.join(working_dir,'probe/*.fif'))
files_delay = glob(os.path.join(working_dir,'delay/*.fif'))
files_encode = glob(os.path.join(working_dir,'encode/*.fif'))

def make_clf(estimator,vec=True):
    clf = []
    if vec:
        clf.append(('vectorizer',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    clf.append(('estimator',estimator))
    clf = Pipeline(clf)
    return clf

"""train on probe, test on encode"""
cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=12345)
interval = 25 # 25 ms window
clfs = {}
scores = {}
scores_test = {}
for probe,encode in zip(files_probe,files_encode):
    epochs_probe = mne.read_epochs(probe)
#    epochs_delay = mne.read_epochs(delay)
    epochs_encode = mne.read_epochs(encode)
    sub,load,day = re.findall('\d+',probe)
    title = 'sub%s, day%s,load2'%(sub,day)
    clfs['sub%s, day%s,load2'%(sub,day)] = []
    scores['sub%s, day%s,load2'%(sub,day)] = []
    estimator = RandomForestClassifier(n_estimators=50,random_state=12345,class_weight='balanced')
    train_data = epochs_probe.get_data()[:,:,50:]
    train_labels = epochs_probe.events[:,-1]
    test_data = epochs_encode.get_data()[:,:,50:]
    test_labels = epochs_encode.events[:,-1]
    chunk_idx = np.arange(0,train_data.shape[-1],interval)
    chunks = np.vstack((chunk_idx[:-1],chunk_idx[1:])).T
    for chunk in tqdm(chunks,desc='chunks'):
        start,stop = chunk
        temp_train = train_data[:,:,start:stop] 
#        print(temp_train.shape)
        temp_clfs = []
        temp_scores = []
        for train,test in cv.split(temp_train,train_labels):
            temp_train_split = temp_train[train]
            temp_train_labels = train_labels[train]
            clf = make_clf(estimator)
            clf.fit(temp_train_split,temp_train_labels)
            temp_clfs.append(clf)
            
            temp_pred = clf.predict(temp_train[test])
            temp_pred_proba = clf.predict_proba(temp_train[test])[:,-1]
            auc = metrics.roc_auc_score(train_labels[test],temp_pred_proba)
            confusion_matrix = metrics.confusion_matrix(train_labels[test],temp_pred)
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(1)[:,np.newaxis]
            mcc = metrics.matthews_corrcoef(train_labels[test],temp_pred)
            temp_scores.append(np.concatenate(([auc],confusion_matrix.flatten(),[mcc])))
        clfs['sub%s, day%s,load2'%(sub,day)].append(temp_clfs)
        scores['sub%s, day%s,load2'%(sub,day)].append(temp_scores)
        
    train_chunk_idx = chunks
    scores_test['sub%s, day%s,load2'%(sub,day)] = []
    chunk_idx = np.arange(0,test_data.shape[-1],interval)
    chunks = np.vstack((chunk_idx[:-1],chunk_idx[1:])).T
    for chunk in tqdm(chunks,desc='test'):
        start,stop = chunk
        temp_test = test_data[:,:,start:stop]
        scores_test_at_one_time_interval = []
        for time_invervals in clfs['sub%s, day%s,load2'%(sub,day)]:
            temp_scores = []
            for cv_blocks in time_invervals:
                temp_pred = cv_blocks.predict(temp_test)
                temp_pred_proba = cv_blocks.predict_proba(temp_test)[:,-1]
                auc = metrics.roc_auc_score(test_labels,temp_pred_proba)
                confusion_matrix = metrics.confusion_matrix(test_labels,temp_pred)
                confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(1)[:,np.newaxis]
                mcc = metrics.matthews_corrcoef(test_labels,temp_pred)
                temp_scores.append(np.concatenate(([auc],confusion_matrix.flatten(),[mcc])))
                
            scores_test_at_one_time_interval.append(temp_scores)
        scores_test['sub%s, day%s,load2'%(sub,day)].append(scores_test_at_one_time_interval)
pickle.dump(scores_test,open(saving_dir+ '\\probe-encode.p','wb'))   
pickle.dump([scores,clfs],open(saving_dir+'\\probe-probe.p','wb'))     
scores_test = pickle.load(open(saving_dir+ '\\probe-encode.p','rb'))

plt.close('all')
for name,value in scores_test.items():
    ccc = np.array(value)
    
    sss = ccc[:,:,:,0]
    sss_mean = sss.mean(2)
    sss_std = sss.std(2)
    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(sss_mean,origin='lower',aspect='auto',extent=[0,2000,0,2000],interpolation='gaussian')
    ax.set(xlabel='test models at delay',ylabel='train models at probe',title=name)
    plt.colorbar(im)
    fig.savefig(os.path.join(saving_dir,'AUC scores_%s.png'%(name)),dpi=300)

plt.close('all')
subplot_titles = ['sensitivity of negative probe',
                  'false alarm',
                  'miss',
                  'sensitivity of positive probe']
for name,value in scores_test.items():
    ccc = np.array(value)
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    for ii,ax in enumerate(axes.flatten()):
        sss = ccc[:,:,:,ii+1]
        sss_mean = sss.mean(2)
        sss_std = sss.std(2)
        im = ax.imshow(sss_mean,origin='lower',aspect='auto',extent=[0,2000,0,2000],vmin=0.5,vmax=0.7)
        ax.set(title=subplot_titles[ii])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    fig.colorbar(im,cax=cbar_ax)    
    fig.suptitle(name)
    fig.savefig(os.path.join(saving_dir,'confusion matrix of matices_%s.png'%(name)),dpi=300)
#        plt.colorbar(im)


"""train on probe, test on delay"""
cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=12345)
interval = 25 # 25 ms window
clfs = {}
scores = {}
scores_test = {}
for probe,delay in zip(files_probe,files_delay):
    epochs_probe = mne.read_epochs(probe)
    epochs_delay = mne.read_epochs(delay)
#    epochs_encode = mne.read_epochs(encode)
    sub,load,day = re.findall('\d+',probe)
    title = 'sub%s, day%s,load2'%(sub,day)
    clfs['sub%s, day%s,load2'%(sub,day)] = []
    scores['sub%s, day%s,load2'%(sub,day)] = []
    estimator = RandomForestClassifier(n_estimators=50,random_state=12345,class_weight='balanced')
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
        for train,test in cv.split(temp_train,train_labels):
            temp_train_split = temp_train[train]
            temp_train_labels = train_labels[train]
            clf = make_clf(estimator)
            clf.fit(temp_train_split,temp_train_labels)
            temp_clfs.append(clf)
            
            temp_pred = clf.predict(temp_train[test])
            temp_pred_proba = clf.predict_proba(temp_train[test])[:,-1]
            auc = metrics.roc_auc_score(train_labels[test],temp_pred_proba)
            confusion_matrix = metrics.confusion_matrix(train_labels[test],temp_pred)
            confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(1)[:,np.newaxis]
            mcc = metrics.matthews_corrcoef(train_labels[test],temp_pred)
            temp_scores.append(np.concatenate(([auc],confusion_matrix.flatten(),[mcc])))
        clfs['sub%s, day%s,load2'%(sub,day)].append(temp_clfs)
        scores['sub%s, day%s,load2'%(sub,day)].append(temp_scores)
        
    train_chunk_idx = chunks
    scores_test['sub%s, day%s,load2'%(sub,day)] = []
    chunk_idx = np.arange(0,test_data.shape[-1],interval)
    chunks = np.vstack((chunk_idx[:-1],chunk_idx[1:])).T
    for chunk in tqdm(chunks,desc='test'):
        start,stop = chunk
        temp_test = test_data[:,:,start:stop]
        scores_test_at_one_time_interval = []
        for time_invervals in clfs['sub%s, day%s,load2'%(sub,day)]:
            temp_scores = []
            for cv_blocks in time_invervals:
                temp_pred = cv_blocks.predict(temp_test)
                temp_pred_proba = cv_blocks.predict_proba(temp_test)[:,-1]
                auc = metrics.roc_auc_score(test_labels,temp_pred_proba)
                confusion_matrix = metrics.confusion_matrix(test_labels,temp_pred)
                confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(1)[:,np.newaxis]
                mcc = metrics.matthews_corrcoef(test_labels,temp_pred)
                temp_scores.append(np.concatenate(([auc],confusion_matrix.flatten(),[mcc])))
                
            scores_test_at_one_time_interval.append(temp_scores)
        scores_test['sub%s, day%s,load2'%(sub,day)].append(scores_test_at_one_time_interval)


saving_dir = 'D:\\working_memory\\working_memory\\results\\train_probe_test_delay'
if os.path.exist(saving_dir):
    os.makedirs(saving_dir)
pickle.dump(scores_test,open(saving_dir+ '\\probe-delay.p','wb'))  
pickle.dump([scores,clfs],open(saving_dir+'\\probe-probe.p','wb'))           


plt.close('all')
for name,value in scores_test.items():
    ccc = np.array(value)
    
    sss = ccc[:,:,:,0]
    sss_mean = sss.mean(2)
    sss_std = sss.std(2)
    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(sss_mean,origin='lower',aspect='auto',extent=[0,2000,0,2000])
    ax.set(xlabel='test models at delay',ylabel='train models at probe',title=name)
    plt.colorbar(im)
    fig.savefig(os.path.join(saving_dir,'AUC scores_%s.png'%(name)),dpi=300)

plt.close('all')
subplot_titles = ['sensitivity of negative probe',
                  'false alarm',
                  'miss',
                  'sensitivity of positive probe']
for name,value in scores_test.items():
    ccc = np.array(value)
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,10))
    for ii,ax in enumerate(axes.flatten()):
        sss = ccc[:,:,:,ii+1]
        sss_mean = sss.mean(2)
        sss_std = sss.std(2)
        im = ax.imshow(sss_mean,origin='lower',aspect='auto',extent=[0,2000,0,2000],vmin=0.5,vmax=0.7)
        ax.set(title=subplot_titles[ii])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    fig.colorbar(im,cax=cbar_ax)    
    fig.suptitle(name)
    fig.savefig(os.path.join(saving_dir,'confusion matrix of matices_%s.png'%(name)),dpi=300)
#        plt.colorbar(im)





