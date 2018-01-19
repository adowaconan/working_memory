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
from autoreject import get_rejection_threshold

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
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
    event_id = {'non target probe':0,'target probe':1}
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=2,baseline=(None,0),picks=picks,preload=True,reject=None,)
    reject = get_rejection_threshold(epochs,)
    epochs.drop_bad(reject=reject)
    epochs.pick_types(meg=False,eeg=True,eog=False)
#    epochs.resample(128) # so that I could decode patterns
    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))
    
    ### delay ###    
    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
    events = pd.read_csv(subject_evt,sep='\t')
    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
    events_delay = events[events['TriNo'] == 71]
    events_delay['Recode'] = 1
    events_ = events_delay[['tms','RT','Recode']].values.astype(int)

    print(files_vhdr[n],subject_evt)
    event_id={'delay':1}
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=6,baseline=(None,0),picks=picks,preload=True,reject=None,)
    reject = get_rejection_threshold(epochs,)
    epochs.drop_bad(reject=reject)
    epochs.pick_types(meg=False,eeg=True,eog=False)
#    epochs.resample(128) # so that I could decode patterns
    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))

#"""delay condition"""
#data_dir_probe = os.path.join(data_dir,'delay')
#if not os.path.exists(data_dir_probe):
#    os.makedirs(data_dir_probe)
##
#for n in range(len(files_vhdr)):
#    sub,_,day = re.findall('\d+',files_vhdr[n])
#    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
#    events = pd.read_csv(subject_evt,sep='\t')
#    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
#    events_delay = events[events['TriNo'] == 71]
#    events_delay['Recode'] = 1
#    
#    events_ = events_delay[['tms','RT','Recode']].values.astype(int)
#    
#    print(files_vhdr[n],subject_evt)
#    raw = mne.io.read_raw_brainvision(files_vhdr[n],montage='standard_1020',
#                                         eog=('LOc','ROc','Aux1'),preload=True)
#    raw.set_channel_types({'Aux1':'stim'})
#    raw.set_eeg_reference().apply_proj()
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
#    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
#    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
#    event_id={'delay':1}
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
#    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=6,baseline=(None,0),picks=picks,preload=True,reject=None,)
#    reject = get_rejection_threshold(epochs,)
#    epochs.drop_bad(reject=reject)
#    epochs.pick_types(meg=False,eeg=True,eog=False)
##    epochs.resample(128) # so that I could decode patterns
#    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))





















































#if not os.path.exists(data_dir+'\\cleaned_data'):
#    os.makedirs(data_dir+'\\cleaned_data')
#
#for n in range(len(files_vhdr)):
#    sub,_,day = re.findall('\d+',files_vhdr[n])
#    raw = mne.io.read_raw_brainvision(files_vhdr[n],montage='standard_1020',
#                                         eog=('LOc','ROc','Aux1'),preload=True)
#    raw.set_channel_types({'Aux1':'stim'})
#    raw.set_eeg_reference().apply_proj()
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False)
#    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
#    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
#    ##### fit ICA ##############################
#    reject={'eog':400e-6,'eeg':150e-6}
#    noise_cov = mne.compute_raw_covariance(raw,tmin=0,tmax=None,reject=reject)
#    n_components = .99  
#    method = 'extended-infomax'  
#    decim = 3  
#    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
#                               method=method,random_state=12345,max_iter=3000,)
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False)
#    reject = {'eog':260e-6,'eeg':80e-6}
#    ica.fit(raw,picks=picks,decim=decim,reject=reject)
#    ica.detect_artifacts(raw,eog_ch=['LOc','ROc'],eog_criterion=0.05,)



#"""probe condition"""
#data_dir_probe = os.path.join(data_dir,'probe')
#if not os.path.exists(data_dir_probe):
#    os.makedirs(data_dir_probe)
#    
#for n in range(len(files_vhdr)):
#    sub,_,day = re.findall('\d+',files_vhdr[n])
#    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
#    events = pd.read_csv(subject_evt,sep='\t')
#    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
#    if events.Recode.sum() != 0:
#        recode = events.Recode
#    if 27 < int(sub) <30:
#        events['Recode'] = events['TriNo'].apply(lambda x:int(str(x)[-1]) if x > 80 else 0)
#        events_probe = events[events.Recode != 0]
#        recode = events.Recode
#    elif int(sub) > 29:
#        events['Recode'] = recode
#    else:
#        events_probe = events[events.Recode != 0]
#    label_dict = {4:0,3:0,2:1,1:1}
#    events_probe['labels'] = events_probe.Recode.map(label_dict)
#    events_probe = events_probe.dropna()
#    events_ = events_probe[['tms','code','labels']].values.astype(int)
#    events_[:,1] = 0
#
#
#    print(files_vhdr[n],subject_evt)
#    raw = mne.io.read_raw_brainvision(files_vhdr[n],montage='standard_1020',
#                                         eog=('LOc','ROc','Aux1'),preload=True)
#    raw.set_channel_types({'Aux1':'stim'})
#    raw.set_eeg_reference().apply_proj()
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False)
#    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
#    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
#    reject={'eog':400e-6,'eeg':150e-6}
#    event_id = {'non target probe':0,'target probe':1}
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
#    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=2,baseline=(None,0),picks=picks,preload=True,reject=reject,)
#    ##### fit ICA ##############################
#    noise_cov = mne.compute_covariance(epochs,tmin=-0.1,tmax=2,)#n_jobs=2)
#    n_components = .99  
#    method = 'extended-infomax'  
#    decim = 3  
#    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
#                               method=method,random_state=12345,max_iter=3000,)
#    picks = mne.pick_types(epochs.info,meg=False,eeg=True,eog=False)
#    reject = {'eog':300e-6,'eeg':100e-6}
#    ica.fit(epochs,picks=picks,decim=decim,reject=reject)
#    ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'],eog_criterion=0.2,)
#    epochs = ica.apply(epochs)
#    epochs.pick_types(meg=False,eeg=True,eog=False)
##    epochs.resample(128) # so that I could decode patterns
#    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))
###### plot average over trials############
##epochs['non target probe'].average().plot_joint(title='non target probe')
##epochs['target probe'].average().plot_joint(title='target probe')
#
#"""delay condition"""
#data_dir_probe = os.path.join(data_dir,'delay')
#if not os.path.exists(data_dir_probe):
#    os.makedirs(data_dir_probe)
##
#for n in range(len(files_vhdr)):
#    sub,_,day = re.findall('\d+',files_vhdr[n])
#    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
#    events = pd.read_csv(subject_evt,sep='\t')
#    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
#    events_delay = events[events['TriNo'] == 71]
#    events_delay['Recode'] = 1
#    
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
#    reject={'eog':400e-6,'eeg':150e-6}
#    event_id={'delay':1}
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
#    epochs = mne.Epochs(raw,events_,event_id,tmin=0,tmax=6,baseline=(None,0),picks=picks,preload=True,reject=reject,)
#    ##### fit ICA ##############################
#    noise_cov = mne.compute_covariance(epochs,tmin=0,tmax=6,)#n_jobs=2)
#    n_components = .99  
#    method = 'extended-infomax'  
#    decim = 3  
#    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
#                               method=method,random_state=12345,max_iter=3000,)
#    picks = mne.pick_types(epochs.info,meg=False,eeg=True,eog=False)
#    reject = {'eog':300e-6,'eeg':100e-6}
#    ica.fit(epochs,picks=picks,decim=decim,reject=reject)
#    ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'],eog_criterion=0.2,)
#    epochs = ica.apply(epochs)
#    epochs.pick_types(meg=False,eeg=True,eog=False)
##    epochs.resample(128) # so that I could decode patterns
#    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))
































