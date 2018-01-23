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
from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults,get_rejection_threshold) 
from functools import partial  # noqa
working_dir = 'D:\\working_memory\\'
data_dir = 'D:\\working_memory\\data_probe_train_test'
result_dir = 'D:\\working_memory\\probe_train_test'
evt_dir = os.path.join(working_dir+'signal detection')
new_evt_save_dir = os.path.join(working_dir + 'EVT')
for directory in [data_dir,result_dir,new_evt_save_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
condition = 'l2_';n = 2
files_vhdr = glob(working_dir+'*%s*.vhdr'%condition)
files_evt = glob(os.path.join(evt_dir,'*%s*'%condition))


n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

missing = np.hstack([np.arange(11,17),[18]])#missing 26  and 64
trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load2_WM',header=None)
trial_orders.columns = ['load','image1','image2','target','probe']
trial_orders['target'] = 1- trial_orders['target']
trial_orders["row"] = np.arange(1,101)

for n in range(len(files_vhdr)):
    sub,_,day = re.findall('\d+',files_vhdr[n])
    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
    events = pd.read_csv(subject_evt,sep='\t')
    events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
    
    if int(sub) in missing:
        #print('in')
        idx = np.logical_and(trial_orders.row!=26,trial_orders.row!=64)
        
    else:
        idx = np.array([True] * 100)
    if int(sub) == 11:
        events_probe = events[events.Recode != 0]
        idx[0] = False
    else:
        events_probe = events[events['Comnt']=='Delay offset']
    
    working_trial_orders = trial_orders[idx]
    print(sub,day,events_probe.shape,working_trial_orders.shape)
    events_probe['Recode']=working_trial_orders['target'].values
    if events_probe['Comnt'].isnull().values.any():
#        print(sub,day)
        events_probe = events_probe.dropna()
    events_delay = events[events['Comnt']=='Delay onset']
    events_delay['Recode']=working_trial_orders['target'].values
    if events_delay['Comnt'].isnull().values.any():
        events_delay = events_delay.dropna()
    events_encode = events[events['Comnt']=='ENC onset']
    encode_images = working_trial_orders[['image1','image2']].values.flatten()
    probe_images = np.vstack([working_trial_orders['probe'],working_trial_orders['probe']]).T.flatten()
    encode_recode = np.array(encode_images == probe_images).astype(int)
    events_encode['Recode'] = encode_recode
    if events_encode['Comnt'].isnull().values.any():
        events_encode = events_encode.dropna()
    
    events_delay.to_csv(os.path.join(new_evt_save_dir,'sub%s_load2_day%s_delay.csv'%(sub,day)),index=False)
    events_encode.to_csv(os.path.join(new_evt_save_dir,'sub%s_load2_day%s_encode.csv'%(sub,day)),index=False)
    events_probe.to_csv(os.path.join(new_evt_save_dir,'sub%s_load2_day%s_probe.csv'%(sub,day)),index=False)
    
    
        

"""probe and delay condition"""
data_dir_probe = os.path.join(data_dir,'probe')
if not os.path.exists(data_dir_probe):
    os.makedirs(data_dir_probe)
data_dir_delay = os.path.join(data_dir,'delay')
if not os.path.exists(data_dir_delay):
    os.makedirs(data_dir_delay) 
data_dir_encode = os.path.join(data_dir,'encode')
if not os.path.exists(data_dir_encode):
    os.makedirs(data_dir_encode)
evt_file_probe=glob(os.path.join(new_evt_save_dir,'*probe.csv'))
evt_file_delay=glob(os.path.join(new_evt_save_dir,'*delay.csv'))
evt_file_encode=glob(os.path.join(new_evt_save_dir,'*encode.csv'))
for vhdr,probe,delay,encode in zip(files_vhdr,evt_file_probe,evt_file_delay,evt_file_encode):
    sub,_,day = re.findall('\d+',vhdr)
    
    events_probe = pd.read_csv(probe)
    events_delay = pd.read_csv(delay)
    events_encode = pd.read_csv(encode)
    raw = mne.io.read_raw_brainvision(vhdr,montage='standard_1020',
                                         eog=('LOc','ROc','Aux1'),preload=True)
    raw.set_channel_types({'Aux1':'stim'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
    event_id = {'non target probe':0,'target probe':1}
#    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    events_ = events_probe[['tms','RT','Recode']].values.astype(int)
    events_[:,1] = 0
    epochs = mne.Epochs(raw,events_,event_id,tmin=-0.05,tmax=2,baseline=(-0.05,0),picks=picks,preload=True,reject=None,)
#    reject = get_rejection_threshold(epochs)
#    reject_ = compute_thresholds(epochs,random_state=12345,picks=picks,)
    #### fix epochs
    thresh_func = partial(compute_thresholds, picks=picks, method='bayesian_optimization',
                      random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks,
                       thresh_func=thresh_func)
    ar.fit(epochs)
    epochs = ar.transform(epochs)
#    ##### fit ICA ##############################
#    noise_cov = mne.compute_covariance(epochs,tmin=-0.05,tmax=2,)#n_jobs=2)
#    n_components = .99  
#    method = 'extended-infomax'  
#    decim = 3  
#    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
#                               method=method,random_state=12345,max_iter=3000,)
#    picks = mne.pick_types(epochs.info,meg=False,eeg=True,eog=False)
#    ica.fit(epochs,picks=picks,decim=decim,reject=reject_)
#    ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'],eog_criterion=0.2,)
#    epochs = ica.apply(epochs)
#    
#    epochs.drop_bad(reject=reject)
    epochs.pick_types(meg=False,eeg=True,eog=False)
#    epochs.resample(128) # so that I could decode patterns
    print(epochs)
    epochs.save(os.path.join(data_dir_probe,'sub%s_load2_day%s-epo.fif'%(sub,day)))
    
    ### delay ###    
    events_ = events_delay[['tms','RT','Recode']].values.astype(int)
    events_[:,1] = 0
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    epochs = mne.Epochs(raw,events_,event_id,tmin=-0.05,tmax=6,baseline=(-0.05,0),picks=picks,preload=True,reject=None,)
#    reject = get_rejection_threshold(epochs)
#    reject_ = compute_thresholds(epochs,random_state=12345,picks=picks,)
    #### fix epochs
    thresh_func = partial(compute_thresholds, picks=picks, method='random_search',
                      random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks,
                       thresh_func=thresh_func)
    ar.fit(epochs)
    epochs = ar.transform(epochs)
#    ##### fit ICA ##############################
#    noise_cov = mne.compute_covariance(epochs,tmin=-0.05,tmax=6,)#n_jobs=2)
#    n_components = .99  
#    method = 'extended-infomax'  
#    decim = 3  
#    ica = mne.preprocessing.ICA(n_components=n_components,noise_cov=noise_cov,
#                               method=method,random_state=12345,max_iter=3000,)
#    picks = mne.pick_types(epochs.info,meg=False,eeg=True,eog=False)
#    ica.fit(epochs,picks=picks,decim=decim,reject=reject_)
#    ica.detect_artifacts(epochs,eog_ch=['LOc','ROc'],eog_criterion=0.2,)
#    epochs = ica.apply(epochs)
#    epochs.drop_bad(reject=reject)
    epochs.pick_types(meg=False,eeg=True,eog=False)
#    epochs.resample(128) # so that I could decode patterns
    print(epochs)
    epochs.save(os.path.join(data_dir_delay,'sub%s_load2_day%s-epo.fif'%(sub,day)))
    
    #### encode #####
    events_ = events_encode[['tms','RT','Recode']].values.astype(int)
    events_[:,1] = 0
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    epochs = mne.Epochs(raw,events_,event_id,tmin=-0.05,tmax=2,baseline=(-0.05,0),picks=picks,preload=True,reject=None,)
    thresh_func = partial(compute_thresholds, picks=picks, method='random_search',
                      random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks,
                       thresh_func=thresh_func)
    ar.fit(epochs)
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
    print(epochs)
    epochs.save(os.path.join(data_dir_encode,'sub%s_load2_day%s-epo.fif'%(sub,day)))
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
        events_probe = events[events['Comnt'] == 'Probe onset']
    events_delay = events[events['TriNo'] == 71]
    label_dict = {4:0,3:0,2:1,1:1}
    events_probe['labels'] = events_probe.Recode.map(label_dict)
    temp_ = []
    for onset in events_delay['tms'].values:
        idx = [i for i,v in enumerate(events_probe['tms'].values) if (0 < v - onset < 6100) ]
        if len(idx) > 0:
            idx = idx[0]
            print(onset,events_probe['tms'].values[idx],events_probe['labels'].values[idx])
            temp_.append([onset,0,events_probe['labels'].values[idx]])
    events_ = np.array(temp_).astype(int)
    #drop nan
    idx = np.logical_or(events_[:,-1]==0, events_[:,-1]==1)
    events_ = events_[idx,:]
    event_id = {'non target probe':0,'target probe':1}
    
    print(files_vhdr[n],subject_evt)
    raw = mne.io.read_raw_brainvision(files_vhdr[n],montage='standard_1020',
                                         eog=('LOc','ROc','Aux1'),preload=True)
    raw.set_channel_types({'Aux1':'stim'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
    
    epochs = mne.Epochs(raw,events_,event_id,tmin=-0.05,tmax=6,baseline=(-0.05,0),picks=picks,preload=True,reject=None,)
    thresh_func = partial(compute_thresholds, picks=picks, method='random_search',
                      random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks,
                       thresh_func=thresh_func)
    ar.fit(epochs)
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
    print(epochs)
    epochs.save(os.path.join(data_dir_delay,'sub%s_load2_day%s-epo.fif'%(sub,day)))




for vhdr,probe,delay,encode in zip(files_vhdr,evt_file_probe,evt_file_delay,evt_file_encode):
    sub,_,day = re.findall('\d+',vhdr)
    
    events_probe = pd.read_csv(probe)
    events_delay = pd.read_csv(delay)
    events_encode = pd.read_csv(encode)
    raw = mne.io.read_raw_brainvision(vhdr,montage='standard_1020',
                                         eog=('LOc','ROc','Aux1'),preload=True)
    raw.set_channel_types({'Aux1':'stim'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')#n_jobs=2)
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')#n_jobs=2)
    event_id = {'non target probe':0,'target probe':1}
    #### encode #####
    events_ = events_encode[['tms','RT','Recode']].values.astype(int)
    events_[:,1] = 0
    picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=True)
    epochs = mne.Epochs(raw,events_,event_id,tmin=-0.05,tmax=2,baseline=(-0.05,0),picks=picks,preload=True,reject=None,)
    thresh_func = partial(compute_thresholds, picks=picks, method='random_search',
                      random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks,
                       thresh_func=thresh_func)
    ar.fit(epochs)
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
    print(epochs)
    epochs.save(os.path.join(data_dir_encode,'sub%s_load2_day%s-epo.fif'%(sub,day)))






































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
































