# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:01:04 2018

@author: ning
"""

import mne
import os
from glob import glob
import pandas as pd
import numpy as np
from autoreject import (LocalAutoRejectCV,compute_thresholds)
from functools import partial
import re
from matplotlib import pyplot as plt

working_dir = 'D:\\working_memory'
os.chdir(working_dir)
saving_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
# pick all the load2 files, eeg and event files
eegs = glob(os.path.join(working_dir,'*l2*.vhdr'))
evts = glob(os.path.join(working_dir+'\\EVT','*_probe.csv'))
# hyperparameters for auto artifact correction
n_interpolates = np.array([1,4,32])
consensus_percs = np.linspace(0,1.0,11)
for raw_,evt in zip(eegs,evts):
    print(raw_,evt)
    events = pd.read_csv(evt)
    events = events[['tms','code','Recode']].values.astype(int)
    events[:,1] = 0
#    events = events[::2,:]# get the odd rows
    event_id = {'negative probe':0,'positive probe':1}
    raw = mne.io.read_raw_brainvision(raw_,preload=True,montage='standard_1020',eog=['LOc','ROc','Aux1'])# load raw eeg data
    raw.set_channel_types({'STI 014':'stim','Aux1':'stim','LOc':'eog','ROc':'eog'})# define channel types
    raw.set_eeg_reference().apply_proj()# average re-referencing
    picks = mne.pick_types(raw.info,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')# band pass filter
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')# notch filter
    # epoch the data without rejecting any segment
    epochs = mne.Epochs(raw,events=events,event_id=event_id,tmin=-10,tmax=2,baseline=(-0.2,0),preload=True,picks=picks,detrend=1,proj=False)
    #epochs['0'].average().plot()
    sub,load,day = re.findall('\d+',raw_)
    # define threshold fitting function
    thresh_func = partial(compute_thresholds,picks=picks,method='bayesian_optimization',random_state=12345)
    # define auto-correction method, based on local values, both eeg and eog channels are included
    ar = LocalAutoRejectCV(n_interpolates,consensus_percs,picks=picks,thresh_func=thresh_func)
    ar.fit(epochs)# why did I do fit_transform?
    print('transform the data')
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)# take away the eog channels
    epochs.save(os.path.join(saving_dir,'sub_%s_load%s_day%s_encode_delay-epo.fif'%(sub,load,day)))
    fig, axes = plt.subplots(figsize=(16,8),nrows=2)
    for (key, value),ax in zip(event_id.items(),axes):
        epochs[key].average().plot(titles=key,axes=ax)
    fig.savefig(os.path.join(saving_dir,'sub_%s_load%s_day%s_encode_delay-epo.png'%(sub,load,day)))
    plt.close('all')


#################### load 5 ################################################################

"""
working_dir = 'D:\\working_memory\\'
data_dir = 'D:\\working_memory\\data_probe_train_test'
result_dir = 'D:\\working_memory\\probe_train_test'
evt_dir = os.path.join(working_dir+'signal detection')
new_evt_save_dir = os.path.join(working_dir + 'EVT_load5')
if not os.path.exists(new_evt_save_dir):
    os.mkdir(new_evt_save_dir)

condition = 'l5_';#n = 2
files_vhdr = glob(working_dir+'*%s*.vhdr'%condition)
files_evt = glob(os.path.join(evt_dir,'*%s*'%condition))

# get the order of the stimulu
trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load5_WM',header=None)
trial_orders.columns = ['load','image1','image2','image3','image4','image5','target','probe']
trial_orders['target'] = 1- trial_orders['target']
trial_orders["row"] = np.arange(1,41)
# the for-loop is for generating event files only
for n in range(len(files_vhdr)):
    sub,_,day = re.findall('\d+',files_vhdr[n])
    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
    events = pd.read_csv(subject_evt,sep='\t')
    try:# annoying \t and spaces
        events.columns = ['tms','code','TriNo','RT','Recode','Comnt1','Comnt2']
        events['Comnt'] = events['Comnt1'].map(str) +' ' +events['Comnt2'].map(str)    
    except:
        events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
#    print(sub,day,events.shape)
    if int(sub) == 11:# take the first trial out
        #print('in')
        idx = trial_orders.row!=1
        events = events.iloc[5:,:]
    else:
        idx = np.array([True] * 40)
    working_trial_orders = trial_orders[idx]
#    print(np.unique(events['TriNo']))
    
    if len(np.unique(events['TriNo'])) > 4:
        mapping = {int(k):int(8) for k in np.unique(events['TriNo'])[3:]}
#        print(mapping)
        events['TriNo'] = events['TriNo'].values.astype(int)
        events['TriNo'] = events['TriNo'].map(mapping)

    # get the rows of delay
    events_probe = events[events['TriNo']==8]
    if int(sub) == 11:
        events_probe = events_probe.iloc[1:,:]
    # get the target value, not informative if not using machine learning
    events_probe['Recode']=working_trial_orders['target'].values
    if events_probe['Comnt'].isnull().values.any():
        events_probe = events_probe.dropna()
    print(events_probe.shape)
    events_probe.to_csv(os.path.join(new_evt_save_dir,'sub%s_load5_day%s_probe.csv'%(sub,day)),index=False)
#    events_delay.to_csv(os.path.join(new_evt_save_dir,'sub%s_load5_day%s_delay.csv'%(sub,day)),index=False)
#    events_encode.to_csv(os.path.join(new_evt_save_dir,'sub%s_load5_day%s_encode.csv'%(sub,day)),index=False)
    
"""  


working_dir = 'D:\\working_memory'
os.chdir(working_dir)
saving_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

eegs = glob(os.path.join(working_dir,'*l5*.vhdr'))
evts = glob(os.path.join('D:\\working_memory\\EVT_load5','*_probe.csv'))
n_interpolates = np.array([1,4,32])
consensus_percs = np.linspace(0,1.0,11)
for raw_,evt in zip(eegs,evts):
    print(raw_,evt)
    events = pd.read_csv(evt)
    events = events[['tms','code','Recode']].values.astype(int)
    events[:,1] = 0
#    events = events[::5,:]
    print(np.mean(events[:,-1]))
    event_id = {'negative probe':0,'positive probe':1}
    raw = mne.io.read_raw_brainvision(raw_,preload=True,montage='standard_1020',eog=['LOc','ROc','Aux1'])
    raw.set_channel_types({'STI 014':'stim','Aux1':'stim','LOc':'eog','ROc':'eog'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')
    epochs = mne.Epochs(raw,events=events,event_id=event_id,tmin=-16,tmax=2,baseline=(-0.2,0),preload=True,picks=picks,detrend=1,proj=False)
    #epochs['0'].average().plot()
    sub,load,day = re.findall('\d+',raw_)
    thresh_func = partial(compute_thresholds,picks=picks,method='bayesian_optimization',random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates,consensus_percs,picks=picks,thresh_func=thresh_func)
    ar.fit(epochs)
    print('transform the data')
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
    epochs.save(os.path.join(saving_dir,'sub_%s_load%s_day%s_encode_delay-epo.fif'%(sub,load,day)))
    fig, axes = plt.subplots(figsize=(16,8),nrows=2)
    for key, value in event_id.items():
        epochs[key].average().plot(title=key,ax=axes[value])
    fig.savefig(os.path.join(saving_dir,'sub_%s_load%s_day%s_encode_delay-epo.png'%(sub,load,day)))
    plt.close('all')




















