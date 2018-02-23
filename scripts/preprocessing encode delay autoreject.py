# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:58:26 2018

@author: ning
"""

import mne
import os
from glob import glob
import pandas as pd
import numpy as np
from autoreject import (LocalAutoRejectCV,compute_thresholds,
                        get_rejection_threshold)
from functools import partial
import re

working_dir = 'D:\\working_memory'
os.chdir(working_dir)
saving_dir = 'D:/working_memory/encode_delay_similarity/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

eegs = glob(os.path.join(working_dir,'*l2*.vhdr'))
evts = glob(os.path.join(working_dir+'\\EVT','*_encode.csv'))
n_interpolates = np.array([1,4,32])
consensus_percs = np.linspace(0,1.0,11)
for raw_,evt in zip(eegs,evts):
    print(raw_,evt)
    events = pd.read_csv(evt)
    events = events[['tms','code','Recode']].values.astype(int)
    events[:,1] = 0
    events = events[::2,:]
    event_id = {'0':0,'1':1}
    raw = mne.io.read_raw_brainvision(raw_,preload=True,montage='standard_1020',eog=['LOc','ROc','Aux1'])
    raw.set_channel_types({'STI 014':'stim','Aux1':'stim','LOc':'eog','ROc':'eog'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')
    epochs = mne.Epochs(raw,events=events,event_id=event_id,tmin=0,tmax=10,baseline=(9.8,10),preload=True,picks=picks,detrend=1,proj=False)
    #epochs['0'].average().plot()
    sub,load,day = re.findall('\d+',raw_)
    thresh_func = partial(compute_thresholds,picks=picks,method='bayesian_optimization',random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates,consensus_percs,picks=picks,thresh_func=thresh_func)
    ar.fit(epochs)
    print('transform the data')
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
    epochs.save(os.path.join(saving_dir,'sub_%s_load%s_day%s_encode_delay-epo.fif'%(sub,load,day)))



#################### load 5 ################################################################
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


trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load5_WM',header=None)
trial_orders.columns = ['load','image1','image2','image3','image4','image5','target','probe']
trial_orders['target'] = 1- trial_orders['target']
trial_orders["row"] = np.arange(1,41)
for n in range(len(files_vhdr)):
    sub,_,day = re.findall('\d+',files_vhdr[n])
    subject_evt = [f for f in files_evt if ('suj%s_'%sub in f) and ('day%s'%day in f)][0]
    events = pd.read_csv(subject_evt,sep='\t')
    try:
        events.columns = ['tms','code','TriNo','RT','Recode','Comnt1','Comnt2']
        events['Comnt'] = events['Comnt1'].map(str) +' ' +events['Comnt2'].map(str)    
    except:
        events.columns = ['tms','code','TriNo','RT','Recode','Comnt']
#    print(sub,day,events.shape)
    if int(sub) == 11:
        #print('in')
        idx = trial_orders.row!=1
        events = events.iloc[5:,:]
    else:
        idx = np.array([True] * 40)
    working_trial_orders = trial_orders[idx]
    events_delay = events[events['Comnt']=='Delay onset']
    events_delay['Recode']=working_trial_orders['target'].values
    if events_delay['Comnt'].isnull().values.any():
        events_delay = events_delay.dropna()
    events_encode = events[events['Comnt']=='ENC onset']
    encode_images = working_trial_orders[['image1','image2','image3','image4','image5']].values.flatten()
    
    events_encode['Recode'] = np.repeat(working_trial_orders['target'].values,5)
    if events_encode['Comnt'].isnull().values.any():
        events_encode = events_encode.dropna()
    print(sub,day,events_delay.shape,events_encode.shape)
    events_delay.to_csv(os.path.join(new_evt_save_dir,'sub%s_load5_day%s_delay.csv'%(sub,day)),index=False)
    events_encode.to_csv(os.path.join(new_evt_save_dir,'sub%s_load5_day%s_encode.csv'%(sub,day)),index=False)
   


working_dir = 'D:\\working_memory'
os.chdir(working_dir)
saving_dir = 'D:/working_memory/encode_delay_similarity/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

eegs = glob(os.path.join(working_dir,'*l5*.vhdr'))
evts = glob(os.path.join(working_dir+'\\EVT_load5','*_encode.csv'))
n_interpolates = np.array([1,4,32])
consensus_percs = np.linspace(0,1.0,11)
for raw_,evt in zip(eegs,evts):
    print(raw_,evt)
    events = pd.read_csv(evt)
    events = events[['tms','code','Recode']].values.astype(int)
    events[:,1] = 0
    events = events[::5,:]
    event_id = {'0':0,'1':1}
    raw = mne.io.read_raw_brainvision(raw_,preload=True,montage='standard_1020',eog=['LOc','ROc','Aux1'])
    raw.set_channel_types({'STI 014':'stim','Aux1':'stim','LOc':'eog','ROc':'eog'})
    raw.set_eeg_reference().apply_proj()
    picks = mne.pick_types(raw.info,eeg=True,eog=True)
    raw.filter(1,40,picks=picks,fir_design='firwin')
    raw.notch_filter(np.arange(60,241,60),picks=picks,fir_design='firwin')
    epochs = mne.Epochs(raw,events=events,event_id=event_id,tmin=0,tmax=16,baseline=(15.8,16),preload=True,picks=picks,detrend=1,proj=False)
    #epochs['0'].average().plot()
    sub,load,day = re.findall('\d+',raw_)
    thresh_func = partial(compute_thresholds,picks=picks,method='bayesian_optimization',random_state=12345)
    ar = LocalAutoRejectCV(n_interpolates,consensus_percs,picks=picks,thresh_func=thresh_func)
    ar.fit(epochs)
    print('transform the data')
    epochs = ar.transform(epochs)
    epochs.pick_types(meg=False,eeg=True,eog=False)
    epochs.save(os.path.join(saving_dir,'sub_%s_load%s_day%s_encode_delay-epo.fif'%(sub,load,day)))






















