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
