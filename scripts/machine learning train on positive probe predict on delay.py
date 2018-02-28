# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:45:16 2018

@author: ning

I train a classifier to distinguish positive probe (old image) vs negative probe (new image)
and predict on a window of time in the delay period what the probability is the signal is 
indicating the positive probe image. The prediction is done only on the positive probe trials
"""

import os
os.chdir('D:/working_memory/working_memory/scripts')
from helper_functions import make_clf#,prediction_pipeline
import numpy as np
import mne
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import re
working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
saving_dir = 'D:/working_memory/working_memory/results/train_probe_test_delay/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
from glob import glob
from sklearn.model_selection import StratifiedKFold
import pickle
condition = 'load2'
event_dir = 'D:\\working_memory\\EVT\\*_probe.csv'
epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
event_files = glob(event_dir)
missing = np.hstack([np.arange(11,17),[18]])#missing 26  and 64
df = []
for e, e_ in zip(epoch_files,event_files):
#e = epoch_files[0]
#e_= event_files[0]

    epochs = mne.read_epochs(e,preload=True)
    sub,load,day = re.findall('\d+',e)
    epochs.resample(100)
    trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load2_WM',header=None)
    trial_orders.columns = ['load','image1','image2','target','probe']
    trial_orders['target'] = 1- trial_orders['target']
    trial_orders["row"] = np.arange(1,101)
    sub,load,day = re.findall('\d+',e)
    if int(sub) in missing:
        #print('in')
        idx = np.logical_and(trial_orders.row!=26,trial_orders.row!=64)
        
    else:
        idx = np.array([True] * 100)
    if int(sub) == 11:
        idx[0] = False
    working_trial_orders = trial_orders[idx]
    original_events = pd.read_csv(e_)
    labels = epochs.events[:,-1]
    onset_times = epochs.events[:,0]
    C = []
    for k in original_events.iloc[:,0]:
        if any(k == p for p in onset_times):
            c = 1
        else:
            c = 0
        C.append(c)
    C = np.array(C,dtype=bool)
    working_trial_orders = working_trial_orders.iloc[C,:]
    working_events = original_events.iloc[C,:]
    # get training data in probe
    probe = epochs.copy().crop(0,2).get_data()[:,:,:int(2*epochs.info['sfreq'])]
    # get testing data in encoding
    delay = epochs.copy().crop(-6,-0.2).get_data()[:,:,:int(5.8*epochs.info['sfreq'])]
    
    # not over fit the model
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
    
    delay_dynamic_pred = []
    for train,test in cv.split(probe,labels):
        clf = make_clf(hard_soft='soft',voting=False)
        clf.fit(probe[train],labels[train])
        temporal_idx = 0
        pred_store = []
        for ii in tqdm(range(60000)):
            # only look at the positive probe trials because it make sense to only look at them exclusively
            pos_select = np.array(labels,dtype=bool)
            pred_ = clf.predict_proba(delay[pos_select,:,temporal_idx:temporal_idx+int(2*epochs.info['sfreq'])])
            pred_store.append(pred_)
            temporal_idx += 1
            if delay.shape[-1] - (temporal_idx+int(2*epochs.info['sfreq'])) <=0:
                break
        pred_store = np.array(pred_store)
        delay_dynamic_pred.append(pred_store.mean(1)[:,-1])
    delay_dynamic_pred = np.array(delay_dynamic_pred)
    df.append([int(sub),int(load),int(day),delay_dynamic_pred])
    delay_dynamic_pred_mean = delay_dynamic_pred.mean(0)
    delay_dynamic_pred_se = delay_dynamic_pred.std(0)/np.sqrt(5)
    fig,ax = plt.subplots(figsize=(12,8))
    ax.plot(np.linspace(0,6000,delay_dynamic_pred.shape[1]),delay_dynamic_pred_mean,color='k',alpha=1.,)
    ax.fill_between(np.linspace(0,6000,delay_dynamic_pred.shape[1]),
                                delay_dynamic_pred_mean+delay_dynamic_pred_se,
                                delay_dynamic_pred_mean-delay_dynamic_pred_se,
                                color='red',alpha=.5)
    ax.set(title='sub%s,load%s,day%s,probabilistic prediction of thinking the probe image'%(sub,load,day))
    ax.set(xlabel='delay',ylabel='prod of being the probe image',xlim=(0,6000))
    ax.axhline(0.5,color='blue',linestyle='--',alpha=.8)
    fig.savefig(saving_dir+'sub%s,load%s,day%s,probabilistic prediction of thinking the probe image.png'%(sub,load,day),dpi=400)
    pickle.dump(open(working_dir+'sub%s,load%s,day%s,probabilistic prediction of thinking the probe image'%(sub,load,day),'wb'))
    
                
