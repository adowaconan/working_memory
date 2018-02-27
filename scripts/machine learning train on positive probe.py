# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:00:20 2018

@author: ning
"""
import os
os.chdir('D:/working_memory/working_memory/scripts')
from helper_functions import make_clf,row_selection,prediction_pipeline
import numpy as np
import mne
from matplotlib import pyplot as plt
import pandas as pd
import re
working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'

from glob import glob
from tqdm import tqdm


from sklearn.model_selection import StratifiedKFold
from sklearn import metrics



condition = 'load2'
event_dir = 'D:\\working_memory\\EVT\\*_probe.csv'
epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
event_files = glob(event_dir)
missing = np.hstack([np.arange(11,17),[18]])#missing 26  and 64
df = {'sub':[],'image1':[],'image2':[],'method':[],'positive_vs_negative':[],
      'load':[]}
for e, e_ in zip(epoch_files,event_files):
#e = epoch_files[0]
#e_= event_files[0]

    epochs = mne.read_epochs(e,preload=True)
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
    probe = epochs.copy().crop(0,2).get_data()[:,:,:200]
    # get testing data in encoding
    image1 = epochs.copy().crop(-10,-8).get_data()[:,:,:200]
    image2 = epochs.copy().crop(-8,-6).get_data()[:,:,:200]
    
    # not over fit the model
    predictive_measurements={'predictive ability of order':[],'predictive ability of positive and negative stimuli':[]}
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
    for train,test in cv.split(probe,labels):
        clf = make_clf(hard_soft='soft',voting=True)
        clf.fit(probe[train],labels[train])
        #print(metrics.classification_report(labels[test],clf.predict(probe[test])))
        ck = prediction_pipeline(labels,[image1,image2],clf,working_trial_orders)
        for key,value in ck.items():
            predictive_measurements[key].append(value)
    print('subject',sub,'load',load,'5-fold cv','voting classifier')
    for key,value in predictive_measurements.items():
        print(key,np.mean(value,axis=0))        
    df['sub'].append(int(sub))
    df['load'].append(int(load))
    df['image1'].append(np.mean(predictive_measurements['predictive ability of order'],axis=0)[0])
    df['image2'].append(np.mean(predictive_measurements['predictive ability of order'],axis=0)[1])
    df['positive_vs_negative'].append(np.mean(predictive_measurements['predictive ability of positive and negative stimuli']))
    df['method'].append('5-fold cross validation')
    # what if I over fit the model?
    clf = make_clf()
    clf.fit(probe,labels)
    ck = prediction_pipeline(labels,[image1,image2],clf,working_trial_orders)
    print('subject',sub,'load',load,'over fit',)
    for key,value in ck.items():
        print(key,value)  
    df['sub'].append(int(sub))
    df['load'].append(int(load))
    df['image1'].append(ck['predictive ability of order'][0])
    df['image2'].append(ck['predictive ability of order'][1])
    df['positive_vs_negative'].append(ck['predictive ability of positive and negative stimuli'])
    df['method'].append('over fitting')

df = pd.DataFrame(df)



























































