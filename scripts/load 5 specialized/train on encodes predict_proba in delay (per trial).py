# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:28:38 2018

@author: ning

"""

import os
os.chdir('D:/working_memory/working_memory/scripts')
from helper_functions import make_clf#,prediction_pipeline
import numpy as np
import mne
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from mne.decoding import get_coef
from sklearn import metrics
#    from scipy import stats as stats
import pickle
import re
working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
saving_dir = 'D:/working_memory/working_memory/results/load_5/train_probe_test_encode/train_test_probe_encode/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
#    from mne.decoding import GeneralizingEstimator

condition = 'load5'
event_dir = 'D:\\working_memory\\EVT_load5\\*_probe.csv'
epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
event_files = glob(event_dir)

e = epoch_files[0]
e_ = event_files[0]

epochs = mne.read_epochs(e,preload=True)
epochs.resample(100)
event = pd.read_csv(e_)
sub,load,day = re.findall('\d+',e)
# get the order of the stimulu
trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load5_WM',header=None)
trial_orders.columns = ['load','image1','image2','image3','image4','image5','target','probe']
trial_orders['target'] = 1- trial_orders['target']
trial_orders["row"] = np.arange(1,41)
original_events = pd.read_csv('D:\\working_memory\\signal detection\\suj%s_wml%s_day%s-photo_WM_TS'%(sub,load,day),sep='\t')
original_events = original_events[np.abs(original_events['TriNo']-80)<5]
event['trial']=[np.where(original_events['TMS']==time_)[0][0]+1 for time_ in event['tms']]
working_trial_orders = trial_orders.iloc[event['trial']-1]

# split data into encode, delay, and probe
images = epochs.copy().crop(-16,-6).get_data()[:,:,:1000]# all 5 images - 10 seconds
delay = epochs.copy().crop(-6,0).get_data()[:,:,:600]# delay - 6 seconds
probe = epochs.copy().crop(0,2).get_data()[:,:,:200]# probe - 2 seconds
labels = np.arange(5)
clfs = []
preds = []
for trial in tqdm(range(images.shape[0]),desc=''):
    X_train = images[trial].reshape(5,61,200)
    X_test = delay[trial]
    clfs_ = [make_clf(vectorized=False,voting='linear',decoding=False).fit(X_train[:,:,ii],
             labels) for ii in range(X_train.shape[-1])]
    preds_ = []
    for clf in clfs_:
        preds_.append([clf.predict_proba(X_test[:,ii].reshape(1,-1)) for ii in range(X_test.shape[-1])])
    preds.append(preds_)
preds = np.array(preds)
preds = preds.reshape(preds.shape[0],preds.shape[1],preds.shape[2],preds.shape[-1])
preds_max = preds.argmax(-1)

cmap = plt.cm.tab10
bounds = [0,1,2,3,4,5]
norm = colors.BoundaryNorm(bounds,cmap.N)
im=plt.imshow(preds_max.mean(0),cmap=cmap,origin='lower',
              aspect='auto',extent=[0,6000,0,2000],norm=norm)
plt.colorbar(im,boundaries=bounds,norm=norm,cmap=cmap,ticks=np.arange(1,6))






































































