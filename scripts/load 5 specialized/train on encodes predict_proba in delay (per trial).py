# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:28:38 2018

@author: ning
decoding the order effect of the encoding period - there is no order effect
predict performance using encoding period signals
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
from scipy import stats as stats
import pickle
import re
working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
saving_dir = 'D:/working_memory/working_memory/results/order effect/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,permutation_test_score
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer,LinearModel
from sklearn.utils import shuffle
#    from mne.decoding import GeneralizingEstimator

condition = 'load5'
event_dir = 'D:\\working_memory\\EVT_load5\\*_probe.csv'
epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
event_files = glob(event_dir)

e = epoch_files[9]
e_ = event_files[9]

epochs = mne.read_epochs(e,preload=True)
epochs.resample(100)
event = epochs.events
sub,load,day = re.findall('\d+',e)
# get the order of the stimulu
trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load5_WM',header=None)
trial_orders.columns = ['load','image1','image2','image3','image4','image5','target','probe']
trial_orders['target'] = 1- trial_orders['target']
trial_orders["row"] = np.arange(1,41)
original_events = pd.read_csv('D:\\working_memory\\signal detection\\suj%s_wml%s_day%s-photo_WM_TS'%(sub,load,day),sep='\t')
original_events = original_events[np.abs(original_events['TriNo']-80)<5]
event = pd.DataFrame(event,columns=['tms','e','Comnt'])
event['trial']=[np.where(original_events['TMS']==time_)[0][0]+1 for time_ in event['tms']]
working_trial_orders = trial_orders.iloc[event['trial']-1]
working_events = original_events.iloc[event['trial']-1]

# split data into encode, delay, and probe
images = epochs.copy().crop(-16,-6).get_data()[:,:,:1000]# all 5 images - 10 seconds
delay = epochs.copy().crop(-6,0).get_data()[:,:,:600]# delay - 6 seconds
probe = epochs.copy().crop(0,2).get_data()[:,:,:200]# probe - 2 seconds

# prepare train-test data in the encodeing period
#X = np.concatenate(np.split(images,5,axis=-1),axis=0)# this is the correct way to stack the data
X = delay
Comnt_dict = {'Correct Rejection':1,'Hit':1,'False Alarm':0,'Miss':0,'No Response':0}
labels = working_events['Comnt'].map(Comnt_dict)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
clf = []
clf.append(('Vec',Vectorizer()))
clf.append(('ST',StandardScaler()))
clf.append(('EST',SVC(random_state=12345,kernel='linear',class_weight='balanced')))
clf = Pipeline(clf,)


res = [[permutation_test_score(clf,X[:,:,ii],labels,cv=cv,n_permutations=int(5e2),verbose=1,n_jobs=4)]for ii in range(X.shape[-1])]

def ST(X):
    return (X - X.mean(0))/X.std(0)
def Comnt_dict(x):
    if x == 'Correct Rejection':
        return 1
    elif x == 'Hit':
        return 1
    elif x == 'False Alarm':
        return 0
    elif x == 'Miss':
        return 0
    else:
        return 0

X = np.concatenate([ST(mne.read_epochs(e).resample(100,n_jobs=4).copy().crop(-6,0).get_data()[:,:,:600]) for e in epoch_files[:-4]],axis=0)
X=[]
labels = []
for e,e_ in zip(epoch_files[:-4],event_files[:-4]):
    epochs = mne.read_epochs(e,preload=True)
    epochs.resample(100,n_jobs=4)
    xx = stats.zscore(epochs.copy().crop(-6,0).get_data()[:,:,:600],axis=0,ddof=1)
    X.append(xx)
    event = epochs.events
    sub,load,day = re.findall('\d+',e)
    # get the order of the stimulu
    trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load5_WM',header=None)
    trial_orders.columns = ['load','image1','image2','image3','image4','image5','target','probe']
    trial_orders['target'] = 1- trial_orders['target']
    trial_orders["row"] = np.arange(1,41)
    original_events = pd.read_csv('D:\\working_memory\\signal detection\\suj%s_wml%s_day%s-photo_WM_TS'%(sub,load,day),sep='\t')
    original_events = original_events[np.abs(original_events['TriNo']-80)<5]
    if original_events.shape == (0,6):
        original_events = pd.read_csv('D:\\working_memory\\signal detection\\suj%s_wml%s_day%s-photo_WM_TS'%(sub,load,day),sep='\t')
        original_events = original_events[original_events['TriNo']==8]
    event = pd.DataFrame(event,columns=['tms','e','Comnt'])
    event['trial']=[np.where(original_events['TMS']==time_)[0][0]+1 for time_ in event['tms']]
    working_trial_orders = trial_orders.iloc[event['trial']-1]
    working_events = original_events.iloc[event['trial']-1]
    
    labels_ = working_events['Comnt'].apply(Comnt_dict)
    labels.append(labels_)
labels = np.concatenate(labels).astype(int)
X = np.concatenate(X).astype(np.float32)

cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
clf = []
clf.append(('Vec',Vectorizer()))
clf.append(('ST',StandardScaler()))
clf.append(('EST',SVC(random_state=12345,kernel='linear',class_weight='balanced')))
clf = Pipeline(clf,)
a,b,c = permutation_test_score(clf,X,labels,cv=cv,n_permutations=int(5e2),verbose=1,n_jobs=4)


cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
clf = []
clf.append(('ST',StandardScaler()))
clf.append(('EST',SVC(random_state=12345,kernel='linear',class_weight='balanced')))
clf = Pipeline(clf,)
res = [[permutation_test_score(clf,X[:,:,ii],labels,cv=cv,n_permutations=int(5e2),verbose=1,n_jobs=4)]for ii in range(X.shape[-1])]







































