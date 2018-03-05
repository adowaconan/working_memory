# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:57:56 2018

@author: ning
python "D:\\working_memory\\working_memory\\scripts\load 5 specialized\\train on positive and test on encode (linear,custom time_gen).py"
"""

import os
os.chdir('D:/working_memory/working_memory/scripts')
from helper_functions import make_clf#,prediction_pipeline
import numpy as np
import mne
from matplotlib import pyplot as plt
#from matplotlib import colors
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

#e = epoch_files[0]
#e_ = event_files[0]
for e,e_ in zip(epoch_files,event_files):
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
    labels = event['Recode'].values # positive probe == 1, negative probe == 0
    
    # train a series of classifiers in probe data points
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
    clfs = []
    patterns = []
    for train,test in tqdm(cv.split(probe,labels),desc='training'):
        X = probe[train]
        y = labels[train]
    #    clfs_ = [make_clf(vectorized=False,voting='linear').fit(X,y)]
        clfs.append([make_clf(vectorized=False,voting='linear').fit(X[:,:,ii],y) for ii in range(X.shape[-1])])
        temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs[-1]])
        patterns.append(temp_patterns)
    patterns=np.array(patterns)
    # test these classifiers in probe data points
    scores_within = []
    
    for fold,(train,test) in tqdm(enumerate(cv.split(probe,labels)),desc='test within'):
        X = probe[test]
        y = labels[test]
        
        scores_ = []
        for clf in clfs[fold]:
            scores_temp = [metrics.roc_auc_score(y,clf.predict_proba(X[:,:,ii])[:,-1]) for ii in range(X.shape[-1])]
            scores_.append(scores_temp)
        scores_within.append(scores_)
    scores_within = np.array(scores_within)
    
    # test these classifier in encode, but we need to make the test data and test labels first
    test_data = images.reshape(-1,61,200)
    test_labels = working_trial_orders[['image1','image2','image3','image4','image5']].values == working_trial_orders['probe'].values[:,np.newaxis]
    test_labels = test_labels.flatten()
    
    scores_cross = []
    for fold in tqdm(range(5),desc='test cross'):
        X = test_data
        y = test_labels
        scores_ = []
        for clf in clfs[fold]:
            scores_temp = [metrics.roc_auc_score(y,clf.predict_proba(X[:,:,ii])[:,-1]) for ii in range(X.shape[-1])]
            scores_.append(scores_temp)
        scores_cross.append(scores_)
    scores_cross = np.array(scores_cross)
    to_save = {'within':scores_within,'cross':scores_cross}
    pickle.dump(to_save,open(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.p'%(sub,load,day),'wb'))
    info = epochs.info
    evoked = mne.EvokedArray(-patterns.mean(0).T,info)
    evoked.times = np.linspace(0,2,probe.shape[-1])
    evoked.save(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s-ave.fif'%(sub,load,day))
    
    
    fig = plt.figure(figsize=(12,26))
    # subplot row 1 - temporal generalization
    ax=fig.add_subplot(421)
    vmax=[.7,.6]
    im = ax.imshow(scores_within.mean(0),extent=[0,2000,0,2000],origin='lower',aspect='auto',
                   cmap=plt.cm.coolwarm,vmin=.5,vmax=vmax[0])
    driver = make_axes_locatable(ax)
    cax = driver.append_axes('right',size='5%',pad=0.05)
    fig.colorbar(im,cax=cax, )
    ax.set(xlabel='test time',ylabel='train time',title='train test within probe')
    ax=fig.add_subplot(422)
    im = ax.imshow(scores_cross.mean(0),extent=[0,2000,0,2000],origin='lower',aspect='auto',
                   cmap=plt.cm.coolwarm,vmin=.5,vmax=vmax[1])
    driver = make_axes_locatable(ax)
    cax = driver.append_axes('right',size='5%',pad=0.05)
    fig.colorbar(im,cax=cax, )
    ax.set(xlabel='test time',title='train on probe test on encode')
    fig.suptitle('decode positive probe trial vs negative probe trial\nsub_%s,load_%s,day_%s'%(sub,load,day))
    # subplot row 2 - temporal decoding
    ax=fig.add_subplot(412)
    decoding_mean = scores_within.mean(0).diagonal()
    decoding_std = scores_within.std(0).diagonal()
    kernel_size = 20
    decoding_mean_smooth = np.convolve(decoding_mean,[1/kernel_size for ii in range(kernel_size)],'same')
    decoding_std_smooth = np.convolve(decoding_std,[1/kernel_size for ii in range(kernel_size)],'same')
    ax.plot(evoked.times*1000,decoding_mean,color='k',alpha=1.,
            label='Decoding scores (uniform-%d kernel smoothed)'%kernel_size)
    ax.fill_between(evoked.times*1000,
                    decoding_mean+decoding_std_smooth,
                    decoding_mean-decoding_std_smooth,
                    color='red',alpha=0.5,)
    ax.axhline(0.5,color='blue',alpha=.6,linestyle='--',label='Chance level')
    ax.set(xlim=(0,2000),xlabel='Time (ms)',ylabel='Classifi.score (AUC)',title='Train-test at same time (Probe)')
    ax.legend()
    # subplot row 3 - topomap of patterns
    ax = fig.add_subplot(413)
    evoked.plot(show=False,axes=ax,spatial_colors=True,titles='Neg Probe - Pos Probe')
    
    axes=[]
    for k,time_ in enumerate([250,500,750,1000,1250,1500,1750]):
        axes.append(fig.add_subplot(4,7,k+22))
    mne.viz.plot_evoked_topomap(evoked,times=np.array([250,500,750,1000,1250,1500,1750])/1000,
                                    show=False,axes=axes,colorbar=False)
    fig.savefig(saving_dir+'diff_scale_within_cross_modal_time_generalization_sub%sload%sday%s.png'%(sub,load,day),
                dpi=400)




























































