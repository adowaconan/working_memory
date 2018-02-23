# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:30:55 2018

@author: ning
"""

import mne
import os
from glob import glob
#import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib import pyplot as plt
from tqdm import tqdm
#import scipy.io as so
import re
import scipy.spatial as sp
import pickle

working_dir = 'D:/working_memory/encode_delay_similarity/'
saving_dir = 'D:\\working_memory\\working_memory\\results\\similarity measure'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
epochs_ = glob(os.path.join(working_dir,'*-epo.fif'))
results = {}
for e in epochs_:
    epochs = mne.read_epochs(e,preload=True)
    images = epochs.copy().crop(0,4).get_data()[:,:,:4000]
    delay = epochs.copy().crop(4,9.8).get_data()[:,:,:5800]
    
    n_trials = images.shape[0]
    images = images.reshape(n_trials,61,20,-1)
    delay = delay.reshape(n_trials,61,20,-1)
    
    confusion = []
    for trial in tqdm(range(images.shape[0])):
        temp_confusion = np.zeros((images.shape[-1],delay.shape[-1]))
        for ii in range(images.shape[-1]):
            for jj in range(delay.shape[-1]):
                temp_confusion[ii,jj] = euclidean_distances(np.mean(images[trial,:,:,ii],axis=1).reshape(1,-1),
                                                            np.mean(delay[trial,:,:,jj], axis=1).reshape(1,-1))
        
        confusion.append(temp_confusion)
    
    confusion = np.array(confusion)
    sub,load,day = re.findall('\d+',e)
    results['sub%s_load%s_day%s'%(sub,load,day)]={'euclidean_distances':confusion}
    
    confusion_mean = confusion.mean(0)

    fig,ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(confusion_mean,origin='lower',aspect='auto',cmap='RdBu_r',extent=[0,6000,0,4000])
    ax.set(xlabel='Delay',ylabel='Encode',title='Euclidean distance\nsub%s_load%s_day%s'%(sub,load,day))
    plt.colorbar(im)
    fig.savefig(os.path.join(saving_dir,'Euclidean distance_sub%s_load%s_day%s.png'%(sub,load,day)),dpi=300)
    
    confusion = []
    for trial in tqdm(range(images.shape[0])):
        temp_confusion = np.zeros((images.shape[-1],delay.shape[-1]))
        for ii in range(images.shape[-1]):
            for jj in range(delay.shape[-1]):
                temp_confusion[ii,jj] = 1- sp.distance.cdist(images[trial,:,:,ii],delay[trial,:,:,jj],'cosine')[0,1]
        
        confusion.append(temp_confusion)
    
    confusion = np.array(confusion)
    results['sub%s_load%s_day%s'%(sub,load,day)]['cosine_distance']=confusion
    
    fig,ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(confusion_mean,origin='lower',aspect='auto',cmap='RdBu_r',extent=[0,6000,0,4000])
    ax.set(xlabel='Delay',ylabel='Encode',title='Cosine distance\nsub%s_load%s_day%s'%(sub,load,day))
    plt.colorbar(im)
    fig.savefig(os.path.join(saving_dir,'Cosine distance_sub%s_load%s_day%s.png'%(sub,load,day)),dpi=300)

pickle.dump(results,open(os.path.join(working_dir,'similarity measure results.p'),'wb')  )  
    