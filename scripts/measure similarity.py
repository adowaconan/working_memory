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
    plt.close('all')
    epochs = mne.read_epochs(e,preload=True)
    sub,load,day = re.findall('\d+',e)
    # due to mne python add additional last sample to the data, we take it out and make it a even number length for the time dimension
    images = epochs.copy().crop(0,4).get_data()[:,:,:4000]
    delay = epochs.copy().crop(4,9.8).get_data()[:,:,:5800]
    # get the number of trials
    n_trials = images.shape[0]
    # segmenting the data with a 20 ms nonoverlapping sharp window
    images = images.reshape(n_trials,61,20,-1)
    delay = delay.reshape(n_trials,61,20,-1)
    # average over the 20 ms window
    images_ = images.mean(2)
    delay_ = delay.mean(2)
    # compute the euclidean distance and cosine distance in matrix operations
    
    euclidean,cosine = [],[]
    for trial in tqdm(range(images.shape[0]),desc='distance'):
        # euclidean distance
        temp_confusion = sp.distance.cdist(images_[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion)
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(images_[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion)
    # covert to numpy array
    euclidean = np.array(euclidean)
    cosine = np.array(cosine)
    # save the result in a dictionary
    results={'euclidean_distances':euclidean,'cosine_distance':cosine}
    pickle.dump(results,open(os.path.join(working_dir,'similarity measure result_sub%s_load%s_day%s.p'%(sub,load,day)),'wb')  )  
    # take the mean over the trials
    euclidean_mean = euclidean.mean(0)
    # plot the mean euclidean distance over trials
    fig,ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(euclidean_mean,origin='lower',aspect='auto',cmap='RdBu_r',extent=[0,6000,0,4000])
    ax.set(xlabel='Delay',ylabel='Encode',title='Euclidean distance\nsub%s_load%s_day%s'%(sub,load,day))
    plt.colorbar(im)
    fig.savefig(os.path.join(saving_dir,'Euclidean distance_sub%s_load%s_day%s.png'%(sub,load,day)),dpi=300)
    # plot the mean cosine distance over trials
    cosine_mean = cosine.mean(0)
    fig,ax = plt.subplots(figsize=(12,12))
    im = ax.imshow(cosine_mean,origin='lower',aspect='auto',cmap='RdBu_r',extent=[0,6000,0,4000])
    ax.set(xlabel='Delay',ylabel='Encode',title='Cosine distance\nsub%s_load%s_day%s'%(sub,load,day))
    plt.colorbar(im)
    fig.savefig(os.path.join(saving_dir,'Cosine distance_sub%s_load%s_day%s.png'%(sub,load,day)),dpi=300)
    plt.close('all')
#pickle.dump(results,open(os.path.join(working_dir,'similarity measure results.p'),'wb')  )  
    
#python "D:\\working_memory\\working_memory\\scripts\\measure similarity.py"
