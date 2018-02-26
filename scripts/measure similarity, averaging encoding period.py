# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:59:20 2018

@author: ning
"""

import mne
import os
from glob import glob
#import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
#import scipy.io as so
import re
import scipy.spatial as sp
import pickle

working_dir = 'D:/working_memory/encode_delay_similarity/'
saving_dir = 'D:\\working_memory\\working_memory\\results\\similarity measure_average encode/pdfs/'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
epochs_ = glob(os.path.join(working_dir,'*load2*-epo.fif'))
window_size = 10

for e in epochs_:
    results = {}
    plt.close('all')
    epochs = mne.read_epochs(e,preload=True)
    sub,load,day = re.findall('\d+',e)
    image1 = epochs.copy().crop(0,2).get_data()[:,:,:2000]*1e6
    image2 = epochs.copy().crop(2,4).get_data()[:,:,:2000]*1e6
    delay = epochs.copy().crop(4,9.8).get_data()[:,:,:5800]*1e6
    # get the number of trials
    n_trials = image1.shape[0]
    # segmenting the data with a window_size ms nonoverlapping sharp window
    
    delay = delay.reshape(n_trials,61,window_size,-1)
    times_  = np.linspace(0,6000,delay.shape[-1])
    # average within coresponding windows
#    image1_ = image1.mean(2)
#    image2_ = image2.mean(2)
    delay_ = delay.mean(2)
    # image1
    euclidean,cosine = [],[]
    for trial in tqdm(range(image1.shape[0]),desc='image 1, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image1[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image1[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image1']=[np.array(euclidean),np.array(cosine)]
    
    # image2
    euclidean,cosine = [],[]
    for trial in tqdm(range(image1.shape[0]),desc='image 2, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image2[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image2[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image2']=[np.array(euclidean),np.array(cosine)]
    pickle.dump(results,open(working_dir+'averaging encoding (euclidean and cosine)_sub%s_load%s_day%s'%(sub,load,day),'wb'))
    
    # plotting
    ## Euclidean distance
    
    fig,ax = plt.subplots(figsize=(16,8),)
    title='Euclidean distance\nsub%s_load%s_day%s'%(sub,load,day)
    linestyle=['-','--']
    for ii in [0,1]:
        euclidean_1_mean = np.mean(results['image%d'%(ii+1)][0],axis=0)
        euclidean_1_se = np.std(results['image%d'%(ii+1)][0],axis=0)/np.sqrt(n_trials)
        
        ax.plot(times_,euclidean_1_mean,color='k',linestyle=linestyle[ii],
                lw=2,alpha=1.,label='image %d'%(ii+1))
        ax.fill_between(times_,euclidean_1_mean+euclidean_1_se,
                        euclidean_1_mean-euclidean_1_se,color='red',alpha=.3,)#label='SE')
    ax.legend(loc='best')
    ax.set(title=title,xlabel='Time (ms)',
           ylabel='Euclidean Distance',xlim=(0,6000))
    fig.savefig(saving_dir+'Euclidean distance_sub%s_load%s_day%s.pdf'%(sub,load,day),dpi=300)
    
    # plotting
    ## Cosine distance
    
    fig,ax = plt.subplots(figsize=(16,8),)
    title='Cosine distance\nsub%s_load%s_day%s'%(sub,load,day)
    linestyle=['-','--']
    for ii in [0,1]:
        cosine_1_mean = np.mean(results['image%d'%(ii+1)][1],axis=0)
        cosine_1_se = np.std(results['image%d'%(ii+1)][1],axis=0)/np.sqrt(n_trials)
        
        ax.plot(times_,cosine_1_mean,color='k',linestyle=linestyle[ii],
                lw=2,alpha=1.,label='image %d'%(ii+1))
        ax.fill_between(times_,cosine_1_mean+cosine_1_se,
                        cosine_1_mean-cosine_1_se,color='red',alpha=.3,)#label='SE')
    ax.legend(loc='best')
    ax.set(title=title,xlabel='Time (ms)',
           ylabel='Cosine Distance',xlim=(0,6000))
    fig.savefig(saving_dir+'Cosine distance_sub%s_load%s_day%s.pdf'%(sub,load,day),dpi=300)
    plt.close('all')
    
    



epochs_ = glob(os.path.join(working_dir,'*load5*-epo.fif'))
window_size = 10

for e in epochs_:
    results = {}
    plt.close('all')
    epochs = mne.read_epochs(e,preload=True)
    sub,load,day = re.findall('\d+',e)
    image1 = epochs.copy().crop(0,2).get_data()[:,:,:2000]*1e6
    image2 = epochs.copy().crop(2,4).get_data()[:,:,:2000]*1e6
    image3 = epochs.copy().crop(4,6).get_data()[:,:,:2000]*1e6
    image4 = epochs.copy().crop(6,8).get_data()[:,:,:2000]*1e6
    image5 = epochs.copy().crop(8,10).get_data()[:,:,:2000]*1e6
    delay = epochs.copy().crop(10,15.8).get_data()[:,:,:5800]*1e6
    # get the number of trials
    n_trials = image1.shape[0]
    # segmenting the data with a window_size ms nonoverlapping sharp window
    
    delay = delay.reshape(n_trials,61,window_size,-1)
    times_  = np.linspace(0,6000,delay.shape[-1])
    # average within coresponding windows
#    image1_ = image1.mean(2)
#    image2_ = image2.mean(2)
    delay_ = delay.mean(2)
    # image1
    euclidean,cosine = [],[]
    for trial in tqdm(range(image1.shape[0]),desc='image 1, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image1[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image1[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image1']=[np.array(euclidean),np.array(cosine)]
    
    # image2
    euclidean,cosine = [],[]
    for trial in tqdm(range(image2.shape[0]),desc='image 2, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image2[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image2[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image2']=[np.array(euclidean),np.array(cosine)]
    
    # image3
    euclidean,cosine = [],[]
    for trial in tqdm(range(image3.shape[0]),desc='image 3, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image3[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image3[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image3']=[np.array(euclidean),np.array(cosine)]
    
    # image4
    euclidean,cosine = [],[]
    for trial in tqdm(range(image4.shape[0]),desc='image 4, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image4[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image4[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image4']=[np.array(euclidean),np.array(cosine)]
    
    # image5
    euclidean,cosine = [],[]
    for trial in tqdm(range(image5.shape[0]),desc='image 5, sub%s_load%s_day%s'%(sub,load,day)):
        # euclidean distance
        temp_confusion = 0-sp.distance.cdist(image5[trial].T,delay_[trial].T,'euclidean')
        euclidean.append(temp_confusion.mean(0))
        # cosine distance
        # subtract from 1 because the algorithm have cosine distance subtracted from 1,
        # so subtract it from 1 will give us the value of cosine 
        temp_confusion = 1 - sp.distance.cdist(image5[trial].T,delay_[trial].T,'cosine')
        cosine.append(temp_confusion.mean(0))
    results['image5']=[np.array(euclidean),np.array(cosine)]
    pickle.dump(results,open(working_dir+'averaging encoding (euclidean and cosine)_sub%s_load%s_day%s'%(sub,load,day),'wb'))
    
    # plotting
    ## Euclidean distance
    
    fig,ax = plt.subplots(figsize=(16,8),)
    title='Euclidean distance\nsub%s_load%s_day%s'%(sub,load,day)
    color=['k','red','blue','green','pink']
    for ii in [0,1,2,3,4]:
        euclidean_1_mean = np.mean(results['image%d'%(ii+1)][0],axis=0)
        euclidean_1_se = np.std(results['image%d'%(ii+1)][0],axis=0)/np.sqrt(n_trials)
        
        ax.plot(times_,euclidean_1_mean,color=color[ii],linestyle='-',
                lw=2,alpha=1.,label='image %d'%(ii+1))
        ax.fill_between(times_,euclidean_1_mean+euclidean_1_se,
                        euclidean_1_mean-euclidean_1_se,color='grey',alpha=.3,)#label='SE')
    ax.legend(loc='best')
    ax.set(title=title,xlabel='Time (ms)',
           ylabel='Euclidean Distance',xlim=(0,6000))
    fig.savefig(saving_dir+'Euclidean distance_sub%s_load%s_day%s.pdf'%(sub,load,day),dpi=300)
    
    # plotting
    ## Cosine distance
    
    fig,ax = plt.subplots(figsize=(16,8),)
    title='Cosine distance\nsub%s_load%s_day%s'%(sub,load,day)
    color=['k','red','blue','green','pink']
    for ii in [0,1,2,3,4]:
        cosine_1_mean = np.mean(results['image%d'%(ii+1)][1],axis=0)
        cosine_1_se = np.std(results['image%d'%(ii+1)][1],axis=0)/np.sqrt(n_trials)
        
        ax.plot(times_,cosine_1_mean,color=color[ii],linestyle='-',
                lw=2,alpha=1.,label='image %d'%(ii+1))
        ax.fill_between(times_,cosine_1_mean+cosine_1_se,
                        cosine_1_mean-cosine_1_se,color='grey',alpha=.3,)#label='SE')
    ax.legend(loc='best')
    ax.set(title=title,xlabel='Time (ms)',
           ylabel='Cosine Distance',xlim=(0,6000))
    fig.savefig(saving_dir+'Cosine distance_sub%s_load%s_day%s.pdf'%(sub,load,day),dpi=300)
    plt.close('all')
















































