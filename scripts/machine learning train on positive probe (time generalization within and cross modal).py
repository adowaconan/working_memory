# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:09:45 2018

@author: ning
python "D:/working_memory/working_memory/scripts/machine learning train on positive probe (time generalization within and cross modal).py"

within the probe, we train test the classifier over different time points

cross-modal: train different classifiers at different time points, test the classifiers at different time points of the encode,
testing should be done only in positive probe trials

"""
if __name__ == '__main__':#  the way to force parellel processing
    import os
    os.chdir('D:/working_memory/working_memory/scripts')
    from helper_functions import make_clf#,prediction_pipeline
    import numpy as np
    import mne
    from matplotlib import pyplot as plt
    import pandas as pd
    from scipy import stats as stats
    import re
    working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
    saving_dir = 'D:/working_memory/working_memory/results/train_probe_test_encode/within_cross/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    from glob import glob
    from tqdm import tqdm
    from sklearn.model_selection import StratifiedKFold
    from mne.decoding import GeneralizingEstimator
    
    condition = 'load2'
    event_dir = 'D:\\working_memory\\EVT\\*_probe.csv'
    epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
    event_files = glob(event_dir)
    missing = np.hstack([np.arange(11,17),[18]])#missing 26  and 64
    df = {'sub':[],'image1':[],'image2':[],'method':[],'positive_vs_negative':[],
          'load':[]}
    for e, e_ in zip(epoch_files,event_files):
    #e = epoch_files[0] # debugging stuff
    #e_= event_files[0] # debugging stuff
    
        epochs = mne.read_epochs(e,preload=True)
        epochs.resample(100)
        # # experiment setting
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
        # the original event file before artifact correction
        original_events = pd.read_csv(e_)
        labels = epochs.events[:,-1]
        onset_times = epochs.events[:,0]
        # to get which trials are left in the processed data
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
        idx_positive = np.array(labels,dtype=bool)# get the rows of positive probe trials
        image1 = epochs.copy().crop(-10,-8).get_data()[idx_positive,:,:200]
        image2 = epochs.copy().crop(-8,-6).get_data()[idx_positive,:,:200]
        test_data = np.concatenate([image1,image2],axis=0)
        # get testing labels in encoding
        m1=np.array(working_trial_orders['probe'] == working_trial_orders['image1'],dtype=int)[idx_positive]
        m2=np.array(working_trial_orders['probe'] == working_trial_orders['image2'],dtype=int)[idx_positive]
        test_label = np.concatenate([m1,m2])
        
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
        clf = make_clf(voting=False)# use less computational expensive classifiers
        scores_within = []
        scores_encode = []
        # time generalization within probe and cross modal
        for train,test in tqdm(cv.split(probe,labels)):
            X = probe[train]
            y = labels[train]
            time_gen = GeneralizingEstimator(clf,scoring='roc_auc',n_jobs=4)# define the time generalization model
            time_gen.fit(X,y) # fit the model using probe period
            scores_ = time_gen.score(probe[test],labels[test])# test the model in probe period
            scores_cross = time_gen.score(test_data,test_label)# test the model in encode period
            scores_within.append(scores_)
            scores_encode.append(scores_cross)
        
        scores_within = np.array(scores_within)
        scores_encode = np.array(scores_encode)
        # plotting
        vmax = np.min([scores_within.flatten().max(),scores_encode.flatten().max()])-.1
        fig,axes = plt.subplots(figsize=(16,7),ncols=2)
        ax=axes[0]
        im=ax.imshow(scores_within.mean(0),origin='lower',aspect='auto',cmap=plt.cm.coolwarm,
                     extent=[0,2000,0,2000],vmin=.5,vmax=vmax)
        ax.set(xlabel='test time',ylabel='train time',title='train test within probe')
        
        
        ax=axes[1]
        im=ax.imshow(scores_encode.mean(0),origin='lower',aspect='auto',cmap=plt.cm.coolwarm,
                     extent=[0,2000,0,2000],vmin=.5,vmax=vmax)
        ax.set(xlabel='test time',ylabel='train time',title='train on probe, test on encode')
        plt.colorbar(im)
        fig.suptitle('sub_%s,load_%s,day_%s'%(sub,load,day))
        fig.tight_layout()
        fig.savefig(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.png'%(sub,load,day),dpi=300)
        