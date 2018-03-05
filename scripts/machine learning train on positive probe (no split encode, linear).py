# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:09:45 2018

@author: ning
python "D:/working_memory/working_memory/scripts/machine learning train on positive probe (no split encode, linear).py"

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
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pandas as pd
    from mne.decoding import get_coef
#    from scipy import stats as stats
    import pickle
    import re
    working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
    saving_dir = 'D:/working_memory/working_memory/results/train_probe_test_encode/full_window_linear/'
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
        # get testing data in encoding,delay
        
        image1 = epochs.copy().crop(-10,-8).get_data()[:,:,:200]
        image2 = epochs.copy().crop(-8,-6).get_data()[:,:,:200]
        delay = epochs.copy().crop(-6,0).get_data()[:,:,:600]
        # get testing labels in encoding
        idx_positive = np.array(labels,dtype=bool)# get the rows of positive probe trials
        m1=np.array(working_trial_orders['probe'] == working_trial_orders['image1'],dtype=int)[idx_positive]
        m2=np.array(working_trial_orders['probe'] == working_trial_orders['image2'],dtype=int)[idx_positive]
        test_label = np.concatenate([m1,m2])
        
        data = []
        
        for ii,(im1,im2,(_,trial_)) in enumerate(zip(image1,image2,working_trial_orders.iterrows())):
           
            if (trial_['probe']==trial_['image1']) or (trial_['probe']==trial_['image2']):
                 
#                print(np.array([(trial_['probe']==trial_['image1']) , (trial_['probe']==trial_['image2'])],dtype=int))
                if np.sum(np.array([(trial_['probe']==trial_['image1']) , (trial_['probe']==trial_['image2'])],dtype=int)==[1,0])==2:
#                    print(np.array([(trial_['probe']==trial_['image1']) , (trial_['probe']==trial_['image2'])],dtype=int),'\n')
                    data.append(np.concatenate([im1,im2,delay[ii],probe[ii]],axis=1))
                elif np.sum(np.array([(trial_['probe']==trial_['image1']) , (trial_['probe']==trial_['image2'])],dtype=int)==[0,1])==2:
#                    print(np.array([(trial_['probe']==trial_['image1']) , (trial_['probe']==trial_['image2'])],dtype=int),'\n')
                    data.append(np.concatenate([im2,im1,delay[ii],probe[ii]],axis=1))
            else:
                data.append(np.concatenate([im1,im2,delay[ii],probe[ii]],axis=1))
        data = np.array(data)
        data_ = data.copy()
        data = np.concatenate([delay,probe],axis=2)        
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
        clf = make_clf(voting='linear',vectorized=False)# use less computational expensive linear classifiers
        scores = []
        patterns = []
        # time generalization within probe and cross modal
        for train,test in tqdm(cv.split(data,labels)):
            X = data[train]
            y = labels[train]
            time_gen = GeneralizingEstimator(clf,scoring='roc_auc',n_jobs=4)# define the time generalization model
            time_gen.fit(X,y) # fit the model using probe period
            scores_ = time_gen.score(data[test],labels[test])# test the model in probe period
            scores.append(scores_)
            temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in time_gen.estimators_])
            patterns.append(temp_patterns)
        
        scores = np.array(scores)
        patterns = np.array(patterns)
        
        info = epochs.info
        evoked = mne.EvokedArray(-patterns.mean(0).T,info)
        evoked.times = np.linspace(-6,2,scores.shape[-1])
        evoked.save(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s-ave.fif'%(sub,load,day))
        to_save = {'scores':scores,}
        pickle.dump(to_save,open(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.p'%(sub,load,day),'wb'))
        # plotting, contour
        vmax = .65
        fig,axes = plt.subplots(figsize=(10,10))
        ax=axes
        im = ax.imshow(scores.mean(0),extent=[-6000,2000,-6000,2000],cmap=plt.cm.coolwarm,
                        vmin=.5,vmax=vmax,origin='lower',aspect='auto')
        ax.set(xlabel='test time',ylabel='train time',title='train test within probe\nsub_%s,load_%s,day_%s'%(sub,load,day))
        driver = make_axes_locatable(ax)
        cax = driver.append_axes('right',size='5%',pad=0.05)
        fig.colorbar(im,cax=cax,)
        ax.axvline(-8000,linestyle='--',color='k');ax.axvline(-6000,linestyle='--',color='k');ax.axvline(0,linestyle='--',color='k')
        ax.axhline(-8000,linestyle='--',color='k');ax.axhline(-6000,linestyle='--',color='k');ax.axhline(0,linestyle='--',color='k')
        
#        fig.tight_layout()
        fig.savefig(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.png'%(sub,load,day),dpi=300)
        
        
        fig = plt.figure(figsize=(16,16))
        
        ax=fig.add_subplot(212)
        decoding_mean = scores.mean(0).diagonal()
        decoding_std = scores.std(0).diagonal()
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
        ax.set(xlim=(-6000,2000),xlabel='Time (ms)',ylabel='Classifi.score (AUC)',title='Train-test at same time (Probe)\nsub%sload%sday%s'%(sub,load,day))
        ax.legend()
        
        
        fig.tight_layout(pad=3.5)
        fig.savefig(saving_dir+'diff_scale_within_cross_modal_time_generalization_sub%sload%sday%s.png'%(sub,load,day),dpi=300)
        plt.close('all')
        
        



