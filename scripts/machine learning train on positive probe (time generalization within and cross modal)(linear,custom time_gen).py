# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:09:45 2018

@author: ning
python "D:/working_memory/working_memory/scripts/machine learning train on positive probe (time generalization within and cross modal)(linear,custom time_gen).py"

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
    from sklearn import metrics
#    from scipy import stats as stats
    import pickle
    import re
    working_dir = 'D:/working_memory/encode_delay_prode_RSA_preprocessing/'
    saving_dir = 'D:/working_memory/working_memory/results/train_probe_test_encode/train_test_probe_encode/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    from glob import glob
    from tqdm import tqdm
    from sklearn.model_selection import StratifiedKFold
#    from mne.decoding import GeneralizingEstimator
    
    condition = 'load2'
    event_dir = 'D:\\working_memory\\EVT\\*_probe.csv'
    epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
    event_files = glob(event_dir)
    missing = np.hstack([np.arange(11,17),[18]])#missing 26  and 64
    for e, e_ in zip(epoch_files,event_files):
    #e = epoch_files[0] # debugging stuff
    #e_= event_files[0] # debugging stuff
        sub,load,day = re.findall('\d+',e)
        if os.path.exists(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.p'%(sub,load,day)):
            to_save=pickle.load(open(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.p'%(sub,load,day),'rb'))
            scores_within, scores_cross= to_save['within'],to_save['cross']
            evoked = mne.read_evokeds(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s-ave.fif'%(sub,load,day))[0]
        else:
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
            patterns = []
            # time generalization within probe and cross modal
            time_gen = {'estimators_':[]}
            # train the classifiers in probe time points
            for train,test in tqdm(cv.split(probe,labels)):
                X = probe[train]
                y = labels[train]
                
                
                clfs = []
                for ii,time_ in tqdm(enumerate(range(probe.shape[-1]))):
                    X_ =X[:,:,ii]
                    clf = make_clf(voting='linear',vectorized=False)# use less computational expensive linear classifiers
                    clf.fit(X_,y)
                    clfs.append(clf)
                time_gen['estimators_'].append(clfs)
                temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs])
                patterns.append(temp_patterns)
            # test the classifier at different probe time points
            scores_within = []
            for fold,(train,test) in tqdm(enumerate(cv.split(probe,labels))):
                X = probe[test]
                y = labels[test]
                scores_ = []
                for clf in time_gen['estimators_'][fold]:
                    scores_temp = [metrics.roc_auc_score(y,clf.predict_proba(X[:,:,ii])[:,-1]) for ii in range(X.shape[-1])]
                
                
                    scores_.append(scores_temp)
                scores_within.append(scores_)
            
            scores_within = np.array(scores_within)
            patterns = np.array(patterns)
            info = epochs.info
            evoked = mne.EvokedArray(-patterns.mean(0).T,info)
            evoked.times = np.linspace(0,2,probe.shape[-1])
            evoked.save(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s-ave.fif'%(sub,load,day))
            
            # test the classifier at encode time points
            scores_cross = []
            for fold in tqdm(range(5)):
                X = test_data
                y = test_label
                scores_ = []
                for clf in time_gen['estimators_'][fold]:
                    scores_temp = [metrics.roc_auc_score(y,clf.predict_proba(X[:,:,ii])[:,-1]) for ii in range(X.shape[-1])]
                    scores_.append(scores_temp)
                scores_cross.append(scores_)
            scores_cross = np.array(scores_cross)
            
            
            to_save = {'within':scores_within,'cross':scores_cross,}
            pickle.dump(to_save,open(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.p'%(sub,load,day),'wb'))
        # plotting, contour
        vmax = np.min([scores_within.flatten().max(),scores_cross.flatten().max()])-.1
        fig,axes = plt.subplots(figsize=(16,7),ncols=2)
        ax=axes[0]
        CS = ax.contour(scores_within.mean(0),extent=[0,2000,0,2000],cmap=plt.cm.coolwarm,
                        vmin=.5,vmax=vmax)
        ax.set(xlabel='test time',ylabel='train time',title='train test within probe')
        ax=axes[1]
        CS = ax.contour(scores_cross.mean(0),extent=[0,2000,0,2000],cmap=plt.cm.coolwarm,
                        vmin=.5,vmax=vmax)
        ax.set(xlabel='test time',title='train on probe, test on encode')
        norm= colors.Normalize(vmin=0.5, vmax=CS.vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        driver = make_axes_locatable(ax)
        cax = driver.append_axes('right',size='5%',pad=0.05)
        fig.colorbar(sm,cax=cax, ticks=CS.levels)
        fig.suptitle('sub_%s,load_%s,day_%s'%(sub,load,day))
        fig.tight_layout()
        fig.savefig(saving_dir+'within_cross_modal_time_generalization_sub%sload%sday%s.png'%(sub,load,day),dpi=300)
        
        
        
        fig = plt.figure(figsize=(15,26))
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
        fig.suptitle('sub_%s,load_%s,day_%s'%(sub,load,day))
        
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
        
        ax = fig.add_subplot(413)
        evoked.plot(show=False,axes=ax,spatial_colors=True,titles='noProbe - Probe')
        
        axes=[]
        for k,time_ in enumerate([250,500,750,1000,1250,1500,1750]):
            axes.append(fig.add_subplot(4,7,k+22))
        mne.viz.plot_evoked_topomap(evoked,times=np.array([250,500,750,1000,1250,1500,1750])/1000,
                                        show=False,axes=axes,colorbar=False)
        
        
#        fig.tight_layout(pad=3.5)
        fig.savefig(saving_dir+'diff_scale_within_cross_modal_time_generalization_sub%sload%sday%s.png'%(sub,load,day),
                    dpi=300,bbox_inches='tight')
        plt.close('all')
        
        



