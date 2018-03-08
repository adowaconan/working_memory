# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:28:38 2018

@author: ning
decoding the order effect of the encoding period - there is no order effect
predict performance using encoding period signals
"""
if __name__ == '__main__':
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
    saving_dir = 'D:/working_memory/delay performance/'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    from glob import glob
    from tqdm import tqdm
    from sklearn.model_selection import (StratifiedKFold,permutation_test_score,cross_val_score,LeaveOneOut,
                                         StratifiedShuffleSplit,cross_val_predict)
    from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from mne.decoding import Vectorizer,LinearModel
    from sklearn.utils import shuffle
    from imblearn import under_sampling,ensemble,over_sampling
    from imblearn.pipeline import make_pipeline
    from mne.decoding import GeneralizingEstimator,cross_val_multiscore,SlidingEstimator
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
    
    """
    condition = 'load5'
    event_dir = 'D:\\working_memory\\EVT_load5\\*_probe.csv'
    epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
    event_files = glob(event_dir)
    
    # stack the normalized within subject data together
    #X = np.concatenate([ST(mne.read_epochs(e).resample(100,n_jobs=4).copy().crop(-6,0).get_data()[:,:,:600]) for e in epoch_files[:-4]],axis=0)
    X=[]
    labels = []
    for e,e_ in zip(epoch_files[:-3],event_files[:-3]):
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
        print(working_events['Comnt'])
        labels.append(labels_)
    labels_load5 = np.concatenate(labels).astype(int)
    X_load5 = np.concatenate(X).astype(np.float32)
    
    
    condition = 'load2'
    event_dir = 'D:\\working_memory\\EVT\\*_probe.csv'
    epoch_files = glob(os.path.join(working_dir,'*%s*-epo.fif'%(condition)))
    event_files = glob(event_dir)
    missing = np.hstack([np.arange(11,17),[18]])#missing 26  and 64
    X = []
    labels = []
    for e, e_ in zip(epoch_files[:-3],event_files[:-3]):
    #e = epoch_files[0] # debugging stuff
    #e_= event_files[0] # debugging stuff
        sub,load,day = re.findall('\d+',e)
        epochs = mne.read_epochs(e,preload=True)
        epochs.resample(100,n_jobs=4)
        event = epochs.events
        # # experiment setting
        trial_orders = pd.read_excel('D:\\working_memory\\working_memory\\EEG Load 5 and 2 Design Fall 2015.xlsx',sheetname='EEG_Load2_WM',header=None)
        trial_orders.columns = ['load','image1','image2','target','probe']
        trial_orders['target'] = 1- trial_orders['target']
        trial_orders["row"] = np.arange(1,101)
        sub,load,day = re.findall('\d+',e)
        
        original_events = pd.read_csv('D:\\working_memory\\signal detection\\suj%s_wml%s_day%s-photo_WM_TS'%(sub,load,day),sep='\t')
        original_events = original_events[np.abs(original_events['TriNo']-80)<5]
        if original_events.shape == (0,6):
            original_events = pd.read_csv('D:\\working_memory\\signal detection\\suj%s_wml%s_day%s-photo_WM_TS'%(sub,load,day),sep='\t')
#        print(original_events['Comnt'])
        event = pd.DataFrame(event,columns=['tms','e','Comnt'])
        try:
            event['trial']=[np.where(original_events['TMS']==time_)[0][0]+1 for time_ in event['tms']]
            working_trial_orders = trial_orders.iloc[event['trial']-1]
            working_events = original_events.iloc[event['trial']-1]
            
            labels_ = working_events['Comnt'].apply(Comnt_dict)
            print(working_events['Comnt'],labels_)
            labels.append(labels_)
            X.append(stats.zscore(epochs.copy().crop(-6,0).get_data()[:,:,:600],axis=0,ddof=1))
        except:
            print(sub,load,day)
            pass
#            temp1 = []
#            for time_ in event['tms']:
#                if len(np.where(original_events['TMS']==time_)[0])>0:
#                    temp1.append(np.where(original_events['TMS']==time_)[0][0]+1)
#            temp2 = []
#            for time_ in original_events['TMS']:
#                if len(np.where(event['tms']==time_)[0])>0:
#                    temp2.append(np.where(event['tms']==time_)[0][0]+1)
#            temp=list(set(temp1) & set(temp2))
#            event['trial']=temp
        
    
    X_load2 = np.concatenate(X).astype(np.float32)
    labels_load2 = np.concatenate(labels).astype(int)
    
    data = {'load2':X_load2,'load5':X_load5,'l2':labels_load2,'l5':labels_load5}
    pickle.dump(data,open(saving_dir+'delay_performance_25','wb'))
    """
    data = pickle.load(open(saving_dir+'delay_performance_25','rb'))
    X_load2,X_load5,labels_load2,labels_load5=data['load2'],data['load5'],data['l2'],data['l5']
    
    cv = StratifiedShuffleSplit(n_splits=10,random_state=12345,test_size=.1)
    
    vec = Vectorizer()
    sm = under_sampling.RandomUnderSampler(random_state=12345)
    est = SVC(kernel='linear',class_weight='balanced',random_state=12345)
    
    clf = make_pipeline(vec,sm,est)
    
    # fit in load 2
    clf.fit(X_load2,labels_load2)
    # test in load 5
    print(metrics.classification_report(labels_load5,clf.predict(X_load5)))    
    print(metrics.roc_auc_score(labels_load5,clf.predict(X_load5)))
    
    # fit in load 5
    clf.fit(X_load5,labels_load5)    
    # test in load 2
    print(metrics.classification_report(labels_load2,clf.predict(X_load2)))    
    print(metrics.roc_auc_score(labels_load2,clf.predict(X_load2)))
    # train test in load 2
    scores_within_load2 = []
    scores_cross_load5  = []
    for train,test in cv.split(X_load2,labels_load2):
        time_gen = GeneralizingEstimator(clf,scoring='roc_auc',n_jobs=4)
        time_gen.fit(X_load2[train],labels_load2[train])
        scores_=time_gen.score(X_load2,labels_load2)
        scores__ = time_gen.score(X_load5,labels_load5)
        scores_within_load2.append(scores_)
        scores_cross_load5.append(scores__)
    scores_within_load2 = np.array(scores_within_load2)
    scores_cross_load5 = np.array(scores_cross_load5)
    pickle.dump(scores_within_load2,open(saving_dir+'scores_within_load2.p','wb'))
    pickle.dump(scores_cross_load5,open(saving_dir+'scores_cross_load5','wb'))
    
    # train test in load 5
    scores_within_load5 = []
    scores_cross_load2 = []
    for train,test in cv.split(X_load5,labels_load5):
        time_gen = GeneralizingEstimator(clf,scoring='roc_auc',n_jobs=4)
        time_gen.fit(X_load5[train],labels_load5[train])
        scores_=time_gen.score(X_load5,labels_load5)
        scores__ = time_gen.score(X_load2,labels_load2)
        scores_within_load5.append(scores_)
        scores_cross_load2.append(scores__)
    scores_within_load5 = np.array(scores_within_load5)
    scores_cross_load2 = np.array(scores_cross_load2)
    pickle.dump(scores_within_load5,open(saving_dir+'scores_within_load5','wb'))
    pickle.dump(scores_cross_load2,open(saving_dir+'scores_cross_load2','wb'))
    ###############################################################################################################################################
    ###########################   plotting   ######################################################################################################
    ###############################################################################################################################################
    fig,axes = plt.subplots(figsize=(20,20),nrows=2,ncols=2)
    ax = axes[0][0] # train-test in load 2
    im = ax.imshow(scores_within_load2.mean(0),origin='lower',aspect='auto',extent=[0,6000,0,6000],vmin=.5,vmax=.65,cmap=plt.cm.RdBu_r)
    ax.set(ylabel='load2\n\n\ntrain time (ms)',title='Load 2')
    
    ax = axes[0][1] # train in load 2 and test in load 5
    im = ax.imshow(scores_cross_load5.mean(0),origin='lower',aspect='auto',extent=[0,6000,0,6000],vmin=.5,vmax=.65,cmap=plt.cm.RdBu_r)
    ax.set(title='load 5')
    
    ax = axes[1][0] # train in load 5 and test in load 2
    im = ax.imshow(scores_cross_load2.mean(0),origin='lower',aspect='auto',extent=[0,6000,0,6000],vmin=.5,vmax=.65,cmap=plt.cm.RdBu_r)
    ax.set(ylabel='load5\n\n\ntrain time (ms)',xlabel='test time (ms)',)
    
    ax = axes[1][1]# train in load 5 and test in load 5
    im = ax.imshow(scores_within_load5.mean(0),origin='lower',aspect='auto',extent=[0,6000,0,6000],vmin=.5,vmax=.65,cmap=plt.cm.RdBu_r)
    ax.set(xlabel='test time (ms)')
    
    fig.subplots_adjust(bottom=0.1, top=0.96, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with 
    # axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    fig.suptitle('Cross Condition Temporal Generalization Decoding\nCorrect VS. Incorrect')
    
    
    # temporal decoding of load 2
    # temporal decoding of load 5
    time_dec = SlidingEstimator(clf,scoring='roc_auc')
    sc2 = cross_val_multiscore(time_dec,X_load2,labels_load2,cv=cv,n_jobs=4)
    time_dec = SlidingEstimator(clf,scoring='roc_auc')
    sc5 = cross_val_multiscore(time_dec,X_load5,labels_load5,cv=cv,n_jobs=4)
    
    fig, axes = plt.subplots(figsize=(20,12),nrows=2)
    ax = axes[0]
    ax.plot(np.linspace(0,6000,sc2.shape[1]),sc2.mean(0),color='k',alpha=1.,label='Decoding Scores')
    ax.fill_between(np.linspace(0,6000,sc2.shape[1]),
                                sc2.mean(0)-sc2.std(0)/np.sqrt(10),
                                sc2.mean(0)+sc2.std(0)/np.sqrt(10),
                                color='red',alpha=.5,label='Standard Error')
    ax.legend(loc='best')
    ax.axhline(0.5,linestyle='--',color='blue',alpha=.7,label='Chance Level')
    ax.set(xlabel='Time (ms)',ylabel='Classifi.Score (ROC AUC)',title='Temporal Decoding [load 2]',xlim=(0,6000))

    ax = axes[1]
    ax.plot(np.linspace(0,6000,sc5.shape[1]),sc5.mean(0),color='k',alpha=1.,label='Decoding Scores')
    ax.fill_between(np.linspace(0,6000,sc5.shape[1]),
                                sc5.mean(0)-sc5.std(0)/np.sqrt(10),
                                sc5.mean(0)+sc5.std(0)/np.sqrt(10),
                                color='red',alpha=.5,label='Standard Error')
    ax.legend(loc='best')
    ax.axhline(0.5,linestyle='--',color='blue',alpha=.7,label='Chance Level')
    ax.set(xlabel='Time (ms)',ylabel='Classifi.Score (ROC AUC)',title='Temporal Decoding [load 5]',xlim=(0,6000))

    
    # patterns in load 2
    patterns_2 = []
    for train, test in tqdm(cv.split(X_load2,labels_load2),desc='load2'):
        X = X_load2[train]
        y = labels_load2[train]
        clf = make_pipeline(vec,sm,LinearModel(est))
        clfs = [make_pipeline(vec,sm,LinearModel(est)).fit(X[:,:,ii],y) for ii in range(X.shape[-1])]
        patterns_ = [get_coef(clfs[ii],attr='patterns_',inverse_transform=True) for ii in range(X.shape[-1])]
        patterns_2.append(np.array(patterns_))
    # patterns in load 5
    patterns_5 = []
    for train, test in tqdm(cv.split(X_load5,labels_load5),desc='load5'):
        X = X_load5[train]
        y = labels_load5[train]
        clf = make_pipeline(vec,sm,LinearModel(est))
        clfs = [make_pipeline(vec,sm,LinearModel(est)).fit(X[:,:,ii],y) for ii in range(X.shape[-1])]
        patterns_ = [get_coef(clfs[ii],attr='patterns_',inverse_transform=True) for ii in range(X.shape[-1])]
        patterns_5.append(np.array(patterns_))
    
    temp_ = mne.read_epochs('D:\\working_memory\\encode_delay_prode_RSA_preprocessing\\sub_11_load2_day2_encode_delay_probe-epo.fif',
                            preload=False)
    info = temp_.info
    
    evoked_2 = mne.EvokedArray(patterns_2.mean(0),info=info)
    evoked_5 = mne.EvokedArray(patterns_5.mean(0),info=info)










































