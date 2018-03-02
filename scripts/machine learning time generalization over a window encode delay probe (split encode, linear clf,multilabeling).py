# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:56:54 2018

@author: ning
python "D:/working_memory/working_memory/scripts/machine learning time generalization over a window encode delay probe (split encode, linear clf,multilabeling).py"
"""
if __name__ == '__main__':#  the way to force parellel processing
    import os
    os.chdir('D:/working_memory/working_memory/scripts')
#    from helper_functions import make_clf#,prediction_pipeline
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
    saving_dir = 'D:/working_memory/working_memory/results/train_probe_test_encode/full window docoding multilabeling/'
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
    
    for e, e_ in zip(epoch_files,event_files):
    #    e = epoch_files[8] # debugging stuff
    #    e_= event_files[8] # debugging stuff
        
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
        # get data of delay and probe as continued data
        delay_probe_ = epochs.copy().crop(-6,2).get_data()[:,:,:800]
        delay_probe = np.concatenate((delay_probe_,delay_probe_))
        
        # get data in encoding
        image1 = epochs.copy().crop(-10,-8).get_data()[:,:,:200]
        image2 = epochs.copy().crop(-8,-6).get_data()[:,:,:200]
        images = np.concatenate((image1,image2))
        labels1=np.vstack((np.array(working_trial_orders['probe'] == working_trial_orders['image1'],dtype=int),labels)).T
        labels2=np.vstack((np.array(working_trial_orders['probe'] == working_trial_orders['image2'],dtype=int),labels)).T
        
        encode_delay_probe = np.concatenate((images,delay_probe),axis=2)
        labels = np.concatenate((labels1,labels2))
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
#        clf = make_clf(voting='linear',)# use less computational expensive linear classifiers
#        patterns = []
        scores = []
        # time generalization within probe and cross modal
        clf = []
        from mne.decoding import Vectorizer#,LinearModel
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC
        clf.append(('vectorize',Vectorizer()))
        clf.append(('scaler',StandardScaler()))
        from sklearn.multioutput import MultiOutputClassifier
        from sklearn.metrics import label_ranking_loss,make_scorer
        est = SVC(max_iter=int(1e5),tol=1e-3,random_state=12345,kernel='linear',class_weight='balanced',probability=True,)
        clf.append(('estimator',MultiOutputClassifier(est)))# don't have off-the-shelft decoding method
        clf = Pipeline(clf)
        for train,test in tqdm(cv.split(encode_delay_probe,labels[:,0])):
            X = encode_delay_probe[train]
            y = labels[train]
            time_gen = GeneralizingEstimator(clf,scoring=make_scorer(label_ranking_loss),n_jobs=4)# define the time generalization model
            time_gen.fit(X,y)    
            scores_=time_gen.score(encode_delay_probe[test],labels[test])
            scores.append(scores_)
            # get the learned patterns within the classifier, which is positive probe (old image) - negative probe (new image)
#            temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in time_gen.estimators_])
#            patterns.append(temp_patterns)
            
        scores = np.array(scores)
#        patterns = np.array(patterns)
#        info = epochs.info
#        evoked = mne.EvokedArray(-patterns.mean(0).T,info)
#        evoked.save(saving_dir+'split_encode_linear_time_generalization_sub%sload%sday%s-evo.fif'%(sub,load,day))
        pickle.dump([scores],open(saving_dir+'time_general_en_de_pr_%s_%s_%s.p'%(sub,load,day),'wb'))
        # contour plot
        fig,ax = plt.subplots(figsize=(10,10))
        CS = ax.contour(scores.mean(0),extent=[-8000,2000,-8000,2000],cmap=plt.cm.coolwarm,vmin=0.,vmax=.3)
        norm= colors.Normalize(vmin=0., vmax=.3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap = CS.cmap)
        sm.set_array([])
        driver = make_axes_locatable(ax)
        cax = driver.append_axes('right',size='5%',pad=0.05)
        fig.colorbar(sm,cax=cax, ticks=CS.levels)
        ax.set(xlabel='test time',ylabel='train time',title='time generalization\nsub_%s,load_%s,day_%s'%(sub,load,day))
        ax.axvline(0,color='k',linestyle='--');ax.axhline(0,color='k',linestyle='--')
        ax.axvline(-6000,color='k',linestyle='--');ax.axhline(-6000,color='k',linestyle='--')
        fig.savefig(saving_dir+'time generalization_contour_sub_%s,load_%s,day_%s.png'%(sub,load,day),dpi=300)
        
        # imshow plot
        fig,ax = plt.subplots(figsize=(12,10))
        im = ax.imshow(scores.mean(0),origin='lower',aspect='auto',extent=[-8000,2000,-8000,2000],cmap=plt.cm.coolwarm,)#vmin=0,vmax=.4)
        fig.colorbar(im)
        ax.set(xlabel='test time',ylabel='train time',title='time generalization\nsub_%s,load_%s,day_%s'%(sub,load,day))
        ax.axvline(0,color='k',linestyle='--');ax.axhline(0,color='k',linestyle='--')
        ax.axvline(-6000,color='k',linestyle='--');ax.axhline(-6000,color='k',linestyle='--')
        fig.savefig(saving_dir+'time generalization_implot_sub_%s,load_%s,day_%s.png'%(sub,load,day),dpi=300)
        
        # diagonal decoding result
        fig,ax = plt.subplots(figsize=(12,6))
        times = np.linspace(-8000,2000,scores.shape[-1])
        decoding_mean = scores.mean(0).diagonal()
        decoding_std = scores.std(0).diagonal()
        kernel_size = 20
        decoding_mean_smooth = np.convolve(decoding_mean,[1/kernel_size for ii in range(kernel_size)],'same')
        decoding_std_smooth = np.convolve(decoding_std,[1/kernel_size for ii in range(kernel_size)],'same')
        ax.plot(times,decoding_mean_smooth,color='k',alpha=1.,label='Decoding scores')
        ax.fill_between(times,
                        decoding_mean_smooth+decoding_std_smooth,
                        decoding_mean_smooth-decoding_std_smooth,
                        color='red',alpha=.5)
    #    ax.axhline(0,color='blue',linestyle='--',alpha=.5,label='chance level')
        ax.axvline(0,color='green',linestyle='--',alpha=.5,label='Probe onset');ax.axvline(-6000,color='k',linestyle='--',alpha=.5,label='Delay onset')
        ax.set(xlim=(-8000,2000),xlabel='Time (ms)',ylabel='Classifi.score (label ranking loss)',
               title='Temporal Decoding\nsub_%s,load_%s,day_%s'%(sub,load,day))
        ax.legend(loc='best')
        fig.savefig(saving_dir+'temporal_decoding_sub_%s,load_%s,day_%s.png'%(sub,load,day),dpi=300)
        plt.close('all')



"""
from sklearn.metrics import label_ranking_loss
a= np.array([[0,0],[1,0],[1,1],[0,1]])
idx = np.random.choice(4,100)
l = a[idx]
p = a[np.random.choice(4,100)]
label_ranking_loss(l,p)


erererwe = np.zeros((1000,1000))
for ii in tqdm(range(encode_delay_probe.shape[-1])):
    for jj in range(encode_delay_probe.shape[-1]):
        temp =np.array(time_gen.estimators_[ii].predict_proba(encode_delay_probe[test,:,jj]))[:,:,-1].T
        erererwe[ii,jj] = label_ranking_loss(labels[test],temp)


pppp = np.array(time_gen.estimators_[ii].predict_proba(encode_delay_probe[test,:,jj]))


from joblib import Parallel, delayed
def myfun(jj):
    global ii
    temp=np.array(time_gen.estimators_[ii].predict_proba(encode_delay_probe[test,:,jj]))[:,:,-1].T
    result=label_ranking_loss(labels[test],temp)
    return result

def myfun2(ii):
    result = Parallel(n_jobs=-1, verbose=1, backend="multiprocessing")(
                 map(delayed(myfun), range(1000)))

results = Parallel(n_jobs=-1,verbose=1, backend="multiprocessing")(
                 map(delayed(myfun2), range(1000)))


results = np.zeros((1000,1000))
for ii in tqdm(range(1000)):
    results[ii] = Parallel(n_jobs=4, verbose=1, backend="threading")(
                 map(delayed(myfun), range(1000)))


import itertools
import multiprocessing
ii = range(1000)
jj = range(1000)
paramlist = list(itertools.product(ii,jj))

def myfun_(ii,jj):
    temp =np.array(time_gen.estimators_[ii].predict_proba(encode_delay_probe[test,:,jj]))[:,:,-1].T
    return label_ranking_loss(labels[test],temp)

res = Parallel(n_jobs=4,verbose=1,backend='multiprocessing')(
        map(delayed(myfun_),itertools.product(range(1000),range(1000))))

for ii,jj in tqdm(itertools.product(range(1000),range(1000))):
#    print(ii,jj)
    res = myfun_(ii,jj)
"""
















