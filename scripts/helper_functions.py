# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:14:39 2018

@author: ning
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.pipeline import Pipeline
from mne.decoding import Vectorizer
from sklearn import metrics
import pandas as pd
import numpy as np

def make_clf(vectorized = True,hard_soft='soft',voting=True):
    linear_ = SGDClassifier(max_iter=int(2e3),tol=1e-3,random_state=12345,loss='modified_huber')
    svc = SVC(max_iter=int(5e2),tol=1e-3,random_state=12345,kernel='rbf',probability=True,)
    rf = RandomForestClassifier(n_estimators=50,random_state=12345)
    
    clf = []
    if vectorized:
        clf.append(('vectorize',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    if voting:
        clf.append(('estimator',VotingClassifier([('SGD',linear_),('SVM',svc),('RF',rf)],voting=hard_soft,)))
    else:
        clf.append(('estimator',rf))
    clf = Pipeline(clf)
    return clf

def row_selection(row_element,idx_pos):
    if row_element in idx_pos:
        return True
    else:
        return False
def prediction_pipeline(labels,images,clf,working_trial_orders,condition='load2'):
    if condition == 'load2':
        image1,image2 = images
        positive,negative = [],[]
        idx_pos,idx_neg = [],[]
        for ii,(label, image1_,image2_) in enumerate(zip(labels,image1,image2)):
        #    print(label,image1_.shape,image2_.shape)
            if label:
                positive.append([clf.predict_proba(image1_.reshape(1,61,200))[0],
                                 clf.predict_proba(image2_.reshape(1,61,200))[0]])
                idx_pos.append(ii)
            else:
                negative.append([clf.predict_proba(image1_.reshape(1,61,200))[0],
                                 clf.predict_proba(image2_.reshape(1,61,200))[0]])
                idx_neg.append(ii)
        
        positive = np.array(positive)
        negative = np.array(negative)
        positive_soft_max = np.argmax(positive,axis=1)
        negative_soft_max = np.argmax(negative,axis=1)
        positive_prob = positive[:,:,-1]
        negative_prob = negative[:,:,-1]
        
        soft_max_ = np.concatenate((positive_soft_max,negative_soft_max))
        prob_ = np.concatenate((positive_prob,negative_prob))
        soft_max_idx = np.concatenate((idx_pos,idx_neg))
        results = pd.DataFrame(soft_max_,columns=['image1_pred','image2_pred'])
        results['order'] = soft_max_idx
        results['image1_pred_prob'] = prob_[:,0]
        results['image2_pred_prob'] = prob_[:,1]
        results = results.sort_values('order').reset_index()
        results['labels'] = labels
        results['image1']=np.array(working_trial_orders['probe'] == working_trial_orders['image1'],dtype=int)
        results['image2']=np.array(working_trial_orders['probe'] == working_trial_orders['image2'],dtype=int)
    #    pred = results[['image1_pred','image2_pred']].values
        pred_prob = results[['image1_pred_prob','image2_pred_prob']].values
        truth = results[['image1','image2']].values
    #    print(metrics.classification_report(truth,pred))
        
        # predictive ability of order
        print('predictive ability of order',metrics.roc_auc_score(truth[:,0],pred_prob[:,0]),metrics.roc_auc_score(truth[:,1],pred_prob[:,1]))
        # predictive ability of positive and negative stimuli
        results['trial']=[row_selection(row_element,idx_pos) for row_element in results['order'].values]
        positive_trials = results.iloc[results['trial'].values]
        truth_ = positive_trials[['image1','image2']].values.flatten()
        pred_prob_ = positive_trials[['image1_pred_prob','image2_pred_prob']].values.flatten()
        print('predictive ability of positive and negative stimuli',metrics.roc_auc_score(truth_,pred_prob_))
        return {'predictive ability of order':[metrics.roc_auc_score(truth[:,0],pred_prob[:,0]),
                                               metrics.roc_auc_score(truth[:,1],pred_prob[:,1])],
                'predictive ability of positive and negative stimuli':metrics.roc_auc_score(truth_,pred_prob_)}
    elif condition == 'load5':
        iamge1,image2,image3,image4,image5 = images
        positive,negative = [],[]
        idx_pos,idx_neg = [],[]
        for ii, (label,image1_,image2_,image3_,image4_,image5_) in enumerate(zip(labels,image1,image2,image3,image4,image5)):
            if label:
                positive.append([clf.predict_proba(image1_.reshape(1,61,200))[0],
                                 clf.predict_proba(image2_.reshape(1,61,200))[0],
                                 clf.predict_proba(image3_.reshape(1,61,200))[0],
                                 clf.predict_proba(image4_.reshape(1,61,200))[0],
                                 clf.predict_proba(image5_.reshape(1,61,200))[0]])
                idx_pos.append(ii)
            else:
                negative.append([clf.predict_proba(image1_.reshape(1,61,200))[0],
                                 clf.predict_proba(image2_.reshape(1,61,200))[0],
                                 clf.predict_proba(image3_.reshape(1,61,200))[0],
                                 clf.predict_proba(image4_.reshape(1,61,200))[0],
                                 clf.predict_proba(image5_.reshape(1,61,200))[0]])
                idx_neg.append(ii)

        positive = np.array(positive)
        negative = np.array(negative)
        positive_soft_max = np.argmax(positive,axis=1)
        negative_soft_max = np.argmax(negative,axis=1)
        positive_prob = positive[:,:,-1]
        negative_prob = negative[:,:,-1]
        
        soft_max_ = np.concatenate((positive_soft_max,negative_soft_max))
        prob_ = np.concatenate((positive_prob,negative_prob))
        soft_max_idx = np.concatenate((idx_pos,idx_neg))
        




































































