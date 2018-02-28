# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:14:39 2018

@author: ning
"""
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from mne.decoding import Vectorizer
from sklearn import metrics
import pandas as pd
import numpy as np

def make_clf(vectorized = True,hard_soft='soft',voting=True):
    """
    vectorized: to wrap the 3D matrix to 2D
    hard_soft: decision making step for voting classifier
    voting: if true, classifiers are SGD, SVM, KNN, naive bayes, dense neural networkd, random forest
            and gradient boosting classifier
            
    to make sure the voting classifier returns probabilistic prediction, we need to carefully define
    each of the individual 'voters'
            if false, classifier are SDG, SVM, random forest and gradient boosting classifier
    
    All nested with a standardized scaler - mean centered and unit variance
    """
    linear_ = SGDClassifier(max_iter=int(2e3),tol=1e-3,random_state=12345,loss='modified_huber')
    svc = SVC(max_iter=int(2e3),tol=1e-3,random_state=12345,kernel='rbf',probability=True,)
    rf = RandomForestClassifier(n_estimators=100,random_state=12345)
    knn = KNeighborsClassifier(n_neighbors=10,)
    bayes = GaussianNB(priors=(0.4,0.6))
    gdb = GradientBoostingClassifier(random_state=12345)
    NN = MLPClassifier(hidden_layer_sizes=(100,50,20),learning_rate='adaptive',solver='sgd',max_iter=int(1e3),
                       shuffle=True,random_state=12345)
    
    clf = []
    if vectorized:
        clf.append(('vectorize',Vectorizer()))
    clf.append(('scaler',StandardScaler()))
    if voting:
        clf.append(('estimator',VotingClassifier([('SGD',linear_),
                                                  ('SVM',svc),
                                                  ('RF',rf),
                                                  ('KNN',knn),
                                                  ('naive_bayes',bayes),
                                                  ('DNN',NN),
                                                  ('GDB',gdb)],voting=hard_soft,)))
    else:
        clf.append(('estimator',VotingClassifier([('SGD',linear_),
                                                  ('SVM',svc),
                                                  ('RF',rf),
                                                  ('GDB',gdb)],voting=hard_soft,)))
    clf = Pipeline(clf)
    return clf

def row_selection(row_element,idx_pos):
    """
    small helper function for the next function
    """
    if row_element in idx_pos:
        return True
    else:
        return False
def prediction_pipeline(labels,images,clf,working_trial_orders,condition='load2'):
    """
    This function is to process predicted labels, predicted prbabilities, and true labels in 
    different experimental conditions.
    """
    
    if condition == 'load2':# just in case I figure out how to do this in load 5 condition
        image1,image2 = images
        positive,negative = [],[] # preallocate for the predictions within the positive probe trials and negative probe trials
        idx_pos,idx_neg = [],[]# preallocate for the trial numbers
        for ii,(label, image1_,image2_) in enumerate(zip(labels,image1,image2)):
        #    print(label,image1_.shape,image2_.shape)
            if label:# "1" in label can be used as "true"
                positive.append([clf.predict_proba(image1_.reshape(1,61,200))[0],# single item probabilistic prediction returns a 3D vector, thus, we take the 1st dimension out
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
        positive_prob = positive[:,:,-1]# we only care about the probability of the positive probe images (old images)
        negative_prob = negative[:,:,-1]
        
        soft_max_ = np.concatenate((positive_soft_max,negative_soft_max))# I called this "soft max", but it is not doing such thing
        prob_ = np.concatenate((positive_prob,negative_prob))
        soft_max_idx = np.concatenate((idx_pos,idx_neg))
        # create a data frame with two columns, and each column contains the probability of weather this image will present again in the probe, regardless 
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
    elif condition == 'load5':# to be continue
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
        




































































