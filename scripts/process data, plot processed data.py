# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:11:25 2018

@author: ning
"""

import mne
import os
import matplotlib.pyplot as plt
import re
from glob import glob

working_dir = 'D:\\working_memory\\data_probe_train_test\\'

"""probe"""
files = glob(os.path.join(working_dir,'probe/*.fif'))
for e in files:
    epochs = mne.read_epochs(e)
    sub,load,day = re.findall('\d+',e)
    title = 'sub%s, day%s,load2'%(sub,day)
    fig = epochs.average().plot_joint(title=title,show=False,)
    fig.savefig(working_dir+'probe\\'+title+'.png')

"""delay"""
files = glob(os.path.join(working_dir,'delay/*.fif'))
for e in files:
    epochs = mne.read_epochs(e)
    sub,load,day = re.findall('\d+',e)
    title = 'sub%s, day%s,load2'%(sub,day)
    fig = epochs.average().plot_joint(title=title,show=False,)
    fig.savefig(working_dir+'delay\\'+title+'.png')