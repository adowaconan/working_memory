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
condition = 'probe'
files = glob(os.path.join(working_dir,'%s/*.fif'%condition))
for e in files:
    epochs = mne.read_epochs(e)
    sub,load,day = re.findall('\d+',e)
    title = 'sub%s, day%s,load2\npositive probe'%(sub,day)
    fig = epochs['target probe'].average().plot_joint(title=title,show=False,)# only plot the activity of the target probe trials
    fig.savefig(working_dir+'%s\\'%condition+title.replace('\n',',')+'.png')

"""delay"""
condition = 'delay'
files = glob(os.path.join(working_dir,'%s/*.fif'%condition))
for e in files:
    epochs = mne.read_epochs(e)
    sub,load,day = re.findall('\d+',e)
    title = 'sub%s, day%s,load2'%(sub,day)
    fig = epochs.average().plot_joint(title=title,show=False,)#during delay, subjects don't know if probe target or non target
    fig.savefig(working_dir+'%s\\'%condition+title.replace('\n',',')+'.png')

condition = 'encode'
files = glob(os.path.join(working_dir,'%s/*.fif'%condition))
for e in files:
    epochs = mne.read_epochs(e)
    sub,load,day = re.findall('\d+',e)
    title = 'sub%s, day%s,load2\npresent again in probe'%(sub,day)
    fig = epochs['target probe'].average().plot_joint(title=title,show=False,)
    fig.savefig(working_dir+'%s\\'%condition+title.replace('\n',',')+'.png')