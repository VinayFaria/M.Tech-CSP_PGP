# -*- coding: utf-8 -*-
"""
@author: vinay
"""

import pandas as pd 
import mne

path = 'C:/Users/vinay/Downloads/FEIS_v1_1/experiments/01/'
data = pd.read_csv(path + 'full_eeg.csv', skiprows=0, usecols=[*range(0, 18)])

ch_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'] # channel names
pre_goose_articulator = data[data["Label"].str.contains("goose")]

goose_articulator = pre_goose_articulator.loc[pre_goose_articulator["Time:256Hz"].between(21, 26)]
#time_goose_articulator = pre_goose_articulator['Time:256Hz']
goose_articulator = goose_articulator[ch_names]

sfreq = 256 # sampling frequency, in hertz
info = mne.create_info(ch_names = ch_names, sfreq = sfreq) # See [docs](http://martinos.org/mne/stable/generated/mne.Info.html) for full list of Info options. 

goose_articulator = goose_articulator.transpose()

raw = mne.io.RawArray(goose_articulator, info)
raw.plot()