"""
Objective: To merge all participant EEG signal into a zipped archive of files
@author: vinay
References: https://stackoverflow.com/questions/19587118/iterating-through-directories-with-python
"""

import os
import numpy as np
import pandas as pd

stage = 'stimuli'
phonemes = ['f', 'fleece', 'goose', 'k', 'm', 'n', 'ng', 'p', 's', 'sh', 't', 'thought', 'trap', 'v', 'z', 'zh']
for i in range(len(phonemes)):
    phonemes[i] = stage + '_' + phonemes[i]

data, labels = [], []
#X_train, y_train = np.array([]),np.array([])
X_train, y_train = [],[]
X_test, y_test = np.array([]),np.array([])
rootdir = 'C:/Users/vinay/Downloads/FEIS_v1_1/experiments/'
dummy_count = []
for subdir, dirs, files in os.walk(rootdir):
    if subdir == 'C:/Users/vinay/Downloads/FEIS_v1_1/experiments/chinese-1':
        break
    count = 0
    for file in files:
        
        file_name_list = file.split('_')
        for phoneme in phonemes:
            phoneme_name_list = phoneme.split('_')
            if file_name_list[0] + '_' + file_name_list[1] == phoneme_name_list[0] + '_' + phoneme_name_list[1]:
                path = subdir + '/' + file
                df = pd.read_csv(path).T
                dummy = df.values
                data.append(dummy)
                labels.append(phoneme)
                count += 1
                #print('This is subdir',subdir)
                #print('This is dirs',dirs)
                #print('This is file',file)
                
                #X_train = np.append(X_train, )
    dummy_count.append(count)
        #print(file)
        #print(os.path.join(subdir, file))

np.savez("C:/Users/vinay/Downloads/FEIS_v1_1/experiments/combined_data.npz", data=data, labels=labels) #saving my two test arrays