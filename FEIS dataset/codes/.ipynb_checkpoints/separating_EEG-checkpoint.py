# -*- coding: utf-8 -*-
"""
@author: Vinay Faria
@institute: IIT Mandi (M.Tech CSP)
"""

import pandas as pd

path = 'C:/Users/vinay/Downloads/FEIS_v1_1/experiments/21/'
data = pd.read_csv(path + 'full_eeg.csv', skiprows=0, usecols=[*range(0, 18)])
#data = pd.read_csv('full_eeg.csv', skiprows=0, usecols=[*range(0, 18)])

data = data.reset_index()  # make sure indexes pair with number of rows

ch_names = ['F3', 'FC5', 'AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'F8', 'AF4', 'FC6', 'F4'] # channel names
phonemes_count = {'articulators_f':0, 'articulators_fleece':0, 'articulators_goose':0, 'articulators_k':0, 'articulators_m':0, 'articulators_n':0, 'articulators_ng':0, 
                  'articulators_p':0, 'articulators_s':0, 'articulators_sh':0, 'articulators_t':0, 'articulators_thought':0, 'articulators_trap':0, 'articulators_v':0, 
                  'articulators_z':0, 'articulators_zh':0, 'resting_f':0, 'resting_fleece':0, 'resting_goose':0, 'resting_k':0, 'resting_m':0, 'resting_n':0, 'resting_ng':0, 
                  'resting_p':0, 'resting_s':0, 'resting_sh':0, 'resting_t':0, 'resting_thought':0, 'resting_trap':0, 'resting_v':0, 
                  'resting_z':0, 'resting_zh':0, 'speaking_f':0, 'speaking_fleece':0, 'speaking_goose':0, 'speaking_k':0, 'speaking_m':0, 'speaking_n':0, 'speaking_ng':0, 
                  'speaking_p':0, 'speaking_s':0, 'speaking_sh':0, 'speaking_t':0, 'speaking_thought':0, 'speaking_trap':0, 'speaking_v':0, 
                  'speaking_z':0, 'speaking_zh':0, 'stimuli_f':0, 'stimuli_fleece':0, 'stimuli_goose':0, 'stimuli_k':0, 'stimuli_m':0, 'stimuli_n':0, 'stimuli_ng':0, 
                  'stimuli_p':0, 'stimuli_s':0, 'stimuli_sh':0, 'stimuli_t':0, 'stimuli_thought':0, 'stimuli_trap':0, 'stimuli_v':0, 
                  'stimuli_z':0, 'stimuli_zh':0, 'thinking_f':0, 'thinking_fleece':0, 'thinking_goose':0, 'thinking_k':0, 'thinking_m':0, 'thinking_n':0, 'thinking_ng':0, 
                  'thinking_p':0, 'thinking_s':0, 'thinking_sh':0, 'thinking_t':0, 'thinking_thought':0, 'thinking_trap':0, 'thinking_v':0, 
                  'thinking_z':0, 'thinking_zh':0, }
previous_label = ""
count = 0
final_dataframe = pd.DataFrame(columns=ch_names)
columns = list(final_dataframe)
rows = []
for ind in data.index:
    current_label = data['Label'][ind]
    values = [data['F3'][ind], data['FC5'][ind], data['AF3'][ind], data['F7'][ind], data['T7'][ind], data['P7'][ind], data['O1'][ind], data['O2'][ind], data['P8'][ind], data['T8'][ind], data['F8'][ind], data['AF4'][ind], data['FC6'][ind], data['F4'][ind]]
    zipped = zip(columns, values)
    a_dictionary = dict(zipped)
    #print(a_dictionary)
    rows.append(a_dictionary)
    if current_label == previous_label or count == 0:
        count += 1
    if count == 1280:
        dummy_1 = data['Stage'][ind] + '_' + data['Label'][ind]
        phonemes_count[dummy_1] += 1
        name_of_file = dummy_1 + "_trial_" + str(phonemes_count[dummy_1])
        final_dataframe = final_dataframe.append(rows, True)
        # saving the dataframe
        #final_dataframe.to_csv(name_of_file + '.csv', index=False)
        final_dataframe.to_csv('C:/Users/vinay/Downloads/FEIS_v1_1/experiments/21/' + data['Stage'][ind] + '_eeg/' + data['Label'][ind] + '/' + name_of_file + '.csv', index=False)
        #final_dataframe.to_csv(name_of_file + '.csv', index=False)
        final_dataframe.drop(final_dataframe.index,inplace=True) 
        count = 0
        previous_label = ""
        rows = []
    if count == 256 and data['Stage'][ind] == "articulators":
        dummy_1 = data['Stage'][ind] + '_' + data['Label'][ind]
        phonemes_count[dummy_1] += 1
        name_of_file = dummy_1 + "_trial_" + str(phonemes_count[dummy_1])
        final_dataframe = final_dataframe.append(rows, True)
        # saving the dataframe
        final_dataframe.to_csv('C:/Users/vinay/Downloads/FEIS_v1_1/experiments/21/' + data['Stage'][ind] + '_eeg/' + data['Label'][ind] + '/' + name_of_file + '.csv', index=False)
        #final_dataframe.to_csv(name_of_file + '.csv', index=False)
        final_dataframe.drop(final_dataframe.index,inplace=True)
        count = 0
        previous_label = ""
        rows = []
    
    previous_label = current_label