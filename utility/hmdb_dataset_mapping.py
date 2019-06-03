# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:35:56 2019

@author: Juen

RUN THIS SCRIPT ONLY WHEN YOU DO NOT HAVE THE MODIFIED HMDB MAPPING FILE
"""

import os
from pathlib import Path

hmdb_directory = Path(r'..\dataset\HMDB-51')

# generate labels to index mapping
hmdb_labels = {}
hmdb_labels_str = os.listdir(hmdb_directory/'hmdb51_org')

hmdb_labels = {label : index for index, label in enumerate(hmdb_labels_str)}

# generate videos to labels mapping
hmdb_video_label_map = {}

for i in range(len(hmdb_labels_str)):
    buffer_video_str = os.listdir(hmdb_directory/'hmdb51_org'/hmdb_labels_str[i])
    buffer_map = { video : hmdb_labels[hmdb_labels_str[i]] for video in buffer_video_str}
    hmdb_video_label_map.update(buffer_map)
    
# generate training and testing list based on split
hmdb_train_txt_str = ['', '', '']
hmdb_test_txt_str = ['', '', '']
hmdb_test_files = os.listdir(hmdb_directory/'testTrainMulti_7030_splits')

# categorize test list based on the split
assert(len(hmdb_test_files) == len(hmdb_labels) * 3)
for i in range(len(hmdb_test_files)):
    # retrieve the current processing spilt and labelled samples
    split = i % 3
    label = i // 3
    
    # read the content from current txt file
    fo_hmdb_txt = open(hmdb_directory/'testTrainMulti_7030_splits'/hmdb_test_files[i], 'r')
    buffer_str = fo_hmdb_txt.read().split('\n')
    
    # process the current txt file to map with label
    # video_label ## 0 -> not included; 1 -> training set; 2 -> testing set ##
    for j in range(len(buffer_str) - 1):
        buffer_entry = buffer_str[j].split(' ')
        buffer_video_label = int(buffer_entry[1])
        assert(buffer_video_label >= 0 and buffer_video_label <= 2)
        if buffer_video_label == 0:
            pass
        elif buffer_video_label == 1:
            hmdb_train_txt_str[split] += (buffer_entry[0] + ' ' + str(label) + '\n')
        elif buffer_video_label == 2:
            hmdb_test_txt_str[split] += (buffer_entry[0] + ' ' + str(label) + '\n')
            
    fo_hmdb_txt.close()
            
# output the video -- label mapping into txt files
for i in range(3):
    train_out_file_path = 'hmdb_trainlist0' + str (i + 1) + '.txt'
    test_out_file_path = 'hmdb_testlist0' + str (i + 1) + '.txt'
    
    fo_train_hmdb = open(hmdb_directory/train_out_file_path, 'w')
    fo_test_hmdb = open(hmdb_directory/test_out_file_path, 'w')
    
    fo_train_hmdb.write(hmdb_train_txt_str[i])
    fo_test_hmdb.write(hmdb_test_txt_str[i])
    
    fo_train_hmdb.close()
    fo_test_hmdb.close()
    
    