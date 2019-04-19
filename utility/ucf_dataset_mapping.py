# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:42:46 2019

@author: Juen

RUN THIS SCRIPT ONLY WHEN YOU DO NOT HAVE THE MODIFIED UCF MAPPING FILE
"""

from pathlib import Path

ucf_directory = Path(r'..\..\dataset\UCF-101\ucfTrainValTestlist')

# generate labels to index mapping
ucf_labels = {}
fo_ucf_ind = open(ucf_directory/'classInd.txt', 'r')
ucf_ind_str = fo_ucf_ind.read().split('\n')

for index in range(len(ucf_ind_str) - 1):
    ucf_labels[ucf_ind_str[index].split(' ')[1]] = index
    
fo_ucf_ind.close()
mode = ['train', 'validation', 'test']

# reading train and test list video name and map them with labels
for i in range(3):
    
    for m in mode:
        
        list_str = ''
        
        # reading list of current training split
        in_file_path = (m + 'list0' + str(i + 1) + '.txt')
        out_file_path = ('ucf_' + m + 'list0'+ str(i + 1) +'.txt')
        
        fo_ucf = open(ucf_directory/in_file_path, 'r')
        fo_ucf_list = open(ucf_directory/out_file_path, 'w')
        ucf_str = fo_ucf.read().split('\n')
        
        # saving train_list into a file, based on the split
        # format ## trimmed video_name label_index ##
        
        for index in range(len(ucf_str) - 1):
            buffer = ucf_str[index].split(' ')
            video = buffer[0].split('/')[1]
            
            # special case where original entry was incorrect
            video = video.replace('HandStandPushups', 'HandstandPushups')
            
            label = video.split('_')[1]
            
            if m is not 'test':
                list_str += (video + ' ' + str(int(buffer[1]) - 1) + '\n')
            else:
                
                list_str += (video + ' ' + str(ucf_labels[label]) + '\n')
            
        fo_ucf_list.write(list_str)
        fo_ucf_list.close()
        fo_ucf.close()
        