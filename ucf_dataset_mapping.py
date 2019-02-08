# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:42:46 2019

@author: Juen
"""

from pathlib import Path

ucf_directory = Path(r'..\dataset\UCF-101\ucfTrainTestlist')

# generate labels to index mapping
ucf_labels = {}
fo_ucf_ind = open(ucf_directory/'classInd.txt', 'r')
ucf_ind_str = fo_ucf_ind.read().split('\n')

for index in range(len(ucf_ind_str) - 1):
    ucf_labels[ucf_ind_str[index].split(' ')[1]] = index
    
fo_ucf_ind.close()

# reading train and test list video name and map them with labels
for i in range(3):
    
    train_list_str = ''
    test_list_str = ''
    
    # reading list of current training split
    in_file_path = 'trainlist0'+ str(i + 1) +'.txt'
    out_file_path = 'ucf_trainlist0'+ str(i + 1) +'.txt'
    
    fo_ucf_train = open(ucf_directory/in_file_path, 'r')
    fo_ucf_train_list = open(ucf_directory/out_file_path, 'w')
    ucf_train_str = fo_ucf_train.read().split('\n')
    
    # saving train_list into a file, based on the split
    # format ## trimmed video_name label_index ##
    
    for index in range(len(ucf_train_str) - 1):
        buffer = ucf_train_str[index].split(' ')
        train_list_str += (buffer[0].split('/')[1] + ' ' + str(int(buffer[1]) - 1) + '\n')
    
    fo_ucf_train_list.write(train_list_str)
    fo_ucf_train_list.close()
    fo_ucf_train.close()
    
    # reading list of current testing split
    in_file_path = 'testlist0'+ str(i + 1) + '.txt'
    out_file_path = 'ucf_testlist0'+ str(i + 1) +'.txt'
    
    fo_ucf_test = open(ucf_directory/in_file_path, 'r')
    fo_ucf_test_list = open(ucf_directory/out_file_path, 'w')
    ucf_test_str = fo_ucf_test.read().split('\n')
    
    # saving test_list into a file, based on the split
    # format ## trimmed video_name label_index ##
    
    for index in range(len(ucf_test_str) - 1):
        label, buffer = ucf_test_str[index].split('/')
        test_list_str += (buffer + ' ' + str(ucf_labels[label]) + '\n')
        
    fo_ucf_test_list.write(test_list_str)
    fo_ucf_test_list.close()
    fo_ucf_test.close()