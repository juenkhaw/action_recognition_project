# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:09:32 2019

@author: Juen
"""

import os
from pathlib import Path
import numpy as np
import random
import cv2

from torch.utils.data import Dataset

import video_module

class VideoDataset(Dataset):
    """
    Constructor of this class requires:
        dataset : [ucf / hmdb]
        split : [0/1/2/3] where 0 indicates loading all dataset at once
        mode : [train / test]
        modelity : [rgb / flow]
        clip_len : [8 / 16] Target depth of training/testing clips
        test_mode : Activates to run on small samples to verify the correctness of implementation
        test_amt : Amount of labelled samples to be used if test_mode is activated
    """
    
    def __init__(self, dataset, split, mode, modality, clip_len = 16, test_mode = True, test_amt = 8):
        
        # declare the parameters chosen as described in R(2+1)D papers
        self._resize_height = 128
        self._resize_width = 171
        self._crop_height = 112
        self._crop_width = 112
        self._crop_depth = clip_len
        
        self._modality = modality
        
        # validate the arguments
        assert(dataset in ['ucf', 'hmdb'])
        assert(split in list(range(4)))
        assert(mode in ['train', 'test'])
        assert(modality in ['rgb', 'flow'])
        
        # locate the video <-> label mapping text files
        txt_files = []
        
        if dataset == 'ucf':
            main_dir = Path(r'..\dataset\UCF-101')
            if self._modality == 'rgb':
                frame_dir = Path(main_dir/'ucf101_jpegs_256'/'jpegs_256')
            else:
                frame_dir = Path(main_dir/'ucf101_tvl1_flow'/'tvl1_flow')
        else:
            main_dir = Path(r'..\dataset\HMDB-51')
            if self._modality == 'rgb':
                frame_dir = Path(main_dir/'hmdb51_jpegs_256'/'jpegs_256')
            else:
                frame_dir = Path(main_dir/'hmdb51_tvl1_flow'/'tvl1_flow')
            
        if split == 0:
            for i in range(3):
                txt_files.append(dataset + '_' + mode + 'list0' + str(i + 1) + '.txt')
        else:
            txt_files.append(dataset + '_' + mode + 'list0' + str(split) + '.txt')
            
        # reading in the content of mapping text files
        buffer_str = []
        for i in range(len(txt_files)):
            fo_txt = open(main_dir/txt_files[i], 'r')
            buffer_str.extend((fo_txt.read().split('\n'))[:-1])
        fo_txt.close()
        
        # organize raw strings mapping into X and y np arrays
        self._clip_names, labels = [], []
        for i in range(len(buffer_str)):
            buffer_map = buffer_str[i].split(' ')
            self._clip_names.append([])
            if self._modality == 'rgb':
                self._clip_names[i].append(os.path.join(frame_dir, buffer_map[0].split('.')[0]))
            else:
                self._clip_names[i].append(os.path.join(frame_dir, 'u', buffer_map[0].split('.')[0]))
                self._clip_names[i].append(os.path.join(frame_dir, 'v', buffer_map[0].split('.')[0]))
            labels.append(buffer_map[1])
        
        # convert the labels list into an np array
        self._labels = np.array([label for label in labels], dtype = np.int)
        
        # implement test mode to extremely scale down the dataset
        if test_mode:
            indices = [random.randrange(0, self.__len__()) for i in range(test_amt)]
            self._clip_names = (np.array(self._clip_names))[indices]
            self._labels = self._labels[indices]
            #for i in range(test_amt):
            #    print(self._clip_names[i], self._labels[i])
        
    def __getitem__(self, index):
        # retrieve the preprocessed clip np array
        buffer = video_module.load_video(self._clip_names[index], self._modality, 
                                         self._resize_height, self._resize_width, 
                                         self._crop_height, self._crop_width, self._crop_depth)
        return buffer, self._labels[index]
    
    def __len__(self):
        # ensure that the length of X is same with y
        assert(len(self._clip_names) == len(self._labels))
        return len(self._labels)
    
if __name__ == '__main__':
    test = VideoDataset('hmdb', 0, 'train', 'rgb', test_mode = False)
    img, _ = test.__getitem__(0)
    
    for i in range(img.shape[1]):
        frame = img[:, i, :, :].transpose(1, 2, 0)
        cv2.imshow('buffer', frame)
        cv2.waitKey(100)
    
    cv2.destroyAllWindows()