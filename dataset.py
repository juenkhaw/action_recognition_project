# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:09:32 2019

@author: Juen
"""

import os
from pathlib import Path

from torch.utils.data import Dataset

import video_module

class VideoDataset(Dataset):
    """
    Constructor of this class requires:
        dataset : [ucf / hmdb]
        split : [0/1/2/3] where 0 indicates loading all dataset at once
        mode : [train / test] Special notes: [val] option to be implemented in future
        modelity : [rgb / flow]
        clip_len : [8 / 16] Target depth of training/testing clips
        test_mode : Activates to run on small samples to verify the correctness of implementation
    """
    
    def __init__(self, dataset, split, mode, modality, clip_len = 16, test_mode = True):
        
        # declare the parameters chosen as described in R(2+1)D papers
        self._resize_height = 128
        self._resize_width = 171
        self._crop_height = 112
        self._crop_width = 112
        self._crop_depth = clip_len
        
        # validate the arguments
        assert(dataset in ['ucf', 'hmdb'])
        assert(split in list(range(4)))
        assert(mode in ['train', 'test'])
        assert(modality in ['rgb', 'flow'])
        
        # locate the video <-> label mapping text files
        txt_files = []
        
        if dataset == 'ucf':
            self._dir = Path(r'..\dataset\UCF-101')
        else:
            self._dir = Path(r'..\dataset\HMDB-51')
            
        if split == 0:
            for i in range(3):
                txt_files.append(dataset + '_' + mode + 'list0' + str(i + 1) + '.txt')
        else:
            txt_files.append(dataset + '_' + mode + 'list0' + str(split) + '.txt')
            
        # reading in the content of mapping text files
        for i in len(txt_files):
            
        
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
    
if __name__ == '__main__':
    test = VideoDataset('ucf', 0, 'train', 'rgb')