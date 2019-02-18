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

from video_module import load_clips

class VideoDataset(Dataset):
    """
    Dataset class for managing training / testing video dataset and Dataloader object
    
    Constructor requires:
        dataset : [ucf / hmdb] videoset to be loaded
        split : [1/2/3] split set to be loaded
        mode : [train / test] training or testing dataset to be loaded
        modelity : [rgb / flow] modality of the videoset to be loaded
        clip_len : [8 / 16] Target depth of training/testing clips
        test_mode : Activates to run on small samples to verify the correctness of implementation
        test_amt : Amount of labelled samples to be used if test_mode is activated
    """
    
    def __init__(self, dataset, split, mode, modality, clip_len = 16, 
                 test_mode = True, test_amt = 8, 
                 load_mode = 'clip', clips_per_video = 1):
   
        # declare the parameters chosen as described in R(2+1)D papers
        self._resize_height = 128
        self._resize_width = 171
        self._crop_height = 112
        self._crop_width = 112
        self._mode = mode
        self._crop_depth = clip_len
        self._load_mode = load_mode
        self._clips_per_video = clips_per_video
        
        self._modality = modality
        
        # validate the arguments
        assert(dataset in ['ucf', 'hmdb'])
        assert(split in list(range(1, 4)))
        assert(mode in ['train', 'test'])
        assert(modality in ['rgb', 'flow'])
        # training should only be done on clips
        if mode == 'train':
            assert(load_mode == 'clip')
        # clip mode should invovle only one clip per video
        if load_mode == 'clip':
            assert(clips_per_video == 1)
        else:
            assert(clips_per_video > 1)
        
        
        # locate the video <-> label mapping text files
        txt_files = []
        
        if dataset == 'ucf':
            # ************CRUCIAL DATASET DIRECTORY*******************
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
        
        # appends mapping from all splits if loading of whole dataset is intended
        if split == 0:
            for i in range(3):
                txt_files.append(dataset + '_' + mode + 'list0' + str(i + 1) + '.txt')
        else:
            txt_files.append(dataset + '_' + mode + 'list0' + str(split) + '.txt')
            
        # reading in the content of mapping text files into a buffer
        buffer_str = []
        for i in range(len(txt_files)):
            # ************CRUCIAL MAPPING FILE PATH************************
            fo_txt = open(main_dir/txt_files[i], 'r')
            buffer_str.extend((fo_txt.read().split('\n'))[:-1])
        fo_txt.close()
        
        # organize raw strings mapping into X and y np arrays
        # X is the path to the directory where frames/flows are located
        # y is the true label
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
        buffer = load_clips(self._clip_names[index], self._modality, 
                                         self._resize_height, self._resize_width, 
                                         self._crop_height, self._crop_width, self._crop_depth, 
                                         mode = self._load_mode, 
                                         clips_per_video = self._clips_per_video)
        return buffer, self._labels[index]
    
    def __len__(self):
        # ensure that the length of X is same with y
        assert(len(self._clip_names) == len(self._labels))
        return len(self._labels)
    
if __name__ == '__main__':
    test = VideoDataset('hmdb', 1, 'test', 'flow', test_mode = False, load_mode = 'video', clips_per_video = 10)
    img, _ = test.__getitem__(0)
    
#    for i in range(img.shape[1]):
#        frame = img[:, i, :, :].transpose(1, 2, 0)
#        cv2.imshow('buffer', frame)
#        cv2.waitKey(100)
    
    cv2.destroyAllWindows()