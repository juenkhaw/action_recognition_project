# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:09:32 2019

@author: Juen
"""
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader

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
        laod_mode : select 'clip' for training, while 'video' for testing
        clips_per_video : number of clips to be contained within one video
    """
    
    def __init__(self, path, dataset, split, mode, modality, clip_len = 8, 
                 test_mode = True, test_amt = [8], mean_sub = False):
   
        # declare the parameters chosen as described in R(2+1)D papers
        self._resize_height = 128
        self._resize_width = 171
        self._crop_height = 112
        self._crop_width = 112
        self._mode = mode
        self._crop_depth = clip_len
        self._modality = modality
        self._mean_sub = mean_sub
        
        # validate the arguments
        assert(dataset in ['ucf', 'hmdb'])
        assert(split in list(range(1, 4)))
        assert(mode in ['train', 'test', 'validation'])
        assert(modality in ['rgb', 'flow'])
        # training should only be done on clips
            
        dataset_name = {'ucf' : 'UCF-101', 'hmdb' : 'HMDB-51'}
        _test_amt = {'train' : 0, 'test' : 2, 'validation' : 1}
        
        # ************CRUCIAL DATASET DIRECTORY*******************
        main_dir = path
        if dataset == 'ucf':
            # ************CRUCIAL DATASET DIRECTORY*******************
            #main_dir = Path(r'..\dataset\UCF-101')
            if self._modality == 'rgb':
                frame_dir = main_dir + '/ucf101_jpegs_256/jpegs_256'
            else:
                frame_dir = main_dir + '/ucf101_tvl1_flow/tvl1_flow'
        else:
            #main_dir = Path(r'..\dataset\HMDB-51')
            # ************CRUCIAL DATASET DIRECTORY*******************
            if self._modality == 'rgb':
                frame_dir = main_dir + '/hmdb51_jpegs_256/jpegs_256'
            else:
                frame_dir = main_dir + '/hmdb51_tvl1_flow/tvl1_flow'
        
        txt_file = dataset + '_' + mode + 'list0' + str(split) + '.txt'
            
        # reading in the content of mapping text files into a buffer
        # ************CRUCIAL MAPPING FILE PATH************************
        fo_txt = open('mapping/' + dataset_name[dataset] + '/' + txt_file, 'r')
        buffer_str = (fo_txt.read().split('\n'))[:-1]
        fo_txt.close()
        
        # organize raw strings mapping into X and y np arrays
        # X is the path to the directory where frames/flows are located
        # y is the true label
        self._clip_names, labels = [], []
        for i in range(len(buffer_str)):
            buffer_map = buffer_str[i].split(' ')
            self._clip_names.append([])
            if self._modality == 'rgb':
                self._clip_names[i].append(frame_dir + '/' + buffer_map[0].split('.')[0])
            else:
                self._clip_names[i].append(frame_dir + '/u/' + buffer_map[0].split('.')[0])
                self._clip_names[i].append(frame_dir + '/v/' + buffer_map[0].split('.')[0])
            labels.append(buffer_map[1])
        
        # convert the labels list into an np array
        self._labels = np.array([label for label in labels], dtype = np.int)
        
        # implement test mode to extremely scale down the dataset
        if test_mode == 'peek':
            #indices = [random.randrange(0, self.__len__()) for i in range(test_amt)]
            indices = list(range(int(test_amt[0])))
            self._clip_names = (np.array(self._clip_names))[indices]
            self._labels = self._labels[indices]
#            #for i in range(test_amt):
#            #    print(self._clip_names[i], self._labels[i])
            
        elif test_mode == 'distributed':
            
            _amt = int(test_amt[_test_amt[mode]])
            freqs = np.bincount(self._labels)
            assert(_amt <= np.min(freqs))
            test_clip_names = [[] for i in range(len(freqs) * _amt)]
            test_labels = []
            freq_sum = 0
            for i in range(len(freqs)):
                freq_sum += freqs[i]
                for j in range(_amt):
                    test_clip_names[i * _amt + j] = self._clip_names[freq_sum - freqs[i] + j]
                    test_labels.append(self._labels[freq_sum - freqs[i] + j])
            self._clip_names = test_clip_names
            self._labels = np.array([label for label in test_labels], dtype = np.int)
            
        
    def __getitem__(self, index):
        # retrieve the preprocessed clip np array
        #print(index)
        buffer = load_clips(self._clip_names[index], self._modality, 
                                         self._resize_height, self._resize_width, 
                                         self._crop_height, self._crop_width, self._crop_depth, 
                                         mode = self._mode, 
                                         mean_sub = self._mean_sub)
        return buffer, self._labels[index]
    
    def __len__(self):
        # ensure that the length of X is same with y
        #assert(len(self._clip_names) == len(self._labels))
        return len(self._labels)
    
class TwoStreamDataset(Dataset):
    """
    Dataset class that manages both rgb and flow datasets, and load both at once into a DataLoader
    
    Constructor requires:
        dataset : [ucf / hmdb] videoset to be loaded
        split : [1/2/3] split set to be loaded
        mode : [train / test] training or testing dataset to be loaded
        clip_len : [8 / 16] Target depth of training/testing clips
        test_mode : Activates to run on small samples to verify the correctness of implementation
        test_amt : Amount of labelled samples to be used if test_mode is activated
        laod_mode : select 'clip' for training, while 'video' for testing
        clips_per_video : number of clips to be contained within one video
    """
    
    def __init__(self, path, dataset, split, mode, clip_len = 8, 
                 test_mode = 'distributed', test_amt = 8, mean_sub = True):
        
        self._rgb_set = VideoDataset(path, dataset, split, mode, 'rgb', clip_len = clip_len, 
                                     test_mode = test_mode, test_amt = test_amt, mean_sub = False)
        
        self._flow_set = VideoDataset(path, dataset, split, mode, 'flow', clip_len = clip_len, 
                                     test_mode = test_mode, test_amt = test_amt, mean_sub = mean_sub)
        
    def __getitem__(self, index):
        # returning rgb, flow and labels at once
        rgbX, rgbY = self._rgb_set.__getitem__(index)
        flowX, flowY = self._flow_set.__getitem__(index)
        #print(self._rgb_set._clip_names[index], self._flow_set._clip_names[index])
        return rgbX, flowX, rgbY
    
    def __len__(self):
        assert(self._rgb_set.__len__() == self._flow_set.__len__())
        return self._rgb_set.__len__()
    
if __name__ == '__main__':
    train = TwoStreamDataset('../dataset/UCF-101', 'ucf', 1, 'train', test_mode = 'none')
    trainlaoder = DataLoader(train, batch_size = 2, shuffle = True)
    
    for i in range(len(train._rgb_set)):
        train.__getitem__(i)