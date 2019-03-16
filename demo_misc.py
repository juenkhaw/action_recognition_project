# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:11:52 2019

@author: Juen
"""
import numpy as np

def get_class_label(mapfile_path):
    # read in the UCF class labels for visualization purpose
    class_f = open(mapfile_path, 'r')
    class_raw_str = class_f.read().split('\n')[:-1]
    class_label = [raw_str.split(' ')[1] for raw_str in class_raw_str]
    
    return class_label

def get_prediction(scores, top_k, class_label):
    
    scores = scores.cpu().detach().numpy()
    indices = np.argsort(scores, axis = 1)[0, (101 - top_k) :][::-1]
    return {class_label[index] : scores[0, index] for index in indices}