# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:42:57 2019

@author: Juen
"""

import time

import torch

from network_r2p1d import R2Plus1DNet

def generate_subbatches(sbs, *tensors):
    """
    Generate list of subbtaches from a batch of sample data
    
    Inputs:
        tensors : series of tensor batch to be partitioned
        
    Returns:
        subbatches : list of partitioned subbateches
    
    """
    
    #print(tensors.__class__)
    # engulf tensor into a list if there is only one tensor passed in
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
        
    subbatches = []
    for i in range(len(tensors)):
        subbatch = []
        part_num = tensors[i].shape[0] // sbs
        # if subbatch size is lower than the normal batch size
        if sbs < tensors[i].shape[0]:
            # partitioning batch with subbatch size
            subbatch = [tensors[i][j * sbs : j * sbs + sbs] for j in range(part_num)]
            # if there is any remainder in the batch
            if part_num * sbs < tensors[i].shape[0]:
                subbatch.append(tensors[i][part_num * sbs : ])
            subbatches.append(subbatch)
        else:
            subbatches.append([tensors[i]])
    
    return subbatches if len(tensors) > 1 else subbatches[0]

def train_stream(args, device, model, dataloaders, optimizer, criterion, scheduler = None, 
                pretrained_content = None):
    return    