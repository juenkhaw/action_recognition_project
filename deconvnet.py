# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:25:34 2019

@author: Juen
"""

import torch
import torch.nn as nn

import numpy as np

class DeConvNet(nn.Module):
    
    def __init__(self):
        
        super(DeConvNet, self).__init__()
        
        self.spatial_deconv = nn.Conv3d(45, 3, kernel_size = (1, 7, 7), 
                                        stride = (1, 2, 2), padding = (0, 3, 3), bias = False)
        self.temporal_deconv = nn.Conv3d(64, 45, kernel_size = (3, 1, 1), 
                                        stride = 1, padding = (1, 0, 0), bias = False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(x)
        x = self.temporal_deconv(x)
        x = self.relu(x)
        x = self.spatial_deconv(x)
        
        return x
    
if __name__ == '__main__':
    model = DeConvNet()
    x = torch.randn((1, 64, 8, 56, 56))
    y = model(x)