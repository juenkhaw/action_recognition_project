# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:25:34 2019

@author: Juen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def compute_pad(dim_size, kernel_size, stride):
    """
    Dynamically computes padding for each dimension of the input volume
    
    Inputs:
        dim_size : dimension (w, h, t) of the input volume
        kernel_size : dimension (w, h, t) of kernel
        stride : stride applied for each dimension
        
    Returns:
        list of 6 ints with padding on both side of all dimensions
    """
    pads = []
    for i in range(len(dim_size) - 1, -1, -1):
        if dim_size[i] % stride[i] == 0:
            pad = max(kernel_size[i] - stride[i], 0)
        else:
            pad = max(kernel_size[i] - (dim_size[i] % stride[i]), 0)
        pads.append(pad // 2)
        pads.append(pad - pad // 2)
    
    return pads

class SpatioTemporalDeconv(nn.Module):
    
    def __init__(self, out_planes, in_planes, kernel_size, stride = (1, 1, 1), 
                 inter_planes = None, temporal_relu = False, spatial_relu = True, 
                 padding = 'SAME', use_bias = False, name = 'SpatioTemporalDeconv'):
        
        super(SpatioTemporalDeconv, self).__init__()
        
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._t_relu = temporal_relu
        self._s_relu = spatial_relu
        
        if inter_planes is None:
            inter_planes = int((kernel_size[0] * in_planes * out_planes * kernel_size[1] * kernel_size[2]) / 
                               (in_planes * kernel_size[1] * kernel_size[2] + kernel_size[0] * out_planes))
            
        if spatial_relu:
            self.spatial_relu = nn.ReLU()
        self.spatial_deconv = nn.Conv3d(inter_planes, in_planes, kernel_size = (1, kernel_size[1], kernel_size[2]), 
                                         stride = (1, stride[1], stride[2]), padding = (0, 0, 0), bias = use_bias)
        
        if temporal_relu:
            self.temporal_relu = nn.ReLU()
        self.temporal_deconv = nn.Conv3d(out_planes, inter_planes, kernel_size = (kernel_size[0], 1, 1), 
                                         stride = (stride[0], 1, 1), padding = (0, 0, 0), bias = use_bias)
        
    def forward(self, x):
        
        if self._padding == 'SAME':
            x = F.pad(x, compute_pad(x.shape[2:], self._kernel_size, self._stride))
        
        if self._t_relu:
            x = self.temporal_relu(x)
        x = self.temporal_deconv(x)
        if self._s_relu:
            x = self.spatial_deconv(x)
            
        return x

#class DeConvNet(nn.Module):
#    
#    def __init__(self):
#        
#        super(DeConvNet, self).__init__()
#        
#        self.spatial_deconv = nn.Conv3d(45, 3, kernel_size = (1, 7, 7), 
#                                        stride = (1, 2, 2), padding = (0, 3, 3), bias = False)
#        self.temporal_deconv = nn.Conv3d(64, 45, kernel_size = (3, 1, 1), 
#                                        stride = 1, padding = (1, 0, 0), bias = False)
#        self.relu = nn.ReLU()
#        
#    def forward(self, x):
#        x = self.relu(x)
#        x = self.temporal_deconv(x)
#        x = self.relu(x)
#        x = self.spatial_deconv(x)
#        
#        return x
    
if __name__ == '__main__':
    model = SpatioTemporalDeconv(64, 3, (3, 7, 7), stride = (1, 2, 2), 
                                 inter_planes = 45, temporal_relu = True)
    x = torch.randn((1, 64, 8, 56, 56))
    y = model(x)