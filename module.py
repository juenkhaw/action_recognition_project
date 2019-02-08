# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:58:13 2019

@author: Juen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3D(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, 
                 stride = (1, 1, 1), padding = 'SAME', activation = F.relu, 
                 use_BN = True, use_bias = False, name = '3D_Conv'):
        
        super(Conv3D, self).__init__()
       
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._activation = activation
        self._use_BN = use_BN
        
        #presummed padding = 0, to be dynamically padded later on
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size = self._kernel_size, 
                               stride = self._stride, padding = (0, 0, 0), bias = use_bias)
        
        if use_BN:
            self.bn1 = nn.BatchNorm3d(out_planes, eps=0.001, momentum=0.01)
        
    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_size[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_size[dim] - (s % self._stride[dim]), 0)     
        
    def forward(self, x):
        
        if self._padding == 'SAME':
            
            #retrieve temporal and spatial dimension of the input
            t, h, w = x.size()[2:]
            
            #padding for each dimension of the input
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)
            
            #padding for each side for each dimension
            pad = (pad_w // 2, pad_w - pad_w // 2,
                   pad_h // 2, pad_h - pad_h // 2,
                   pad_t // 2, pad_t - pad_t // 2)
            
            x = F.pad(x, pad)
            
        x = self.conv1(x)
        
        if self._use_BN:
            x = self.bn1(x)
        if self._activation is not None:
            x = self._activation(x)
            
        return x

class MaxPool3DSame(nn.MaxPool3d):
        
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)
        
    def forward(self, x):
            
        #retrieve temporal and spatial dimension of the input
        t, h, w = x.size()[2:]
            
        #padding for each dimension of the input
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
            
        #padding for each side for each dimension
        pad = (pad_w // 2, pad_w - pad_w // 2,
             pad_h // 2, pad_h - pad_h // 2,
             pad_t // 2, pad_t - pad_t // 2)
          
        x = F.pad(x, pad)
            
        return super(MaxPool3DSame, self).forward(x)
    
if __name__ is '__main__':
    
    conv3D_test = Conv3D(109, 56, (3, 3, 3), stride=(2, 2, 2), padding = 'VALID')
    mp3D_test = MaxPool3DSame(kernel_size=(1,3,3), stride=(1,2,2))
    
    x = torch.randn((1, 64, 32, 112, 112))
    print(mp3D_test(x).shape)