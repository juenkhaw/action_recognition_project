# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:58:13 2019

@author: Juen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from collections import OrderedDict

class Conv3D(nn.Module):
    """
    Module consisting of 3D convolution, 3D BN and ReLU in combination
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        stride : (t, h, w) striding over the input volume
        padding : [SAME/VALID] padding technique to be applied
        activation : module for activation function (can be None)
        use_BN : applies Batch Normalization or not
        bn_mom : BN momentum hyperparameter
        bn_eps : BN epsilon hyperparameter
        use_bias : invovles bias learning in 3DConv/Linear or not
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, 
                 stride = (1, 1, 1), padding = 'SAME', activation = True, 
                 use_BN = True, bn_mom = 0.1, bn_eps = 1e-3, 
                 use_bias = False, name = '3D_Conv'):
        
        super(Conv3D, self).__init__()
       
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self.activation = activation
        self._use_BN = use_BN
        
        self.conv = nn.Sequential(OrderedDict([]))
        
        #presummed padding = 0, to be dynamically padded later on
        self.conv.add_module(name + 'conv', 
                               nn.Conv3d(in_planes, out_planes, kernel_size = self._kernel_size, 
                                         stride = self._stride, padding = (0, 0, 0), bias = use_bias))
        
        if use_BN:
            self.conv.add_module(name + 'bn', nn.BatchNorm3d(out_planes, eps = bn_eps, momentum = bn_mom))
            
        if activation:
            self.conv.add_module(name + 'relu', nn.ReLU())
        
    def compute_pad(self, dim, s):
        """
        Dynamically computes padding for each dimension of the input volume
        
        Inputs:
            dim : dimension of the input volume
            s : size (h/w/depth) of that particular dimension
            
        Returns:
            padding of that particular dimension
        """
        
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
            
        return self.conv(x)

class MaxPool3DSame(nn.MaxPool3d):
    """
    Module of SAME max 3D pooling with dynamic padding
    """
        
    def compute_pad(self, dim, s):
        """
        Dynamically computes padding for each dimension of the input volume
        
        Inputs:
            dim : dimension of the input volume
            s : size (h/w/depth) of that particular dimension
            
        Returns:
            padding of that particular dimension
        """
        
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
    
def msra_init(net):
    """
    Initializes parameters of the model with MSRA initialization method as 
    implemented in the author program
    
    Inputs:
        net : model to be trained on
        
    Returns:
        None
    """
    
    #count = [0, 0, 0]
    for module in net.modules():
        if isinstance(module, nn.Conv3d):
            #count[0] += 1
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            #count[1] += 1
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            #count[2] += 1
            init.normal_(module.weight, std = 1e-3)
            if module.bias is not None:
                init.constant_(module.bias, 0)
    #print(count)
    
def getModuleCount(net):
    """
    Displays number of Conv3D, BN3D and Linear module in the model
    
    Inputs:
        net : model to be trained on
        
    Returns:
        None
    """
    
    count = [0, 0, 0]
    for module in net.modules():
        if isinstance(module, nn.Conv3d):
            count[0] += 1
        elif isinstance(module, nn.BatchNorm3d):
            count[1] += 1
        elif isinstance(module, nn.Linear):
            count[2] += 1
    print(count)
    
if __name__ is '__main__':
    
    conv3D_test = Conv3D(109, 56, (3, 3, 3), stride=(2, 2, 2), padding = 'VALID')
    mp3D_test = MaxPool3DSame(kernel_size=(1,3,3), stride=(1,2,2))
    
    x = torch.randn((1, 64, 32, 112, 112))
    print(mp3D_test(x).shape)