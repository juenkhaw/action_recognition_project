# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:21:55 2019

@author: Juen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import module

class SpatioTemporalConv(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, stride = (1, 1, 1), 
                 bn_mom = 0.1, bn_eps = 1e-3, mi = None, 
                 bn_relu_first_conv = True, bn_relu_second_conv = False, 
                 padding = 'SAME', use_bias = False, name = 'SpatioTemporalConv'):
        
        super(SpatioTemporalConv, self).__init__()
        
        # decomposing parameters into spatial and temporal components respectively
        spatial_f = [1, kernel_size[1], kernel_size[2]]
        spatial_s = [1, stride[1], stride[2]]
        spatial_p = [0, padding[1], padding[2]]
        
        temporal_f = [kernel_size[0], 1, 1]
        temporal_s = [stride[0], 1, 1]
        temporal_p = [padding[0], 0, 0]
        
        # compute the intermediate planes between spatial and temporal conv
        if mi is None:
            inter_planes = int((kernel_size[0] * in_planes * out_planes * kernel_size[1] * kernel_size[2]) / 
                               (in_planes * kernel_size[1] * kernel_size[2] + kernel_size[0] * out_planes))
        else:
            inter_planes = mi
        
        # perform 2D Conv --> BN --> ReLU --> 1D Conv
        self.spatial_conv = module.Conv3D(in_planes, inter_planes, kernel_size = spatial_f, 
                                          stride = spatial_s, padding = padding, 
                                          use_BN = bn_relu_first_conv, bn_mom = bn_mom, bn_eps = bn_eps, 
                                          activation = F.relu if bn_relu_first_conv else None, use_bias = use_bias)
        
        self.temporal_conv = module.Conv3D(inter_planes, out_planes, kernel_size = temporal_f, 
                                           stride = temporal_s, padding = padding, use_BN = bn_relu_second_conv, 
                                           bn_mom = bn_mom, bn_eps = bn_eps, 
                                           activation = F.relu if bn_relu_second_conv else None, use_bias = use_bias)
        
    def forward(self, x):
        x = self.spatial_conv(x)
        #print(x.shape)
        x = self.temporal_conv(x)
        #print(x.shape)
        
        return x
    
class SpatioTemporalResBlock(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, downsample = False, 
                 bn_eps = 1e-3, bn_mom = 0.1, name = 'SpatioTemporalResBlock'):
        
        super(SpatioTemporalResBlock, self).__init__()
        
        self._downsample = downsample
        
        if self._downsample:
            #downsample x to be the same spatiotemporal dimension as output
            self.downsampleconv = nn.Conv3d(in_planes, out_planes, [1, 1, 1], stride = [2, 2, 2], bias = False)
            self.downsamplebn = nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps)
            
            #downsample the residual
            self.conv1 = SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                            stride = (2, 2, 2), bn_mom = bn_mom, bn_eps = bn_eps)
            
        else:
            #do not downsample the x
            self.conv1 = SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                            bn_mom = bn_mom, bn_eps = bn_eps)
            
        self.bn1 = nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps)
        self.relu1 = nn.ReLU()
        
        self.conv2 = SpatioTemporalConv(out_planes, out_planes, kernel_size, padding = 'SAME', 
                                        bn_mom = bn_mom, bn_eps = bn_eps)
        self.bn2 = nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        
        res = self.relu1(self.bn1(self.conv1(x)))
        #print(res.shape)
        res = self.bn2(self.conv2(res))
        #print(res.shape)
        
        if self._downsample:
            x = self.downsamplebn(self.downsampleconv(x))
            
        #print(x.shape)
        return self.relu2(x + res)
    
class SpatioTemporalResModule(nn.Module):
    
    def __init__(self, in_planes, out_planes, kernel_size, layer_size, 
                 block_type = SpatioTemporalResBlock, downsample = False, 
                 bn_eps = 1e-3, bn_mom = 0.1, 
                 name = 'SpatioTemporalResModule'):
        
        super(SpatioTemporalResModule, self).__init__()
        
        #implement the first conv to increase channels
        self.block1 = block_type(in_planes, out_planes, kernel_size, downsample, bn_mom = bn_mom, bn_eps = bn_eps)
        
        #the rest conv operations are identical
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_planes, out_planes, kernel_size, bn_mom = bn_mom, bn_eps = bn_eps)]
            
    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
            
        return x
    
class R2Plus1DNet(nn.Module):
    
    VALID_ENDPOINTS = (
        'Conv3d_1_3x7x7',
        'Conv3d_2_x',
        'Conv3d_3_x',
        'Conv3d_4_x',
        'Conv3d_5_x',
        'Logits',
    )
    
    def __init__(self, layer_sizes, num_classes, device, block_type = SpatioTemporalResBlock, 
                 in_channels = 3, bn_momentum = 0.1, bn_epson = 1e-3, name = 'R2+1D', verbose = 'True'):
            
        super(R2Plus1DNet, self).__init__()
        
        self._num_classes = num_classes
        self._verbose = verbose
        
        self.net = nn.ModuleList([
                SpatioTemporalConv(in_channels, 64, kernel_size = (3, 7, 7), 
                       stride = (1, 2, 2), padding = 'SAME', mi = 45, name = name + self.VALID_ENDPOINTS[0], 
                       bn_mom = bn_momentum, bn_eps = bn_epson, bn_relu_second_conv = True).to(device),
                SpatioTemporalResModule(64, 64, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[0], downsample = False, name = name + self.VALID_ENDPOINTS[1], 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device),
                SpatioTemporalResModule(64, 128, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[1], downsample = True, name = name + self.VALID_ENDPOINTS[2], 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device),
                SpatioTemporalResModule(128, 256, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[2], downsample = True, name = name + self.VALID_ENDPOINTS[3], 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device),
                SpatioTemporalResModule(256, 512, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[3], downsample = True, name = name + self.VALID_ENDPOINTS[4], 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device),
                ])
        
        # Logits
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        #self.linear = nn.Linear(512, num_classes)
        self.linear = nn.Linear(512, num_classes)
        

    def replaceLinear(self, num_classes):
        self._num_classes = num_classes
        self.linear = nn.Linear(512, num_classes)        
        
    def forward(self, x):
        
        # perform each module until reaching final endpoint
        if self._verbose:
            print('Input', x.shape)
            
        for i, module in enumerate(self.net):
            x = self.net[i](x)
            if self._verbose:
                print(self.VALID_ENDPOINTS[i], x.shape)
        
        # pre-fc
        x = self.pool(x)
        x = x.view(-1, 512)
        if self._verbose:
            print('Pre FC', x.shape)
        
        # fc linear layer
        x = self.linear(x)
        if self._verbose:
            print('Post FC', x.shape)
        
        return x        

if __name__ is '__main__':
    device = torch.device('cpu')
    model = R2Plus1DNet(layer_sizes = [2, 2, 2, 2], num_classes = 101, device = device, in_channels = 2, verbose = True).to(device)
    #module.model_summary(model)
    #module.msra_init(model)

    #x = torch.randn((1, 2, 8, 112, 112)).to(device)
    #model(x)
    
#    try:
#        model(x)
#    except RuntimeError as e:
#        pass
#        if 'out of memory' in str(e):
#            for p in model.parameters():
#                if p.grad is not None:
#                    del p.grad
#            torch.cuda.empty_cache()