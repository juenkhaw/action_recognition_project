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
                 padding = 'SAME', use_bias = True, name = 'SpatioTemporalConv'):
        
        super(SpatioTemporalConv, self).__init__()
        
        # decomposing parameters into spatial and temporal components respectively
        spatial_f = [1, kernel_size[1], kernel_size[2]]
        spatial_s = [1, stride[1], stride[2]]
        spatial_p = [0, padding[1], padding[2]]
        
        temporal_f = [kernel_size[0], 1, 1]
        temporal_s = [stride[0], 1, 1]
        temporal_p = [padding[0], 0, 0]
        
        # compute the intermediate planes between spatial and temporal conv
        inter_planes = int(np.floor((np.product(kernel_size) * in_planes * out_planes) / 
                                    (np.product(spatial_f) * in_planes) + (np.product(temporal_f) * out_planes)))
        
        # perform 2D Conv --> BN --> ReLU --> 1D Conv
        self.spatial_conv = module.Conv3D(in_planes, inter_planes, kernel_size = spatial_f, 
                                          stride = spatial_s, padding = padding, use_BN = True, 
                                          activation = F.relu, use_bias = use_bias)
        
        self.temporal_conv = module.Conv3D(inter_planes, out_planes, kernel_size = temporal_f, 
                                           stride = temporal_s, padding = padding, use_BN = False,
                                           activation = None, use_bias = use_bias)
        
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
            self.downsampleconv = SpatioTemporalConv(in_planes, out_planes, kernel_size = (1, 1, 1), 
                                                     stride = (2, 2, 2))
            self.downsamplebn = nn.BatchNorm3d(out_planes)
            
            #downsample the residual
            self.conv1 = SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                            stride = (2, 2, 2))
            
        else:
            #do not downsample the x
            self.conv1 = SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME')
            
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.relu1 = nn.ReLU()
        
        self.conv2 = SpatioTemporalConv(out_planes, out_planes, kernel_size, padding = 'SAME')
        self.bn2 = nn.BatchNorm3d(out_planes)
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
                 name = 'SpatioTemporalResModule'):
        
        super(SpatioTemporalResModule, self).__init__()
        
        #implement the first conv to increase channels
        self.block1 = block_type(in_planes, out_planes, kernel_size, downsample)
        
        #the rest conv operations are identical
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_planes, out_planes, kernel_size)]
            
    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
            
        return x
    
class R2Plus1DNet(nn.Module):
    
    #Contains all modules of the network in orders
    VALID_ENDPOINTS = (
        'Conv3d_1_3x7x7',
        'Conv3d_2_x',
        'Conv3d_3_x',
        'Conv3d_4_x',
        'Conv3d_5_x',
        'Logits',
        'Predictions',
    )
    
    def __init__(self, layer_sizes, num_classes, device, block_type = SpatioTemporalResBlock, 
                 in_channels = 3, final_endpoint = 'Logits', name = 'R2+1D', verbose = 'True'):
        
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
            
        super(R2Plus1DNet, self).__init__()
        
        self._num_classes = num_classes
        self._final_endpoint = final_endpoint
        self._verbose = verbose
        self.end_points = {}
        
        # Conv3d_1_3x7x7
        self.end_points[self.VALID_ENDPOINTS[0]] = SpatioTemporalConv(in_channels, 64, kernel_size = (3, 7, 7), 
                       stride = (1, 2, 2), padding = 'SAME', name = name + self.VALID_ENDPOINTS[0]).to(device)
        
        # Conv3d_2_x
        self.end_points[self.VALID_ENDPOINTS[1]] = SpatioTemporalResModule(64, 64, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[0], downsample = False, name = name + self.VALID_ENDPOINTS[1]).to(device)
        
        # Conv3d_3_x
        self.end_points[self.VALID_ENDPOINTS[2]] = SpatioTemporalResModule(64, 128, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[1], downsample = True, name = name + self.VALID_ENDPOINTS[2]).to(device)
        
        # Conv3d_4_x
        self.end_points[self.VALID_ENDPOINTS[3]] = SpatioTemporalResModule(128, 256, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[2], downsample = True, name = name + self.VALID_ENDPOINTS[3]).to(device)
        
        # Conv3d_5_x
        self.end_points[self.VALID_ENDPOINTS[4]] = SpatioTemporalResModule(256, 512, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[3], downsample = True, name = name + self.VALID_ENDPOINTS[4]).to(device)
        
        # Logits
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        self.linear = nn.Linear(512, num_classes)
        
    def replaceLinear(self, num_classes):
        self._num_classes = num_classes
        self.linear = nn.Linear(512, num_classes)
        
    def forward(self, x):
        
        # perform each module until reaching final endpoint
        if self._verbose:
            print('input', x.shape)
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points.keys():
                x = self.end_points[end_point](x)
                if self._verbose:
                    print(end_point, x.shape)
                if end_point is self._final_endpoint:
                    break
        
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
    for end in model.VALID_ENDPOINTS:
        if end in model.end_points.keys():
            module = model.end_points[end]
            if isinstance(module, nn.Module):
                print(module)
    print(model)

#    x = torch.randn((1, 3, 8, 112, 112)).to(device)
    
#    try:
#        model(x)
#    except RuntimeError as e:
#        pass
#        if 'out of memory' in str(e):
#            for p in model.parameters():
#                if p.grad is not None:
#                    del p.grad
#            torch.cuda.empty_cache()