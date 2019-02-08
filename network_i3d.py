# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 21:56:29 2019

@author: Juen
"""

import torch
import torch.nn as nn

import module

class InceptionModule(nn.Module):
    
    #out_planes expected to have 6 numbers
    #corresponding to each Convs
    def __init__(self, in_planes, out_planes, name):
        
        super(InceptionModule, self).__init__()
        
        self.branch0 = module.Conv3D(in_planes, out_planes[0], kernel_size = (1, 1, 1), name = name + '/branch0_Conv3D(1)')
        self.branch1a = module.Conv3D(in_planes, out_planes[1], kernel_size = (1, 1, 1), name = name + '/branch1a_Conv3D(1)')
        self.branch1b = module.Conv3D(out_planes[1], out_planes[2], kernel_size = (3, 3, 3), name = name + '/branch1b_Conv3D(3)')
        self.branch2a = module.Conv3D(in_planes, out_planes[3], kernel_size = (1, 1, 1), name = name + '/branch2a_Conv3D(1)')
        self.branch2b = module.Conv3D(out_planes[3], out_planes[4], kernel_size = (3, 3, 3), name = name + '/branch2b_Conv3D(3)')
        self.branch3a = module.MaxPool3DSame(kernel_size = (3, 3, 3), stride = (1, 1, 1))
        self.branch3b = module.Conv3D(in_planes, out_planes[5], kernel_size = (1, 1, 1), name = name + '/branch3b_Conv3D(1)')
        
    def forward(self, x):
        
        x0 = self.branch0(x)
        x1 = self.branch1b(self.branch1a(x))
        x2 = self.branch2b(self.branch2a(x))
        x3 = self.branch3b(self.branch3a(x))
        
        #concatenate the activations from each branch at the channel dimension
        return torch.cat([x0, x1, x2, x3], dim = 1)
    
class InceptionI3D(nn.Module):
    
    #Contains all modules of the network in orders
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )
    
    def __init__(self, num_classes, name = 'inception_i3d', in_channels = 3, 
                 final_endpoint = 'Logits', dropout_keep_prob = 0.5, modified = True):
        
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)
            
        super(InceptionI3D, self).__init__()
        
        self._num_classes = num_classes
        self._final_endpoint = final_endpoint
        self._modified = modified
        self.end_points = {}
        
        #Conv3d_1a_7x7
        self.end_points[self.VALID_ENDPOINTS[0]] = module.Conv3D(in_channels, 64, kernel_size = (7, 7, 7), 
                       stride = (2, 2, 2), padding = 'SAME', name = name + self.VALID_ENDPOINTS[0])
        
        # MaxPool3d_2a_3x3
        self.end_points[self.VALID_ENDPOINTS[1]] = module.MaxPool3DSame(kernel_size = (1, 3, 3), 
                       stride = (1, 2, 2), padding = (0, 0, 0))
        
        # Conv3d_2b_1x1
        self.end_points[self.VALID_ENDPOINTS[2]] = module.Conv3D(64, 64, kernel_size = (1, 1, 1), 
                       name = name + self.VALID_ENDPOINTS[2])
        
        # Conv3d_2c_3x3
        self.end_points[self.VALID_ENDPOINTS[3]] = module.Conv3D(64, 192, kernel_size = (3, 3, 3), 
                       padding = 'SAME', name = name + self.VALID_ENDPOINTS[3])
        
        # MaxPool3d_3a_3x3
        self.end_points[self.VALID_ENDPOINTS[4]] = module.MaxPool3DSame(kernel_size = (1, 3, 3), 
                       stride = (1, 2, 2), padding = (0, 0, 0))
        
        # Mixed_3b
        self.end_points[self.VALID_ENDPOINTS[5]] = InceptionModule(192, [64, 96, 128, 16, 32, 32], 
                       name = name + self.VALID_ENDPOINTS[5])
        
        # Mixed_3c
        self.end_points[self.VALID_ENDPOINTS[6]] = InceptionModule(256, [128, 128, 192, 32, 96, 64], 
                       name = name + self.VALID_ENDPOINTS[6])
        
        # MaxPool3d_4a_3x3
        self.end_points[self.VALID_ENDPOINTS[7]] = module.MaxPool3DSame(kernel_size = (3, 3, 3), 
                       stride = (2, 2, 2), padding = (0, 0, 0))
        
        # Mixed_4b
        self.end_points[self.VALID_ENDPOINTS[8]] = InceptionModule(480, [192, 96, 208, 16, 48, 64], 
                       name = name + self.VALID_ENDPOINTS[8])
        
        # Mixed_4c
        self.end_points[self.VALID_ENDPOINTS[9]] = InceptionModule(512, [160, 112, 224, 24, 64, 64], 
                       name = name + self.VALID_ENDPOINTS[9])
        
        # Mixed_4d
        self.end_points[self.VALID_ENDPOINTS[10]] = InceptionModule(512, [128, 128, 256, 24, 64, 64], 
                       name = name + self.VALID_ENDPOINTS[10])
        
        # Mixed_4e
        self.end_points[self.VALID_ENDPOINTS[11]] = InceptionModule(512, [112, 144, 288, 32, 64, 64], 
                       name = name + self.VALID_ENDPOINTS[11])
        
        # Mixed_4f
        self.end_points[self.VALID_ENDPOINTS[12]] = InceptionModule(528, [256, 160, 320, 32, 128, 128], 
                       name = name + self.VALID_ENDPOINTS[8])
        
        # MaxPool3d_5a_2x2
        self.end_points[self.VALID_ENDPOINTS[13]] = module.MaxPool3DSame(kernel_size = (2, 2, 2), 
                       stride = (2, 2, 2), padding = (0, 0, 0))
        
        # Mixed_5b
        self.end_points[self.VALID_ENDPOINTS[14]] = InceptionModule(832, [256, 160, 320, 32, 128, 128], 
                       name = name + self.VALID_ENDPOINTS[14])
        
        # Mixed_5c
        self.end_points[self.VALID_ENDPOINTS[15]] = InceptionModule(832, [384, 192, 384, 48, 128, 128], 
                       name = name + self.VALID_ENDPOINTS[15])
        
        # Logits
        self.avg_pool = nn.AvgPool3d(kernel_size = (2, 7, 7), stride = (1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        
        # linear forward
        self.logits = module.Conv3D(1024, num_classes, kernel_size = (1, 1, 1), 
                                    padding = 'VALID', activation = None, 
                                    use_BN = False, use_bias = True, name = self.VALID_ENDPOINTS[16])
        
        if self._modified:
            self.fc1 = nn.Linear(num_classes * 7, num_classes)
        
    def replace_logits(self, num_classes):
        
        #replace the last linear logits
        self._num_classes = num_classes
        
        self.logits = module.Conv3D(1024, num_classes, kernel_size = (1, 1, 1), 
                                padding = 'SAME', activation = None, 
                                use_BN = False, use_bias = True, name = self.VALID_ENDPOINTS[16])
        if self._modified:
            self.fc1 = nn.Linear(num_classes * 7, num_classes)
        
    def forward(self, x):
        
        # perform each module until reaching final endpoint
        print('input', x.shape)
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points.keys():
                x = self.end_points[end_point](x)
                print(end_point, x.shape)
                if end_point is self._final_endpoint:
                    break
                
        # pre-fc operations
        x = self.logits(self.dropout(self.avg_pool(x)))
        print('Pre-fc', x.shape)
        
        #squeeze the spatial dimension before fc layer
        logits = x.view(-1, self._num_classes * 7)
        print('Pre-fc reshaped', logits.shape)
        logits = self.fc1(logits)
        
        #logits is expected to be (batch_size, num_classes, temporal)
        print('Post-fc', logits.shape)
        
        return logits

if __name__ is '__main__':
    
    model = InceptionI3D(101)
    x = torch.randn((1, 3, 64, 224, 224))
    
    model(x)
    
    