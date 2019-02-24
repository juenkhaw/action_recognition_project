# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:56:16 2019

@author: Juen
"""
import torch
import torch.nn as nn

from network_r2p1d import R2Plus1DNet

class FusionNet(nn.Module):
    
    def __init__(self, layer_sizes, num_classes, device, network = R2Plus1DNet, 
                 bn_momentum = 0.1, bn_epson = 1e-3, name = 'R2+1D', verbose = False):
        
        super(FusionNet, self).__init__()
        
        self.rgb_net = network(layer_sizes, num_classes, device, in_channels = 3, 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, 
                               name = 'R2P1D_RGB', verbose = verbose)
        self.flow_net = network(layer_sizes, num_classes, device, in_channels = 2, 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, 
                               name = 'R2P1D_FLOW', verbose = verbose)
        
    def forward(self, x_rgb, x_flow):
        s1 = self.rgb_net(x_rgb)
        s2 = self.flow_net(x_flow)
        return (s1 + s2) / 2
    
if __name__ is '__main__':
    device = torch.device('cuda:0')
    model = FusionNet(layer_sizes = [2, 2, 2, 2], num_classes = 101, device = device).to(device)
    
    x1 = torch.randn((1, 3, 8, 112, 112)).to(device)
    x2 = torch.randn((1, 2, 8, 112, 112)).to(device)
    
    s = model(x1, x2)