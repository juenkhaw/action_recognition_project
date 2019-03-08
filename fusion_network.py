# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:56:16 2019

@author: Juen
"""
import torch
import torch.nn as nn

from network_r2p1d import R2Plus1DNet

class FusionNet(nn.Module):
    
    in_channels = {'rgb': 3, 'flow': 1}
    
    def __init__(self, layer_sizes, num_classes, device, network = R2Plus1DNet, 
                 fusion = 'average', endpoint = 'FC', 
                 bn_momentum = 0.1, bn_epson = 1e-3, name = 'R2+1D', verbose = False):
        
        super(FusionNet, self).__init__()
        
        self._num_classes = num_classes
        self._fusion = fusion
        
        assert(endpoint in network.VALID_ENDPOINTS)
        
        endpoint = 'FC' if fusion == 'average' else 'AP'
        
        self.rgb_net = network(layer_sizes, num_classes, device, in_channels = self.in_channels['rgb'], 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, endpoint = endpoint, 
                               name = 'R2P1D_RGB', verbose = verbose)
        self.flow_net = network(layer_sizes, num_classes, device, in_channels = self.in_channels['flow'], 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, endpoint = endpoint,
                               name = 'R2P1D_FLOW', verbose = verbose)
        
        if self._fusion != 'average':
            self.linear1_rgb = nn.Linear(512, num_classes)
            self.linear1_flow = nn.Linear(512, num_classes)
            
        if self._fusion == 'modality-wf':
            self.mfs_linear1 = nn.Linear(512, 128)
            self.mfs_linear2 = nn.Linear(128, 32)
            self.mfs_linear3 = nn.Linear(32, 1)
            self.mfs_softmax = nn.Softmax(dim = 0)
        
    def forward(self, x_rgb, x_flow):
        rgb_out = self.rgb_net(x_rgb)
        flow_out = self.flow_net(x_flow)
        
        if self._fusion == 'average':
            return (rgb_out + flow_out) / 2
        
        if self._fusion == 'modality-wf':
            # compute the scores for each modalities
            s1 = self.linear1_rgb(rgb_out)
            s2 = self.linear1_flow(flow_out)
            
            # taking the average of final feature activations for each modalities
            rgb_out = torch.mean(rgb_out, 0, True)
            flow_out = torch.mean(flow_out, 0, True)
            
            # concat the averaged activations from both modalities
            out = torch.cat((rgb_out, flow_out), dim = 0)
            
            # linear layers for weights
            out = self.mfs_linear3(self.mfs_linear2(self.mfs_linear1(out)))
            
            # softmax to make sure total of all weights = 1
            weights = self.mfs_softmax(out)
            
            print(weights)
            
            # return scores with learned weights
            return s1 * weights[0] + s2 * weights[1]
    
if __name__ is '__main__':
    device = torch.device('cuda:0')
    model = FusionNet(layer_sizes = [2, 2, 2, 2], num_classes = 101, device = device, 
                      endpoint = 'FC', fusion = 'average').to(device)
    
#    x1 = torch.randn((1, 3, 8, 112, 112)).to(device)
#    x2 = torch.randn((1, 2, 8, 112, 112)).to(device)
#    
#    s = model(x1, x2)