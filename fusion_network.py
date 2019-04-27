# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:56:16 2019

@author: Juen
"""
import torch
import torch.nn as nn

gpu_name = 'cuda:0'
    
class FusionNet(nn.Module):
    
    VALID_FUSION = (
            'average',
            'modality-1-layer',
            'modality-3-layer'
            )
    
    VALID_ENDPOINTS = (
            'FC',
            'SCORES'
            )
    
    def __init__(self, fusion = 'average', use_bias = True, bn_momentum = 0.1, bn_epson = 1e-3, 
                 endpoint = ['FC']):
        
        super(FusionNet, self).__init__()
        
        self._fusion = fusion
        
        # validate list of endpoint
        for endp in endpoint:
            assert endp in self.VALID_ENDPOINTS
        self._endpoint = endpoint
        
        assert(self._fusion in self.VALID_FUSION)
        
        if self._fusion == 'modality-1-layer':
            self.linear1 = nn.Linear(1024, 2)
            self.softmax = nn.Softmax(dim = 1)
            
        elif self._fusion == 'modality-3-layer':
            self.linear1 = nn.Linear(1024, 256)
            self.bn1 = nn.BatchNorm1d(256)
            self.linear2 = nn.Linear(256, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.linear3 = nn.Linear(64, 2)
            self.softmax = nn.Softmax(dim = 1)
            self.relu = nn.ReLU()
            
        self.final_softmax = nn.Softmax(dim = 1)
            
    def forward(self, x_rgb, x_flow):
        
        final_out = {}
        
        if self._fusion == 'average':
            final_out['SCORES'] = (x_rgb['SCORES'] + x_flow['SCORES']) / 2
            
        elif self._fusion == 'modality-1-layer':
            # taking the average of final feature activations over samples for each modalities
            rgb_ap = x_rgb['AP'].reshape(x_rgb['AP'].shape[:2])
            flow_ap = x_flow['AP'].reshape(x_flow['AP'].shape[:2])
            
            #print(rgb_ap.shape, flow_ap.shape)
            
            # concat the averaged activations from both modalities
            ratio_out = torch.cat((rgb_ap, flow_ap), dim = 1)
            
            ratio_out = self.softmax(self.linear1(ratio_out))
            
            fusion_out = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1) + x_flow['FC'] * ratio_out[:, 1].reshape(ratio_out.shape[0], 1)
            
            final_out['WEIGHTS'] = ratio_out
            
            if 'FC' in self._endpoint:
                final_out['FC'] = fusion_out
                
            if 'SCORES' in self._endpoint:
                fusion_out = self.final_softmax(fusion_out)
                final_out['SCORES'] = fusion_out
                
        elif self._fusion == 'modality-3-layer':
            # taking the average of final feature activations over samples for each modalities
            rgb_ap = x_rgb['AP'].reshape(x_rgb['AP'].shape[:2])
            flow_ap = x_flow['AP'].reshape(x_flow['AP'].shape[:2])
            
            # concat the averaged activations from both modalities
            ratio_out = torch.cat((rgb_ap, flow_ap), dim = 1)
            
            ratio_out = self.relu(self.bn1(self.linear1(ratio_out)))
            ratio_out = self.relu(self.bn2(self.linear2(ratio_out)))
            ratio_out = self.softmax(self.linear3(ratio_out))
            
            fusion_out = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1) + x_flow['FC'] * ratio_out[:, 1].reshape(ratio_out.shape[0], 1)
            
            final_out['WEIGHTS'] = ratio_out
            
            if 'FC' in self._endpoint:
                final_out['FC'] = fusion_out
                
            if 'SCORES' in self._endpoint:
                fusion_out = self.final_softmax(fusion_out)
                final_out['SCORES'] = fusion_out
            
        return final_out
        
if __name__ is '__main__':
    device = torch.device('cuda:0')
    model = FusionNet(fusion = 'modality-1-layer').to(device)
    
    x1 = {'AP':torch.randn((4, 512, 1, 1, 1)).to(device), 'SCORES':torch.randn((4, 101)).to(device)}
    x2 = {'AP':torch.randn((4, 512, 1, 1, 1)).to(device), 'SCORES':torch.randn((4, 101)).to(device)}
    
    s = model(x1, x2)
