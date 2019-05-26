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
            'vanilla-ld3',
            'vanilla-sigmoid-ld3',
            'feature-ld3'
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
        
#        if self._fusion == 'modality-1-layer':
#            self.linear1 = nn.Linear(1024, 2, bias=use_bias)
#            self.softmax = nn.Softmax(dim = 1)
        
        if self._fusion == 'feature-ld3':
            
            self.linear1 = nn.Linear(1024, 4096, bias = use_bias)
            self.bn1 = nn.BatchNorm1d(4096, momentum=bn_momentum, eps=bn_epson)
            
            self.linear2 = nn.Linear(4096, 2048, bias = use_bias)
            self.bn2 = nn.BatchNorm1d(2048, momentum=bn_momentum, eps=bn_epson)
            
            self.linear3 = nn.Linear(2048, 1024, bias = use_bias)
            
            self.output = nn.Softmax(dim = 1)
            self.relu = nn.ReLU()
            
        elif self._fusion == 'vanilla-ld3':
            
            self.linear1 = nn.Linear(1024, 256, bias=use_bias)
            self.bn1 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
            
            self.linear2 = nn.Linear(256, 64, bias=use_bias)
            self.bn2 = nn.BatchNorm1d(64, momentum=bn_momentum, eps=bn_epson)
            
            self.linear3 = nn.Linear(64, 2, bias=use_bias)
            
            self.output = nn.Softmax(dim = 1)
            self.relu = nn.ReLU()
            
        elif self._fusion == 'vanilla-sigmoid-ld3':
            
            self.linear1 = nn.Linear(1024, 256, bias=use_bias)
            self.bn1 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
            
            self.linear2 = nn.Linear(256, 64, bias=use_bias)
            self.bn2 = nn.BatchNorm1d(64, momentum=bn_momentum, eps=bn_epson)
            
            self.linear3 = nn.Linear(64, 1, bias=use_bias)
            
            self.output = nn.Sigmoid(dim = 1)
            self.relu = nn.ReLU()
            
        elif self._fusion == 'modality-3-layer-PREAP':
            
            self.rgb_conv1 = nn.Conv3d(512, 1024, kernel_size=(1, 7, 7))
            self.flow_conv1 = nn.Conv3d(512, 1024, kernel_size=(2, 7, 7))
            
            self.linear1 = nn.Linear(1024 * 2, 512, bias=use_bias)
            #self.bn1 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
            self.linear2 = nn.Linear(512, 128, bias=use_bias)
            #self.bn2 = nn.BatchNorm1d(64, momentum=bn_momentum, eps=bn_epson)
            self.linear3 = nn.Linear(128, 2, bias=use_bias)
            
            self.softmax = nn.Softmax(dim = 1)
            self.relu = nn.ReLU()
            
        self.final_softmax = nn.Softmax(dim = 1)
            
    def forward(self, x_rgb, x_flow):
        
        final_out = {}
        
        if self._fusion == 'average':
            final_out['SCORES'] = (x_rgb['SCORES'] + x_flow['SCORES']) / 2
            
        else:
            
            if self._fusion in ['vanilla-ld3', 'vanilla-sigmoid-ld3', 'feature-ld3']:
                
                # taking the average of final feature activations over samples for each modalities
                rgb_ap = x_rgb['AP'].reshape(x_rgb['AP'].shape[:2])
                flow_ap = x_flow['AP'].reshape(x_flow['AP'].shape[:2])
                
                # concat the averaged activations from both modalities
                ratio_out = torch.cat((rgb_ap, flow_ap), dim = 1)
                
#                if self._fusion == 'modality-1-layer':
#                    
#                    ratio_out = self.softmax(self.linear1(ratio_out))
#                    
#                    fusion_out = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1) + x_flow['FC'] * ratio_out[:, 1].reshape(ratio_out.shape[0], 1)
                
                ratio_out = self.relu(self.bn1(self.linear1(ratio_out)))
                ratio_out = self.relu(self.bn2(self.linear2(ratio_out)))
#                    ratio_out = self.relu(self.linear1(ratio_out))
#                    ratio_out = self.relu(self.linear2(ratio_out))
                ratio_out = self.output(self.linear3(ratio_out))
                
                if self._fusion == 'feature-ld3':
                    rgb_w = torch.sum(ratio_out[:, 0:512], dim = 1).view((-1, 1))
                    flow_w = torch.sum(ratio_out[:, 512:], dim = 1).view((-1, 1))
                    #ratio_out = torch.cat((rgb_w, flow_w), dim = 1)
                
                elif self._fusion == 'vanilla-ld3':
                    rgb_scores = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1)
                    flow_scores = x_flow['FC'] * ratio_out[:, 1].reshape(ratio_out.shape[0], 1)
                    
                elif self._fusion == 'vanilla-sigmoid-ld3':
                    rgb_scores = x_rgb['FC'] * ratio_out.reshape(ratio_out.shape[0], 1)
                    flow_scores = x_flow['FC'] * (1 - (ratio_out.reshape(ratio_out.shape[0], 1)))
                
#                print('RGB', torch.max(x_rgb['FC'], 1)[1], '\nFLOW', 
#                      torch.max(x_flow['FC'], 1)[1], '\nW', ratio_out)
                
                fusion_out = rgb_scores + flow_scores
                    
                final_out['WEIGHTS'] = ratio_out
                    
            elif self._fusion == 'modality-3-layer-PREAP':
                
                rgb_acv = self.relu(self.rgb_conv1(x_rgb['conv5_x'])).view(-1, 1024)
                flow_acv = self.relu(self.flow_conv1(x_flow['conv5_x'])).view(-1, 1024)
                
#                rgb_acv = x_rgb['Conv3d_5_x'].view(x_rgb['Conv3d_5_x'].shape[0], -1)
#                flow_acv = x_flow['Conv3d_5_x'].view(x_flow['Conv3d_5_x'].shape[0], -1)
                
                # concat the averaged activations from both modalities
                ratio_out = torch.cat((rgb_acv, flow_acv), dim = 1)
                
                ratio_out = self.relu(self.linear1(ratio_out))
                ratio_out = self.relu(self.linear2(ratio_out))
                ratio_out = self.softmax(self.linear3(ratio_out))
                
                fusion_out = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1) + x_flow['FC'] * ratio_out[:, 1].reshape(ratio_out.shape[0], 1)
            
            if 'FC' in self._endpoint:
                final_out['FC'] = fusion_out
                
            if 'SCORES' in self._endpoint:
                fusion_out = self.final_softmax(fusion_out)
                final_out['SCORES'] = fusion_out
            
        return final_out
        
if __name__ is '__main__':
    device = torch.device('cuda:0')
    model = FusionNet(fusion = 'feature-3-layer').to(device)
    
    x1 = {'AP':torch.randn((4, 512, 1, 1, 1)).to(device), 'FC':torch.randn((4, 101)).to(device)}
    x2 = {'AP':torch.randn((4, 512, 1, 1, 1)).to(device), 'FC':torch.randn((4, 101)).to(device)}
    
    s = model(x1, x2)
