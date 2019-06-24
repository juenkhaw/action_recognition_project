# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:56:16 2019

@author: Juen
"""
import torch
import torch.nn as nn

from network_r2p1d import R2Plus1DNet

gpu_name = 'cuda:0'
    
class FusionNet(nn.Module):
    
    VALID_FUSION = (
            'average',
            'vanilla-ld3',
            'vanilla-ld3-2',
            'class-ld3',
            'class-ld3-2',
            'activation-ld3'
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
            
        if 'vanilla-ld3' in self._fusion:
            
            self.linear1 = nn.Linear(1024, 256, bias=use_bias)
            self.bn1 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
            
            self.linear2 = nn.Linear(256, 64, bias=use_bias)
            self.bn2 = nn.BatchNorm1d(64, momentum=bn_momentum, eps=bn_epson)
            
            self.linear3 = nn.Linear(64, 2 if self._fusion == 'vanilla-ld3' else 1, bias=use_bias)
            
            if self._fusion == 'vanilla-ld3':
                self.output = nn.Softmax(dim = 1)
            else:
                self.output = nn.Sigmoid()
            
            self.relu = nn.ReLU()
            
        elif 'class-ld3' in self._fusion:
            
            self.linear1 = nn.Linear(1024, 512, bias=use_bias)
            self.bn1 = nn.BatchNorm1d(512, momentum=bn_momentum, eps=bn_epson)
            
            self.linear2 = nn.Linear(512, 256, bias=use_bias)
            self.bn2 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
            
            self.linear3 = nn.Linear(256, 202 if self._fusion == 'class-ld3' else 101, bias=use_bias)
            
            if self._fusion == 'class-ld3':
                self.output = nn.Softmax(dim = 1)
            else:
                self.output = nn.Sigmoid()
                
            self.relu = nn.ReLU()
            
        elif self._fusion == 'activation-ld3':
            
            self.linear1 = nn.Linear(1024, 512, bias=use_bias)
            self.bn1 = nn.BatchNorm1d(512, momentum=bn_momentum, eps=bn_epson)
            
            self.linear2 = nn.Linear(512, 256, bias=use_bias)
            self.bn2 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
            
            self.linear3 = nn.Linear(256, 101, bias=use_bias)
            
            self.relu = nn.ReLU()
            
        self.final_softmax = nn.Softmax(dim = 1)
            
    def forward(self, x_rgb, x_flow):
        
        final_out = {}
        
        if self._fusion == 'average':
            final_out['SCORES'] = (x_rgb['SCORES'] + x_flow['SCORES']) / 2
            
        else:
                
            # taking the average of final feature activations over samples for each modalities
            rgb_ap = x_rgb['AP'].reshape(x_rgb['AP'].shape[:2])
            flow_ap = x_flow['AP'].reshape(x_flow['AP'].shape[:2])
            
            # concat the averaged activations from both modalities
            ratio_out = torch.cat((rgb_ap, flow_ap), dim = 1)
            
            ratio_out = self.relu(self.bn1(self.linear1(ratio_out)))
            ratio_out = self.relu(self.bn2(self.linear2(ratio_out)))
            
            if self._fusion in ['vanilla-ld3', 'vanilla-ld3-2', 'class-ld3', 'class-ld3-2']:
                ratio_out = self.output(self.linear3(ratio_out))
            else:
                fusion_out = self.linear3(ratio_out)
            
            if self._fusion == 'vanilla-ld3':
                rgb_scores = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1)
                flow_scores = x_flow['FC'] * ratio_out[:, 1].reshape(ratio_out.shape[0], 1)
                
            elif self._fusion == 'vanilla-ld3-2':
                rgb_scores = x_rgb['FC'] * ratio_out[:, 0].reshape(ratio_out.shape[0], 1)
                flow_scores = x_flow['FC'] * (1 - ratio_out[:, 0].reshape(ratio_out.shape[0], 1))
                
            elif self._fusion == 'class-ld3':
                rgb_scores = x_rgb['FC'] * ratio_out[:, :101] * 101
                flow_scores = x_flow['FC'] * ratio_out[:, 101:] * 101
                
            elif self._fusion == 'class-ld3-2':
                rgb_scores = x_rgb['FC'] * ratio_out * 101
                flow_scores = x_flow['FC'] * (1 - ratio_out) * 101
            
#                print('RGB', torch.max(x_rgb['FC'], 1)[1], '\nFLOW', 
#                      torch.max(x_flow['FC'], 1)[1], '\nW', ratio_out)
                
            if self._fusion in ['vanilla-ld3', 'vanilla-ld3-2', 'class-ld3', 'class-ld3-2']: 
                fusion_out = rgb_scores + flow_scores
                final_out['WEIGHTS'] = ratio_out

            if 'FC' in self._endpoint:
                final_out['FC'] = fusion_out
                
            if 'SCORES' in self._endpoint:
                fusion_out = self.final_softmax(fusion_out)
                final_out['SCORES'] = fusion_out
            
        return final_out
    
    def freezeAll(self, unfreeze = False):
        for params in self.parameters():
            params.requires_grad = unfreeze
            
class RelNet(nn.Module):
    
    VALID_NET = (
            'rel-vanilla'
            )
    
    VALID_ENDPOINTS = (
            'FC',
            'SCORES'
            )
    
    def __init__(self, net = 'rel-vanilla', use_bias = True, bn_momentum = 0.1, bn_epson = 1e-3, 
                 endpoint = ['FC']):
        
        super(RelNet, self).__init__()
        self._net = net
        
        assert(self._net in self.VALID_NET)
        
        # validate list of endpoint
        for endp in endpoint:
            assert endp in self.VALID_ENDPOINTS
        self._endpoint = endpoint
        
        self.relu = nn.ReLU()
        
        self.linear1 = nn.Linear(1024, 256, bias=use_bias)
        self.bn1 = nn.BatchNorm1d(256, momentum=bn_momentum, eps=bn_epson)
        
        self.linear2 = nn.Linear(256, 64, bias=use_bias)
        self.bn2 = nn.BatchNorm1d(64, momentum=bn_momentum, eps=bn_epson)
        
        self.linear3 = nn.Linear(64, 2, bias=use_bias)
        self.output = nn.Sigmoid()
        
        self.final_softmax = nn.Softmax(dim = 1)
        
    def forward(self, y_rgb, y_flow):
        """
        y_rgb, y_flow <- AP, FC
        """
        final_out = {}
        
        rgb_ap = y_rgb['AP'].reshape(y_rgb['AP'].shape[:2])
        flow_ap = y_flow['AP'].reshape(y_flow['AP'].shape[:2])
        
        rel_index = torch.cat((rgb_ap, flow_ap), dim = 1)
        
        rel_index = self.relu(self.bn1(self.linear1(rel_index)))
        rel_index = self.relu(self.bn2(self.linear2(rel_index)))
        rel_index = self.output(self.linear3(rel_index))
        
        rgb_scores = y_rgb['FC'] * rel_index[:, 0].reshape(rel_index.shape[0], 1)
        flow_scores = y_flow['FC'] * rel_index[:, 1].reshape(rel_index.shape[0], 1)
        
        if self._net == 'rel-vanilla':
            fusion_out = rgb_scores + flow_scores
            
        if 'FC' in self._endpoint:
            final_out['FC'] = fusion_out
                
        if 'SCORES' in self._endpoint:
            fusion_out = self.final_softmax(fusion_out)
            final_out['SCORES'] = fusion_out
        
        final_out['INDEX'] = rel_index
        return final_out
    
    def freezeAll(self, unfreeze = False):
        for params in self.parameters():
            params.requires_grad = unfreeze
            
class E2EFusionNet(nn.Module):
    
    def __init__(self, device, rgbnet, flownet, fusionnet, 
                 bn_momentum = 0.1, bn_epson = 1e-3, verbose = False):
        
        super(E2EFusionNet, self).__init__()
        
        self.rgbnet = rgbnet
        self.flownet = flownet
        self.fusionnet = fusionnet
        
    def forward(self, x_rgb, x_flow):
        
        rgb_ap = self.rgbnet(x_rgb)
        flow_ap = self.flownet(x_flow)
        
        fusion_out = self.fusionnet(rgb_ap, flow_ap)
        
        if self.fusionnet._fusion in ['vanilla-ld3', 'vanilla-ld3-2', 'class-ld3', 'class-ld3-2']:
            return rgb_ap['FC'], flow_ap['FC'], fusion_out
        else:
            return fusion_out
            
        def freezeAll(self, unfreeze = False):
            for params in self.parameters():
                params.requires_grad = unfreeze
        
if __name__ is '__main__':
    from train_net import mem_state
    device = torch.device('cuda:0')
    
    #rgb = R2Plus1DNet([3, 4, 6, 3], 101, device, in_channels=3, endpoint=['AP', 'FC']).to(device)
    #flow = R2Plus1DNet([3, 4, 6, 3], 101, device, in_channels=2, endpoint=['AP', 'FC']).to(device)
    #fusion = FusionNet(fusion = 'class-ld3-2', endpoint=['FC']).to(device)
    
    #model = E2EFusionNet(device, rgb, flow, fusion)
    #model.eval()
    
    model = RelNet().to(device)
    
#    x1 = torch.randn((1, 3, 8, 112, 112)).to(device)
#    x2 = torch.randn((1, 2, 8, 112, 112)).to(device)
    
#    with torch.set_grad_enabled(False):
#        s = model(x1, x2)
    
    x1 = {'AP':torch.randn((4, 512, 1, 1, 1)).to(device), 'FC':torch.randn((4, 101)).to(device)}
    x2 = {'AP':torch.randn((4, 512, 1, 1, 1)).to(device), 'FC':torch.randn((4, 101)).to(device)}
#    model = FusionNet(fusion = 'class-ld3-2', endpoint=['FC']).to(device) 
    s = model(x1, x2)
