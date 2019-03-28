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
    """
    [Deprecated]
    Fusion network that incorporates with two stream networks from different modalities, it contains various
    type of fusion method to fuse outputs from stream networks and produce a final score/prediction
    
    Constructor requires:
        layer_sizes : list of integer indicating repetation count of residual blocks at each phase
        num_classess : total label count
        device : device id to be used on training/testing
        network : type of stream network
        fusion : fusion method to be used, averaging or modality-wf
        endpoint : list of endpoints on the network where output would be returned
        bn_momentum : BN momentum hyperparameter
        bn_epson : BN epsilon hyperparameter
        name : module name
        verbose : prints activation output size after each phases or not
        load_pretrained_stream : whether to load individually pre-trained stream network state
        load_pretrained_fusion : indicating whether the entire fusion network is loaded with previous
        state in main function
    """
    
    in_channels = {'rgb': 3, 'flow': 1}
    
    def __init__(self, layer_sizes, num_classes, device, network = R2Plus1DNet, 
                 fusion = 'average', endpoint = ['FC'], 
                 bn_momentum = 0.1, bn_epson = 1e-3, name = 'R2+1D', verbose = False, 
                 load_pretrained_stream = False, load_pretrained_fusion = False):
        
        super(FusionNet, self).__init__()
        
        self._num_classes = num_classes
        self._fusion = fusion
        
        # endpoint should be valid and defined in the R(2+1)D network
        assert(endpoint in network.VALID_ENDPOINTS)
        
        endpoint = ['FC'] if fusion == 'average' else ['AP', 'FC']
        
        # define rgb and flow stream network
        self.rgb_net = network(layer_sizes, num_classes, device, in_channels = self.in_channels['rgb'], 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, endpoint = endpoint, 
                               name = 'R2P1D_RGB', verbose = verbose)
        
        self.flow_net = network(layer_sizes, num_classes, device, in_channels = self.in_channels['flow'], 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, endpoint = endpoint,
                               name = 'R2P1D_FLOW', verbose = verbose)
        
        # define stream weightages
        self.stream_weights = None
        
        # define fusion architecture for madality-wf
        if self._fusion == 'modality-wf':
            self.linear = nn.Linear(1024, 2)
            self.softmax = nn.Softmax(dim = 1)
            
        # REFACTOR REQUIRED!!!
        # load pre-trained stream model 
        if load_pretrained_stream:
            
            print('ENTERED EASTER EGG OF PRELOADING!!')
            
            # if it is not loaded in main function from previous fusion network state
            if not load_pretrained_fusion:
                
                # stream network preloading
                content = torch.load('run1_avgfusion.pth.tar', map_location = {'cuda:2' : gpu_name})
                state = content['content']['2-stream']['split1']['state_dict']
                self.load_state_dict(state, strict = False)
                
#                # copy weights from stream linear to local linear
#                local_rgb_linear = OrderedDict({
#                        'weight' : state['rgb_net.linear1.weight'],
#                        'bias' : state['rgb_net.linear1.bias'],
#                        })
#                local_flow_linear = OrderedDict({
#                        'weight' : state['flow_net.linear1.weight'],
#                        'bias' : state['flow_net.linear1.bias'],
#                        })
#                self.linear1_rgb.load_state_dict(local_rgb_linear)
#                self.linear1_flow.load_state_dict(local_flow_linear)
                
            # freeze!
#            for params in self.linear1_rgb.parameters():
#                params.requires_grad = False
#            for params in self.linear1_flow.parameters():
#                params.requires_grad = False
            
            for params in self.rgb_net.parameters():
                params.requires_grad = False
            for params in self.flow_net.parameters():
                params.requires_grad = False  
        
    def forward(self, x_rgb, x_flow):
        
        # get (list of) outputs from each stream networks
        rgb_out = self.rgb_net(x_rgb)
        flow_out = self.flow_net(x_flow)
        
        # averaging method
        if self._fusion == 'average':
            return (rgb_out['SCORES'] + flow_out['SCORES']) / 2
        
        # modality weighted fusion method
        if self._fusion == 'modality-wf':
            
            # if the model is in training mode
            if self.training:
                
                # taking the average of final feature activations over samples for each modalities
                rgb_ap = torch.mean(rgb_out['AP'], 0, True).reshape((1, -1))
                flow_ap = torch.mean(flow_out['AP'], 0, True).reshape((1, -1))
                
                # concat the averaged activations from both modalities
                out = torch.cat((rgb_ap, flow_ap), dim = 1)
                
                # linear layers for weights
                out = self.linear(out)
                
                # softmax to make sure total of all weights = 1
                weights = self.softmax(out)
                self.stream_weights = weights
                
                #print(weights)
                
                # return scores with learned weights
                return rgb_out['SCORES'] * weights[0, 0] + flow_out['SCORES'] * weights[0, 1]
            
            # not letting stream weights to change over testing samples
            else:
                return rgb_out['SCORES'] * self.stream_weights[0, 0] + flow_out['SCORES'] * self.stream_weights[0, 1]
            
class FusionNet2(nn.Module):
    """
    Revised version of fusion network that is capable of loading stream networks with individually trained model
    state, and with standard practice of backward propagation, with freezing of stream networks and supports
    backward on several endpoints instead of just on the last point where the final score is produced
    
    Constructor requires:
        layer_sizes : list of integer indicating repetation count of residual blocks at each phase
        num_classess : total label count
        device : device id to be used on training/testing
        network : type of stream network
        fusion : fusion method to be used, averaging or modality-wf
        endpoint : list of endpoints on the network where output would be returned
        bn_momentum : BN momentum hyperparameter
        bn_epson : BN epsilon hyperparameter
        name : module name
        verbose : prints activation output size after each phases or not
        load_pretrained_stream : whether to load individually pre-trained stream network state
        load_fusion_state : indicating whether the entire fusion network is loaded with previous
        state in main function
    """
    
    in_channels = {'rgb': 3, 'flow': 1}
    
    def __init__(self, layer_sizes, num_classes, device, network = R2Plus1DNet, 
                 fusion = 'average', endpoint = ['FC'], 
                 bn_momentum = 0.1, bn_epson = 1e-3, name = 'R2+1D', verbose = False, 
                 load_pretrained_stream = False, load_fusion_state = False):
        
        super(FusionNet2, self).__init__()
        
        self._num_classes = num_classes
        self._fusion = fusion
        
        # endpoint should be defined in the R(2+1)D network
        assert(endpoint in network.VALID_ENDPOINTS)
        
        # returns outputs after FC if using averaging
        # else, for modality weighted fusion, returns at points after global avg pooling and FC
        endpoint = ['FC'] if fusion == 'average' else ['AP', 'FC']
        
        # define stream networks
        self.rgb_net = network(layer_sizes, num_classes, device, in_channels = self.in_channels['rgb'], 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, endpoint = endpoint, 
                               name = 'R2P1D_RGB', verbose = verbose)
        
        self.flow_net = network(layer_sizes, num_classes, device, in_channels = self.in_channels['flow'], 
                               bn_momentum = bn_momentum, bn_epson = bn_epson, endpoint = endpoint,
                               name = 'R2P1D_FLOW', verbose = verbose)
        
        # define stream weightages
        self.stream_weights = None
        self.pretrained_streams = load_pretrained_stream
        
        # define architecture of modality weighted fusion
        if self._fusion == 'modality-wf':
            self.linear = nn.Linear(1024, 2)
            self.softmax = nn.Softmax(dim = 1)
        
        # if loading pretrained stream networks
        if load_pretrained_stream:
            
            print('PRELOADING STREAM NETWORKS!!')
            
            # if pretrained stream networks are not loaded from previous fusion net state
            if not load_fusion_state:
            
                # stream network preloading
                rgb_state = torch.load('run3_rgb.pth.tar', map_location = {'cuda:2' : gpu_name})['content']['rgb']['split1']['state_dict']
                flow_state = torch.load('run3_flow.pth.tar', map_location = {'cuda:2' : gpu_name})['content']['flow']['split1']['state_dict']
                
                self.rgb_net.load_state_dict(rgb_state)
                self.flow_net.load_state_dict(flow_state)
            
            self.freeze_stream()
            
    def freeze_stream(self, unfreeze = False):
        """
        Toggle freezing of stream networks
        
        Inputs:
            unfreeze : set to True to unfreeze
            
        Returns:
            None
        """
        
        for params in self.rgb_net.parameters():
            params.requires_grad = unfreeze
            
        for params in self.flow_net.parameters():
            params.requires_grad = unfreeze
            
    def forward(self, x_rgb, x_flow):
        
        final_out = {}
        
        # retrieve respective (list of) outputs from stream networks
        rgb_out = self.rgb_net(x_rgb)
        flow_out = self.flow_net(x_flow)
        
        # scores generated from individual stream networks are to be returned
        final_out['RGB_SCORES'] = rgb_out['SCORES']
        final_out['FLOW_SCORES'] = flow_out['SCORES']
        
        # averaging method
        if self._fusion == 'average':
            final_out['FUSION_SCORES'] = (rgb_out['SCORES'] + flow_out['SCORES']) / 2
        
        # modality-specific weighted fusion network
        elif self._fusion == 'modality-wf':
            
            # if the model is in training mode
            if self.training:
                
                # taking the average of final feature activations over samples for each modalities
                rgb_ap = torch.mean(rgb_out['AP'], 0, True).reshape((1, -1))
                flow_ap = torch.mean(flow_out['AP'], 0, True).reshape((1, -1))
                
                # concat the averaged activations from both modalities
                out = torch.cat((rgb_ap, flow_ap), dim = 1)
                
                # linear layers for weights
                out = self.linear(out)
                
                # softmax to make sure total of all weights = 1
                weights = self.softmax(out)
                self.stream_weights = weights
                
                # return scores with learned weights
                final_out['FUSION_SCORES'] = rgb_out['SCORES'] * weights[0, 0] + flow_out['SCORES'] * weights[0, 1]
            
            # not letting stream weights to change over testing samples
            else:
                final_out['FUSION_SCORES'] = rgb_out['SCORES'] * self.stream_weights[0, 0] + flow_out['SCORES'] * self.stream_weights[0, 1]         
        
        # return list of outputs
        return final_out
        
if __name__ is '__main__':
    device = torch.device('cuda:0')
    model = FusionNet2(layer_sizes = [2, 2, 2, 2], num_classes = 101, device = device, 
                      endpoint = ['FC'], fusion = 'average').to(device)
    
    x1 = torch.randn((1, 3, 8, 112, 112)).to(device)
    x2 = torch.randn((1, 1, 16, 112, 112)).to(device)
    
    s = model(x1, x2)
