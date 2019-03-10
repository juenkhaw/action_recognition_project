# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:21:55 2019

@author: Juen
"""
import torch
import torch.nn as nn

from collections import OrderedDict

import module

class SpatioTemporalConv(nn.Module):
    """
    Module consisting of spatial and temporal convolution over input volume
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        stride : (t, h, w) striding over the input volume
        bn_mom : BN momentum hyperparameter
        bn_eps : BN epsilon hyperparameter
        inter_planes : channels of the intermediate convolution (computes with formula if set to None)
        bn_relu_first_conv : applies BN and activation on spatial conv or not
        bn_relu_second_conv : applies BN and activation on temporal conv or not
        padding : [SAME/VALID] padding technique to be applied
        use_BN : applies Batch Normalization or not
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, stride = (1, 1, 1), 
                 bn_mom = 0.1, bn_eps = 1e-3, inter_planes = None, 
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
        
        # compute the intermediate planes between spatial and temporal conv if not explicitly provided
        if inter_planes is None:
            inter_planes = int((kernel_size[0] * in_planes * out_planes * kernel_size[1] * kernel_size[2]) / 
                               (in_planes * kernel_size[1] * kernel_size[2] + kernel_size[0] * out_planes))
        
        # perform 2D Conv --> BN --> ReLU --> 1D Conv
        self.spatial_conv = module.Conv3D(in_planes, inter_planes, kernel_size = spatial_f, 
                                          stride = spatial_s, padding = padding, 
                                          use_BN = bn_relu_first_conv, bn_mom = bn_mom, bn_eps = bn_eps, 
                                          activation = bn_relu_first_conv, use_bias = use_bias, 
                                          name = 'spatial_')
        
        self.temporal_conv = module.Conv3D(inter_planes, out_planes, kernel_size = temporal_f, 
                                           stride = temporal_s, padding = padding, use_BN = bn_relu_second_conv, 
                                           bn_mom = bn_mom, bn_eps = bn_eps, 
                                           activation = bn_relu_second_conv, use_bias = use_bias, 
                                           name = 'temporal_')
        
    def forward(self, x):
        x = self.spatial_conv(x)
        #print(x.shape)
        x = self.temporal_conv(x)
        #print(x.shape)
        
        return x
    
class SpatioTemporalResBlock(nn.Module):
    """
    Residual block consists of multiple modules of spatiotemporal convolution
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        downsample : activates to apply 1*1*1 conv on shortcut to obtain volumes same as the residual
        bn_mom : BN momentum hyperparameter
        bn_eps : BN epsilon hyperparameter
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, downsample = False, 
                 bn_eps = 1e-3, bn_mom = 0.1, name = 'SpatioTemporalResBlock'):
        
        super(SpatioTemporalResBlock, self).__init__()
        
        self._downsample = downsample
        self.res_block = nn.Sequential(OrderedDict([]))
        self.relu = nn.Sequential(OrderedDict([]))
        
        if self._downsample:
            self.downsample_block = nn.Sequential(OrderedDict([
                
                #downsample x to be the same spatiotemporal dimension as output
                (name + '_downsample_1x1x1_conv', 
                 nn.Conv3d(in_planes, out_planes, [1, 1, 1], stride = [2, 2, 2], bias = False)),
                (name + '_downsample_bn',
                 nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps))
                ]))
                     
                    #downsample the residual
            self.res_block.add_module(name + '_conv1',
                SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                   stride = (2, 2, 2), bn_mom = bn_mom, bn_eps = bn_eps))

        else:
            #do not downsample the x
            self.res_block.add_module(name + '_conv1', 
                SpatioTemporalConv(in_planes, out_planes, kernel_size, padding = 'SAME', 
                                   bn_mom = bn_mom, bn_eps = bn_eps))
        
        self.res_block.add_module(name + '_bn1', nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps))
        self.res_block.add_module(name + '_relu1', nn.ReLU())
        
        self.res_block.add_module(name + '_conv2', SpatioTemporalConv(out_planes, out_planes, kernel_size, padding = 'SAME', 
                                        bn_mom = bn_mom, bn_eps = bn_eps))
        self.res_block.add_module(name + '_bn2', nn.BatchNorm3d(out_planes, momentum = bn_mom, eps = bn_eps))
        
        self.relu.add_module(name + '_relu_out', nn.ReLU())
        
    def forward(self, x):
        
        res = self.res_block(x)
        
        if self._downsample:
            x = self.downsample_block(x)
            
        #print(x.shape)
        return self.relu(x + res)
    
class SpatioTemporalResModule(nn.Module):
    """
    Residual module consists of multiple residual blocks based on the network depth
    
    Constructor requires:
        in_planes : channels of input volume
        out_planes : channels of output activations
        kernel_size : filter sizes (t, h, w)
        layer_size : repetation count of residual blocks
        block_type : type of residual block
        downsample : activates to apply 1*1*1 conv on shortcut to obtain volumes same as the residual
        name : module name
    """
    
    def __init__(self, in_planes, out_planes, kernel_size, layer_size, 
                 block_type = SpatioTemporalResBlock, downsample = False, 
                 bn_eps = 1e-3, bn_mom = 0.1, 
                 name = 'SpatioTemporalResModule'):
        
        super(SpatioTemporalResModule, self).__init__()
        
        #implement the first conv to produce activation with volume same as the output
        module_name = name[:-1] + '1'
        self.res_module = nn.Sequential(OrderedDict([
                (module_name, 
                 block_type(in_planes, out_planes, kernel_size, downsample, 
                            bn_mom = bn_mom, bn_eps = bn_eps, name = module_name))
                ]))
        
        #the rest conv operations are identical
        for i in range(layer_size - 1):
            module_name = name[:-1] + str(i + 2)
            self.res_module.add_module(module_name, 
                            block_type(out_planes, out_planes, kernel_size, 
                                       bn_mom = bn_mom, bn_eps = bn_eps, name = module_name))
            
    def forward(self, x):
        for block in self.res_module:
            x = block(x)
            
        return x
    
class R2Plus1DNet(nn.Module):
    """
    Complete network architecture of R2.5D ConvNet
    
    Constructor requires:
        layer_sizes : list of integer indicating repetation count of residual blocks at each phase
        num_classess : total label count
        device : device id to be used on training/testing
        block_type : type of residual block
        in_channels : initial channels of the input volume
        bn_momentum : BN momentum hyperparameter
        bn_epson : BN epsilon hyperparameter
        name : module name
        verbose : prints activation output size after each phases or not
    """
    
    VALID_ENDPOINTS = (
        'Conv3d_1',
        'Conv3d_2_x',
        'Conv3d_3_x',
        'Conv3d_4_x',
        'Conv3d_5_x',
        'AP',
        'FC',
        'SOFTMAX'
    )
    
    def __init__(self, layer_sizes, num_classes, device, block_type = SpatioTemporalResBlock, 
                 in_channels = 3, bn_momentum = 0.1, bn_epson = 1e-3, name = 'R2+1D', verbose = True, 
                 endpoint = 'FC'):
            
        super(R2Plus1DNet, self).__init__()
        
        self._num_classes = num_classes
        self._verbose = verbose
        if endpoint in self.VALID_ENDPOINTS:
            self._endpoint = endpoint
        else:
            self._endpoint = 'FC'
        
        self.net = nn.Sequential(OrderedDict([
                ('conv1',
                SpatioTemporalConv(in_channels, 64, kernel_size = (3, 7, 7), 
                       stride = (1, 2, 2), padding = 'SAME', inter_planes = 45, name = 'conv1', 
                       bn_mom = bn_momentum, bn_eps = bn_epson, bn_relu_second_conv = True).to(device)),
                ('conv2_x', 
                SpatioTemporalResModule(64, 64, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[0], downsample = False, name = 'conv2_x', 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device)),
                ('conv3_x', 
                SpatioTemporalResModule(64, 128, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[1], downsample = True, name = 'conv3_x', 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device)),
                ('conv4_x', 
                SpatioTemporalResModule(128, 256, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[2], downsample = True, name = 'conv4_x', 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device)),
                ('conv5_x', 
                SpatioTemporalResModule(256, 512, kernel_size = (3, 3, 3), 
                       layer_size = layer_sizes[3], downsample = True, name = 'conv5_x', 
                       bn_mom = bn_momentum, bn_eps = bn_epson).to(device)),
                ]))
        
        # Logits
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        #self.linear = nn.Linear(512, num_classes)
        self.linear1 = nn.Linear(512, num_classes)
        
        self.softmax = nn.Softmax(dim = 1)
        

    def replaceLinear(self, num_classes):
        """
        Replaces the FC linear layer with updated label count
        
        Inputs:
            num_classess : updated label count
            
        Returns:
            None
        """
        
        self._num_classes = num_classes
        self.linear1 = nn.Linear(512, num_classes)        
        
    def forward(self, x):
        
        # perform each module until reaching final endpoint
        if self._verbose:
            print('Input', x.shape)
            
        for i, module in enumerate(self.net):
            x = self.net[i](x)
            if self._verbose:
                print(self.VALID_ENDPOINTS[i], x.shape)
            if self._endpoint == self.VALID_ENDPOINTS[i]:
                return x
        
        # pre-fc
        x = self.avgpool(x)
        x = x.view(-1, 512)
        if self._verbose:
            print('Pre FC', x.shape)
            
        if self._endpoint == 'AP':
            return x
        
        # fc linear layer
        x = self.linear1(x)
        if self._verbose:
            print('Post FC', x.shape)
        
        if self._endpoint == 'SOFTMAX':
            return self.softmax(x)
        else:
            return x

if __name__ is '__main__':
    device = torch.device('cpu')
    model = R2Plus1DNet(layer_sizes = [2, 2, 2, 2], num_classes = 101, device = device, in_channels = 3, verbose = True).to(device)
    
    from torchsummary import summary
    summary(model, (3, 8, 112, 112), batch_size = 2, device = "cpu")
    
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