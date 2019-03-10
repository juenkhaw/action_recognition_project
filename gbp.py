# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:25:34 2019

@author: Juen
"""

import torch
import torch.nn as nn

from network_r2p1d import R2Plus1DNet

class GBP(object):
    """
    Class containing guided backprop visualization tools
    
    Constructor requires:
        model : network module to be modified to perform guided backprop
    """
    
    def init_hook_input_space(self):
        """
        register a hook function at the conv module that is nearest to input space
        to retrieve the gradient map flowing to the input layer
        """
        
        def hook_first_layer(module, grad_input, grad_output):
            # expect output grads has shape of (1, 3, t, h, w)
#            print('hello, i ve been through here')
#            print(module)
#            for i in range(len(grad_input)):
#                if grad_input[i] is not None:
#                    print(i, grad_input[i].shape)
#            print(grad_output[0].shape)
            self.output_grads = grad_input[0]
            
        first_block = None
        # finding the first conv module
        for module in self.model.modules():
            if isinstance(module, nn.Conv3d):
                first_block = module
                break
            
        # assign the hook function onto the first conv module
        first_block.register_backward_hook(hook_first_layer)
        
    def init_hook_relu(self):
        """
        register hook functions in all relu modules throughout the network
        forward prop : to store the output activation map as buffer and to be used in backprop
        backward prop : to zero out the upstream gradient of dead neurons based on the corresponding 
                        activation map stored during forward prop
        """
        
        def hook_relu_forward(module, input, output):
            # store the relu output of current layer
            # to be used as mask to zero out the upstream gradient
            self.forward_relu_outputs.append(output)
            
        def hook_relu_backward(module, grad_input, grad_output):

            # create gradient mask on corresponding relu output
            # prevent gradient to flow through dead neurons
            grad_mask = (self.forward_relu_outputs.pop() > 0)
            
            # zeroing out upstream gradients with negative value
            grad = grad_input[0]
            grad[grad < 0] = 0
            
            # back prop
            grad_mask = grad_mask.type_as(grad)
            return (grad_mask * grad, )
        
        # registering relu hook functions
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(hook_relu_forward)
                module.register_backward_hook(hook_relu_backward)
    
    def __init__(self, model):
        
        self.model = model
        self.output_grads = None
        self.forward_relu_outputs = []
        
        self.model.eval()
        self.init_hook_input_space()
        self.init_hook_relu()
        
    def compute_grad(self, input, filter_pos):
        """
        Feeds network with testing input volume and retrieves the input space gradient
        
        Inputs:
            input : testing volume of clip
            filter_pos : selection of a kernel activation map in the top (nearest to output) module
                
        Outputs:
            input sapce gradient map
        """
        
        self.model.zero_grad()
        
        # expected output has shape of (1, chnl, t, h, w)
        output = self.model(input)
        
        # zero out rest of the neurons activation map
        output = torch.sum(torch.abs(output[0, filter_pos]))
        
        # backprop
        output.backward()
        
        return self.output_grads.cpu().detach().numpy()
        
        
if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = R2Plus1DNet(layer_sizes = [2, 2, 2, 2], num_classes = 101, 
                        device = device, in_channels = 3, 
                        verbose = True, endpoint = 'Conv3d_5_x').to(device)
    
    gbp = GBP(model)
    
    x = torch.randn((1, 3, 8, 112, 112)).requires_grad_().to(device)
    
    gbp.compute_grad(x, 0)
        

#def compute_pad(dim_size, kernel_size, stride):
#    """
#    Dynamically computes padding for each dimension of the input volume
#    
#    Inputs:
#        dim_size : dimension (w, h, t) of the input volume
#        kernel_size : dimension (w, h, t) of kernel
#        stride : stride applied for each dimension
#        
#    Returns:
#        list of 6 ints with padding on both side of all dimensions
#    """
#    pads = []
#    for i in range(len(dim_size) - 1, -1, -1):
#        if dim_size[i] % stride[i] == 0:
#            pad = max(kernel_size[i] - stride[i], 0)
#        else:
#            pad = max(kernel_size[i] - (dim_size[i] % stride[i]), 0)
#        pads.append(pad // 2)
#        pads.append(pad - pad // 2)
#    
#    return pads
#
#class SpatioTemporalDeconv(nn.Module):
#    
#    def __init__(self, out_planes, in_planes, kernel_size, stride = (1, 1, 1), 
#                 inter_planes = None, temporal_relu = False, spatial_relu = True, 
#                 padding = 'SAME', use_bias = False, name = 'SpatioTemporalDeconv'):
#        
#        super(SpatioTemporalDeconv, self).__init__()
#        
#        self._kernel_size = kernel_size
#        self._stride = stride
#        self._padding = padding
#        self._t_relu = temporal_relu
#        self._s_relu = spatial_relu
#        
#        if inter_planes is None:
#            inter_planes = int((kernel_size[0] * in_planes * out_planes * kernel_size[1] * kernel_size[2]) / 
#                               (in_planes * kernel_size[1] * kernel_size[2] + kernel_size[0] * out_planes))
#        
#        if temporal_relu:
#            self.temporal_relu = nn.ReLU()
#        self.temporal_deconv = nn.Conv3d(out_planes, inter_planes, kernel_size = (kernel_size[0], 1, 1), 
#                                         stride = (stride[0], 1, 1), padding = (0, 0, 0), bias = use_bias)
#        
#        if spatial_relu:
#            self.spatial_relu = nn.ReLU()
#        self.spatial_deconv = nn.Conv3d(inter_planes, in_planes, kernel_size = (1, kernel_size[1], kernel_size[2]), 
#                                         stride = (1, stride[1], stride[2]), padding = (0, 0, 0), bias = use_bias)
#        
#    def forward(self, x):
#        
#        if self._padding == 'SAME':
#            x = F.pad(x, compute_pad(x.shape[2:], self._kernel_size, self._stride))
#        
#        if self._t_relu:
#            x = self.temporal_relu(x)
#        x = self.temporal_deconv(x)
#        if self._s_relu:
#            x = self.spatial_deconv(x)
#            
#        return x
    
#class SpatioTemporalResDeBlock(nn.Module):
#    
#    def __init__(self, out_planes, in_planes, kernel_size, downsample = False, 
#                 name = 'SpatioTemporalReDeBlock'):
#        
#        super(SpatioTemporalResDeBlock, self).__init__()
#        
#        self._downsample = downsample
##        self.relu = nn.Sequential(OrderedDict([]))
##        self.res_block = nn.Sequential(OrderedDict([]))
#        
#        self.res_deconv_block = nn.Sequential(OrderedDict([
#                (name + '_relu1', nn.ReLU()),
#                (name + '_deconv1', SpatioTemporalDeconv(out_planes, out_planes, kernel_size, 
#                                                         padding = 'SAME')),
#                (name + '_relu2', nn.ReLU())
#            ]))
#        if downsample:
#            self.res_deconv_block.add_module(name + '_deconv2', SpatioTemporalDeconv(out_planes))

#class DeConvNet(nn.Module):
#    
#    def __init__(self):
#        
#        super(DeConvNet, self).__init__()
#        
#        self.spatial_deconv = nn.Conv3d(45, 3, kernel_size = (1, 7, 7), 
#                                        stride = (1, 2, 2), padding = (0, 3, 3), bias = False)
#        self.temporal_deconv = nn.Conv3d(64, 45, kernel_size = (3, 1, 1), 
#                                        stride = 1, padding = (1, 0, 0), bias = False)
#        self.relu = nn.ReLU()
#        
#    def forward(self, x):
#        x = self.relu(x)
#        x = self.temporal_deconv(x)
#        x = self.relu(x)
#        x = self.spatial_deconv(x)
#        
#        return x
#    
#if __name__ == '__main__':
#    model = SpatioTemporalDeconv(64, 3, (3, 7, 7), stride = (1, 2, 2), 
#                                 inter_planes = 45, temporal_relu = True)
#    x = torch.randn((1, 64, 8, 56, 56))
#    y = model(x)
