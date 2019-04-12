# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:25:34 2019

@author: Juen
"""

import torch
import torch.nn as nn
import numpy as np

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
        
        self.model.eval()            
        self.model.zero_grad()
        
        # expected output has shape of (1, chnl, t, h, w)
        output = self.model(input)
        
        # zero out rest of the neurons activation map
        print(output.shape)
        print(filter_pos)
        print(output[0, filter_pos])
        print(len(output.shape))
        if len(output.shape) > 2:
            activation = torch.sum(torch.abs(output[0, filter_pos]))
        else:
            activation = output[0, filter_pos]
        print(activation.shape)
        # backprop
        activation.backward()
        
        if self.model._endpoint == 'SOFTMAX':
            return self.output_grads.cpu().detach().numpy(), output[0, filter_pos].data.cpu().numpy()
        else:
            return self.output_grads.cpu().detach().numpy(), None
    
    def compute_saliency(self, gradient):
        """
        Computed saliency maps for magnitude in both direction
        
        Inputs:
            gradient : gradient maps generated from guided backprop
            
        Returns:
            pos_saliency : numpy volume of normalized positive gradient maps
            neg_saliency : numpy volume of normalized negative gradient maps
        """
        
        pos_saliency = np.maximum(0, gradient) / np.max(gradient)
        neg_saliency = np.abs(np.maximum(0, -1 * gradient) / np.min(-1 * gradient))
        
        return pos_saliency, neg_saliency
              
#if __name__ == '__main__':
#    device = torch.device('cuda:0')
#    model = R2Plus1DNet(layer_sizes = [2, 2, 2, 2], num_classes = 101, 
#                        device = device, in_channels = 1, 
#                        verbose = True, endpoint = 'Conv3d_5_x').to(device)
#    
#    gbp = GBP(model)
#    
#    x = torch.randn((1, 1, 16, 112, 112)).requires_grad_().to(device)
#    
#    gbp.compute_grad(x, 0)
        
