# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:41:08 2019

@author: Juen
"""

import torch
import pickle

# read in caffe model
with open('r2p1d_pretrained/r2.5d_d34_l32_ft_sports1m_optical_flow.pkl', 'rb') as l8:
    ppp = pickle.load(l8, encoding='latin1')
    ppp = ppp['blobs']
    
# retrieve list of keys
keys = list(ppp.keys())

# map between caffe model and pytorch mode
model = {}
param_map = {}
#model = {'train' : {'state_dict' : {}}}

# mapping between shorcuts/downsampling modules
bn_mapping = {'b' : 'bias', 
              'rm' : 'running_mean', 
              'riv' : 'running_var', 
              's' : 'weight'}

# for 34-layer model
sc_mapping = {'3' : ['conv3_x', 'conv3_1'],
              '7' : ['conv4_x', 'conv4_1'],
              '13' : ['conv5_x', 'conv5_1']}
comp_mapping = {'0' : ['conv2_x', 'conv2_1'], 
                '1' : ['conv2_x', 'conv2_2'], 
                '2' : ['conv2_x', 'conv2_3'], 
                '3' : ['conv3_x', 'conv3_1'], 
                '4' : ['conv3_x', 'conv3_2'], 
                '5' : ['conv3_x', 'conv3_3'], 
                '6' : ['conv3_x', 'conv3_4'], 
                '7' : ['conv4_x', 'conv4_1'], 
                '8' : ['conv4_x', 'conv4_2'],
                '9' : ['conv4_x', 'conv4_3'],
                '10' : ['conv4_x', 'conv4_4'],
                '11' : ['conv4_x', 'conv4_5'],
                '12' : ['conv4_x', 'conv4_6'],
                '13' : ['conv5_x', 'conv5_1'],
                '14' : ['conv5_x', 'conv5_2'],
                '15' : ['conv5_x', 'conv5_3'],}

# for 18-layer mode
#sc_mapping = {'2' : ['conv3_x', 'conv3_1'], 
#              '4' : ['conv4_x', 'conv4_1'], 
#              '6' : ['conv5_x', 'conv5_1']}
#comp_mapping = {'0' : ['conv2_x', 'conv2_1'], 
#                '1' : ['conv2_x', 'conv2_2'], 
#                '2' : ['conv3_x', 'conv3_1'], 
#                '3' : ['conv3_x', 'conv3_2'], 
#                '4' : ['conv4_x', 'conv4_1'], 
#                '5' : ['conv4_x', 'conv4_2'], 
#                '6' : ['conv5_x', 'conv5_1'], 
#                '7' : ['conv5_x', 'conv5_2'], }
    
# mapping between state_dict keys
for p in keys:
    buf = p.split('_')
    record = False
    
    # if it is the projection shortcut
    if 'shortcut' in p:
        record = True
        # if it is shortcut conv
        if '_w' in p:
            layer = sc_mapping[p.split('_')[2]]
            pname = 'net.' + layer[0] + '.res_module.' + layer[1] + '.downsample_block.' + layer[1] + '_downsample_1x1x1_conv.weight'
        else: # else shortcut bn
            layer, param = sc_mapping[buf[2]], bn_mapping[buf[4]]
            pname = 'net.' + layer[0] + '.res_module.' + layer[1] + '.downsample_block.' + layer[1] + '_downsample_bn.' + param
    
    # if it belongs to conv1 layer
    elif 'conv1' in p:
        record = True
        # if it is the spatial conv
        if 'middle' in p:
            pname = 'net.conv1.spatial_conv.conv.spatial_'
        else: # else temporal conv
            pname = 'net.conv1.temporal_conv.conv.temporal_'
            
        # if it is a bn layer
        if 'spatbn' in p:
            pname += 'bn.' + bn_mapping[buf[3 if 'middle' not in p else 4]]
        else:
            pname += 'conv.weight'
    
    # if it belongs to other conv layers
    elif 'comp' in p: 
        record = True
        pname = 'net.' + comp_mapping[buf[1]][0] + '.res_module.' + comp_mapping[buf[1]][1] + '.res_block.'
        # if it is a conv layer
        if 'conv' in p:
            pname += comp_mapping[buf[1]][1] + '_conv' + buf[3] + '.'
            if 'middle' in p: # spatial
                pname += 'spatial_conv.conv.spatial_conv.weight'
            else: # temporal
                pname += 'temporal_conv.conv.temporal_conv.weight'
        else: # else bn layer
            if 'middle' in p: # spatial
                pname += comp_mapping[buf[1]][1] + '_conv' + buf[3] + '.spatial_conv.conv.spatial_bn.' + bn_mapping[buf[5]]
            else: # temporal
                pname += comp_mapping[buf[1]][1] + '_bn' + buf[3] + '.' + bn_mapping[buf[4]]
    
    if record:
        tensor = torch.tensor(ppp[p], dtype = torch.float, device = 'cuda:0')
        model[pname] = tensor
        param_map[p] = pname
        
#for p in keys:
#    if p in param_map.keys():
#        print(param_map[p], '\n####', p)#, '\n****', ppp[p].shape, model[param_map[p]].shape)
#              
#qmodel = torch.load('scratch_rgb.pth.tar')['train']['state_dict']
#qparam = list(qmodel.keys())
#for p in model.keys():
#    for q in qparam:
#        if p == q:
#            break
#else:
#    print(p)
        
save = {'train' : {'state_dict' : model}}
torch.save(save, 'kinetic-s1m-d34-l32-of.pth.tar')