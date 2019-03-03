# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:40:47 2019

@author: Juen
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from network_r2p1d import R2Plus1DNet
from video_module import load_clips
from torchsummary import summary

device = torch.device('cuda:0')
p = torch.load('juen/run3_rgb.pth.tar', map_location = 'cuda:0')

#kernels = []
#
#for key, value in p['content']['rgb']['split1']['state_dict'].items():
#    if True:
#        print(key)
#        print('####', value.shape)
        #kernels.append(value)
#
#col = 10
#row = 45 // col + 1
#fig = plt.figure(figsize = (col, row))
#for i in range(0, kernels[0].shape[0]):
#    img = np.array(kernels[0][i].cpu())
#    img = img.reshape(-1, 7, 7).transpose((1, 2, 0))
#    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
#    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    fig.add_subplot(row, col, i + 1)
#    plt.axis('off')
#    plt.imshow(img)
#
#plt.subplots_adjust(hspace=0.1, wspace=0.1)
#plt.show()

# initialize the forward prop network
# endpoint indicates conv block at which forward prop stops
model = R2Plus1DNet(
        layer_sizes = [2, 2, 2, 2], num_classes = 101, 
        device = device, in_channels = 3, verbose = True, endpoint = 'Conv3d_1').to(device)
#summary(model, (3, 8, 112, 112))

# load the pretrained model into memory
model.load_state_dict(p['content']['rgb']['split1']['state_dict'], strict = False)

# read in the testing frames
frame_path = r'..\dataset\UCF-101\ucf101_jpegs_256\jpegs_256\v_ApplyEyeMakeup_g01_c01'
test_frame = load_clips([frame_path], 'rgb', 128, 171, 112, 112, 8, mode = 'video', clips_per_video = 2)
test_frame = torch.tensor(test_frame).to(device)

# retreive features 
model.eval()
feature_map = model(test_frame)

# retrieve only activations of the first clip
feature_map = feature_map[0, :, :, :, :]

# retrieve top 9 most activated feature maps
top_k = 9
activation_sum = feature_map.sum(dim = (1, 2, 3))
top_index = activation_sum.argsort(dim = 0, descending = True)[:top_k]

# convert feature maps back to 5d tensor
feature_map = torch.unsqueeze(feature_map, 0)

from deconvnet import SpatioTemporalDeconv
from collections import OrderedDict

# initialize the back prop deconvnet network
de_model = SpatioTemporalDeconv(64, 3, (3, 7, 7), stride = (1, 2, 2), 
                                 inter_planes = 45, temporal_relu = True).to(device)
#feature_map = feature_map.to(device)

# extract keys for pretrained conv layer params
conv_params_keys = [
        'net.conv1.spatial_conv.conv.spatial_conv.weight',
        'net.conv1.temporal_conv.conv.temporal_conv.weight'
        ]

# extract keys for deconv layer params
deconv_params_keys = list(de_model.state_dict().keys())

# length of conv params and deconv params must be the same
assert(len(conv_params_keys) == len(deconv_params_keys))

# extract the conv pretrained params and prepare the deconv params dict
conv_params = p['content']['rgb']['split1']['state_dict']
deconv_params = OrderedDict({})

# exchange in and out channels
# flip the params of all 3d filters
for i in range(len(conv_params_keys)):
    p = conv_params[conv_params_keys[i]]
    p = torch.transpose(p, 0, 1)
    p = torch.flip(p, [2, 3, 4])
    deconv_params[deconv_params_keys[i]] = p

# load the modified params into deconvnet
de_model.load_state_dict(deconv_params)

# display original frame images
col = 8
row = 1
#row = vis.shape[2] // col + 1

fig = plt.figure(figsize = (col * 2, row * 2))
plt.title('Original Frames', loc = 'left')
plt.axis('off')
for i in range(0, test_frame.shape[2]):
    img = np.array(test_frame[0, :, i, :, :].cpu()).transpose((1, 2, 0))
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    fig.add_subplot(row, col, i + 1)
    plt.axis('off')
    plt.imshow(img)
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.show()

# compute feature input volume of deconvnet
s = feature_map.shape
feature_x = torch.empty(top_k, s[1], s[2], s[3], s[4])

for i in range(top_k):
    map_x = feature_map.clone()
    for k in range(0, feature_map.shape[1]):
        if k != top_index[i].item():
            map_x[0, k, :, :, :] = 0
    feature_x[i, :, :, :, :] = map_x[0, :, :, :, :].clone()
    
de_model.eval()
feature_x = feature_x.to(device)
with torch.set_grad_enabled(False):
    vis = de_model(feature_x)
    
for i in range(vis.shape[0]):
    fig = plt.figure(figsize = (col * 2, row * 2))
    title = 'CONV_1 Feature Map [' + str(top_index[i].item()) + ']'
    plt.title(title, loc = 'left')
    plt.axis('off')
    for j in range(0, vis.shape[2]):
        img = np.array(vis[i, :, j, :, :].cpu()).transpose((1, 2, 0))
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        fig.add_subplot(row, col, j + 1)
        plt.axis('off')
        plt.imshow(img)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.show()

## for each top activations, visualize it
#for i in range(top_k):
#    cur_act = top_index[i]
#    
#    # make all activations to 0 except for current k activation
#    map_x = feature_map.clone()
#    for k in range(0, feature_map.shape[1]):
#        if k != cur_act.item():
#            map_x[0, k, :, :, :] = 0
#    
#    # conpute the visualized matrices with deconvnet
#    de_model.eval()
#    with torch.set_grad_enabled(False):
#        vis = de_model(map_x)
#        
#    # display the visualized image of current activation
#    fig = plt.figure(figsize = (col * 2, row * 2))
#    title = 'CONV_1 Feature Map [' + str(cur_act.item()) + ']'
#    plt.title(title, loc = 'left')
#    plt.axis('off')
#    for j in range(0, vis.shape[2]):
#        img = np.array(vis[0, :, j, :, :].cpu()).transpose((1, 2, 0))
#        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
#        fig.add_subplot(row, col, j + 1)
#        plt.axis('off')
#        plt.imshow(img)
#    plt.subplots_adjust(hspace=0.1, wspace=0.1)
#    plt.show()
