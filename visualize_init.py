# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:40:47 2019

@author: Juen
"""
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from network_r2p1d import R2Plus1DNet
from gbp_video_module import load_clips
from gbp import GBP

parser = argparse.ArgumentParser(
        description = 'PyTorch 2.5D Action Recognition ResNet Visualization with Guided Backprop')

parser.add_argument('dataset_path', help = 'path to directories of rgb frames')
parser.add_argument('model_path', help = 'path to pretrained model')
parser.add_argument('test_video', help = 'video name to be visualized')

parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-endp', '--endpoint', help = 'module block where forprop to and backprop from', default = 'Conv3d_5_x')
parser.add_argument('-filter', '--filter-pos', help = 'filter chosen to be visualised', type = int, default = 0)
parser.add_argument('-nframe', '--frame-num', help = 'frame number for each testing clip', type = int, default = 8)
parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)

args = parser.parse_args()

# device where model params to be put on
device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
p = torch.load(args.model_path, map_location = 'cuda:0')

# read in the testing frames
frame_path = args.dataset_path + '/' + args.test_video
test_frame = load_clips(frame_path, 128, 171, 112, 112, args.frame_num)
test_frame = torch.tensor(test_frame).requires_grad_().to(device)

# read in the UCF class labels for visualization purpose
class_f = open('mapping/UCF-101/classInd.txt', 'r')
class_raw_str = class_f.read().split('\n')[:-1]
class_label = [raw_str.split(' ')[1] for raw_str in class_raw_str]

# initialize the forward prop network
# endpoint indicates conv block at which forward prop stops
model = R2Plus1DNet(
        layer_sizes = [2, 2, 2, 2], num_classes = 101, 
        device = device, in_channels = 3, verbose = args.verbose1,
        endpoint = args.endpoint).to(device)

# load the pretrained model into memory
model.load_state_dict(p['content']['rgb']['split1']['state_dict'], strict = False)

# retrieve gradients, scores, and saliency maps through guided backprop
gbp = GBP(model)
x_grads = gbp.compute_grad(test_frame, args.filter_pos)
saliency = gbp.compute_abs_saliency(x_grads)

# visualize the output


def get_prediction(scores, top_k, class_label):
    
    scores = scores.cpu().detach().numpy()
    indices = np.argsort(scores, axis = 1)[0, (101 - top_k) :][::-1]
    return {class_label[index] : scores[0, index] for index in indices}

def testvis():

    col = 3
    row = args.frame_num
    
    fig = plt.figure(figsize = (col * 4, row * 4))
    #plt.title('Original Frames', loc = 'left')
    plt.axis('off')
    for i in range(0, test_frame.shape[2] * col):
        if i % 3 == 1:
            img = np.array(x_grads[0, :, i//col, :, :]).transpose((1, 2, 0))
        elif i % 3 == 0:
            img = test_frame[0, :, i//col, :, :].cpu().detach().numpy().transpose((1, 2, 0))
        else:
            img = saliency[0, :, i//col, :, :].transpose((1, 2, 0))
            
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        fig.add_subplot(row, col, i + 1)
        plt.axis('off')
        plt.imshow(img)
    plt.subplots_adjust(hspace=0, wspace=0)
    
    return plt
    
testvis().show()