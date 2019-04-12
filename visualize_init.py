# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:40:47 2019

@author: Juen
"""
import argparse
import torch

from network_r2p1d import R2Plus1DNet
from gbp_video_module import load_clips
from gbp import GBP
from visualize_demo_misc import plt_maps_vertical, plt_maps_horizontal, cv2_maps

parser = argparse.ArgumentParser(
        description = 'PyTorch 2.5D Action Recognition ResNet Visualization with Guided Backprop')

parser.add_argument('test_video', help = 'video name to be visualized')
parser.add_argument('test_label', help = 'label index for the testing video', type = int)
parser.add_argument('modality', help = 'modality to test on', choices = ['rgb', 'flow'])

parser.add_argument('-visflow', '--visflow', help = 'determine which direction of optical flows to visualize', choices = ['u', 'v'], default = 'u')
parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-endp', '--endpoint', help = 'module block where forprop to and backprop from', default = 'SOFTMAX')
parser.add_argument('-filter', '--filter-pos', help = 'filter chosen to be visualised', type = int, default = 0)
parser.add_argument('-nframe', '--frame-num', help = 'frame number for each testing clip', type = int, default = 8)
parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)

args = parser.parse_args()
frame_chnl = 1 if args.modality == 'flow' else 3;
if args.endpoint == 'SOFTMAX':
    filter_pos = args.test_label
else:
    filter_pos = args.filter_pos

# device where model params to be put on
device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

if frame_chnl == 1:
    p = torch.load('run3_flow.pth.tar', map_location = 'cuda:0')
else:
    p = torch.load('run3_rgb.pth.tar', map_location = 'cuda:0')

# read in the testing frames
# PATH SETTINGS
dataset_path = '../dataset/UCF-101';
if args.modality == 'rgb':
    frame_path = [dataset_path + '/' + 'ucf101_jpegs_256/jpegs_256/' + args.test_video]
else:
    frame_path = [dataset_path + '/' + 'ucf101_tvl1_flow/tvl1_flow/u/' + args.test_video, 
                  dataset_path + '/' + 'ucf101_tvl1_flow/tvl1_flow/v/' + args.test_video]

test_frame = load_clips(frame_path, 128, 171, 112, 112, args.frame_num, frame_chnl)
test_frame = torch.tensor(test_frame).requires_grad_().to(device)

# read in the UCF class labels for visualization purpose
class_f = open('mapping/UCF-101/classInd.txt', 'r')
class_raw_str = class_f.read().split('\n')[:-1]
class_label = [raw_str.split(' ')[1] for raw_str in class_raw_str]

# initialize the forward prop network
# endpoint indicates conv block at which forward prop stops
model = R2Plus1DNet(
        layer_sizes = [2, 2, 2, 2], num_classes = 101, 
        device = device, in_channels = frame_chnl, verbose = args.verbose1,
        endpoint = args.endpoint).to(device)

# load the pretrained model into memory
model.load_state_dict(p['content'][args.modality]['split1']['state_dict'], strict = False)

# retrieve gradients, scores, and saliency maps through guided backprop
gbp = GBP(model)
x_grads, scores = gbp.compute_grad(test_frame, filter_pos)
pos_sal, neg_sal = gbp.compute_saliency(x_grads)

# visualize the outputs
plt_maps_horizontal(args, test_frame, x_grads, pos_sal, neg_sal, class_label[args.test_label], flow = args.visflow, scores = scores).show()    
#plt_maps_vertical(args, test_frame, x_grads, pos_sal, neg_sal, class_label[args.test_label], flow = args.visflow, scores = scores).show()
#cv2_maps(args, test_frame, x_grads, pos_sal, neg_sal, class_label[args.test_label], flow = args.visflow, scores = scores)