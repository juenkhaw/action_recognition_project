# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:31:41 2019

@author: Juen
"""
import argparse
import torch
import numpy as np
import cv2

from video_module import load_clips, denormalize_buffer
from network_r2p1d import R2Plus1DNet

parser = argparse.ArgumentParser(description = 'fyp1 demo')
parser.add_argument('test_video', help = 'video name to be visualized')
parser.add_argument('test_label', help = 'label index for the testing video', type = int)
parser.add_argument('-save', '--save', help = 'whether to store image', action = 'store_true', default = False)
args = parser.parse_args()

# define device
device = torch.device('cuda:0')

# dataset PATH SETTINGS
rgb_path = '../dataset/UCF-101/ucf101_jpegs_256/jpegs_256/' + args.test_video
flow_u_path = '../dataset/UCF-101/ucf101_tvl1_flow/tvl1_flow/u/' + args.test_video
flow_v_path = '../dataset/UCF-101/ucf101_tvl1_flow/tvl1_flow/v/' + args.test_video

# pre-trained model and stream weights PATH SETTINGS
model_state = torch.load('fyp1_demo.pth.tar')

# reading label index
class_f = open('mapping/UCF-101/classInd.txt', 'r')
class_raw_str = class_f.read().split('\n')[:-1]
class_label = [raw_str.split(' ')[1] for raw_str in class_raw_str]
class_label = np.array(class_label)

# define testing input and ground truth label
rgb_test_X = load_clips([rgb_path], 'rgb', 128, 171, 112, 112, 8, mode = 'video', clips_per_video = 10)
flow_test_X = load_clips([flow_u_path, flow_v_path], 'flow', 128, 171, 112, 112, 8, mode = 'video', clips_per_video = 10)
test_y = args.test_label
test_y_label = class_label[test_y]

# place the input into gpu
rgb_test_X = torch.tensor(rgb_test_X).to(device)
flow_test_X = torch.tensor(flow_test_X).to(device)

# define rgb and flow model
rgb_net = R2Plus1DNet([2,2,2,2], 101, device, endpoint=['SOFTMAX']).to(device)
flow_net = R2Plus1DNet([2,2,2,2], 101, device, in_channels=1, endpoint=['SOFTMAX']).to(device)

# load model state
rgb_net.load_state_dict(model_state['rgb_state'])
flow_net.load_state_dict(model_state['flow_state'])
stream_weights = model_state['stream_weights'].detach().cpu().numpy().ravel()
del model_state

# set models to evaluation mode
rgb_net.eval()
flow_net.eval()

# predict on test_X
with torch.set_grad_enabled(False):
    rgb_out = rgb_net(rgb_test_X)['SOFTMAX'].cpu().numpy()
    del rgb_net
    flow_out = flow_net(flow_test_X)['SOFTMAX'].cpu().numpy()

# expand dimension of scores
rgb_out = np.expand_dims(rgb_out, 0)
flow_out = np.expand_dims(flow_out, 0)

# averaging fusion
avg_out = (rgb_out + flow_out) / 2

# modality-specific weighted fusion
mwf_out = stream_weights[0] * rgb_out + stream_weights[1] * flow_out

# concatenate all scores and process to get predicted label
scores = np.concatenate((rgb_out, flow_out, avg_out, mwf_out))
scores = np.average(scores, axis = 1)
predicted = np.argsort(scores, axis = 1)[:, ::-1][:, :5]
label = class_label[predicted]

# visualize the things
# define metrics
img_h = int(112 * 1.3)
img_w = int(112 * 1.3)
graph_h = 120
graph_w = 250
bar_h = 20
bar_w = 200
padding = 10
title_h = 40 + padding * 2
out_h = graph_h * 4 + padding * 4 + title_h
out_w = img_w + graph_w + padding * 3

output = np.ones((out_h, out_w, 3), np.uint8) * 255

# extract sample rgb and flow frames
rgb_img = rgb_test_X[5, :, 3].cpu().numpy().transpose((1, 2, 0))
rgb_img = cv2.resize(denormalize_buffer(rgb_img), (img_h, img_w))
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

u_flow_img = flow_test_X[5, :, 6].cpu().numpy().transpose((1, 2, 0))
u_flow_img = cv2.resize(denormalize_buffer(u_flow_img), (img_h, img_w))
u_flow_img = np.expand_dims(u_flow_img, axis = 2)

v_flow_img = flow_test_X[5, :, 7].cpu().numpy().transpose((1, 2, 0))
v_flow_img = cv2.resize(denormalize_buffer(v_flow_img), (img_h, img_w))
v_flow_img = np.expand_dims(v_flow_img, axis = 2)

# insert sample frames
cv2.putText(output, test_y_label + ' (' + str(args.test_label + 1) + ')', 
            (padding, padding * 2), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
cv2.putText(output, '(' + args.test_video + ')', 
            (padding, padding * 2 + 15), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
np.copyto(output[title_h : title_h + img_h, 
                 padding : padding + img_w, :], rgb_img)
np.copyto(output[title_h + img_h + padding : title_h + img_h * 2 + padding, 
                 padding : padding + img_w, :], u_flow_img)
np.copyto(output[title_h + img_h * 2 + padding * 2 : title_h + img_h * 3 + padding * 2, 
                 padding : padding + img_w, :], v_flow_img)
    
# plotting prediction result
method = ['RGB stream', 'Optical Flow stream', 'Averaging', 'Weighted Fusion Network']
# blue, green, red
colors = [(255, 165, 109), (109, 255, 140), (119, 119, 255)]
    
for i in range(4):
    cv2.putText(output, method[i], (padding * 2 + img_w, title_h + padding * i + graph_h * i), 
                cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
    for j in range(5):
        cv2.rectangle(output, 
                      (padding * 2 + img_w, title_h + padding * (i - 1) + graph_h * i + bar_h * (j + 1) + 2 - 5),
                      (padding * 2 + img_w + int(scores[i, predicted[i, j]] * bar_w),
                       title_h + padding * (i - 1) + graph_h * i + bar_h * (j + 1) + bar_h - 2 - 5),
                       colors[1 if predicted[i, j] == args.test_label else 0 if j == 0 else 2], 
                       cv2.FILLED)
        cv2.putText(output, label[i, j], 
                    (padding * 2 + img_w, title_h + padding * i + graph_h * i + bar_h * (j + 1)), 
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
        cv2.putText(output, str(scores[i, predicted[i, j]])[:7], 
                    (out_w - 50, title_h + padding * i + graph_h * i + bar_h * (j + 1)), 
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0))
        
if args.save:
    cv2.imwrite('demo_result/' + args.test_video + '.png', output)

cv2.imshow('RESULT', output)
cv2.waitKey(0)

cv2.destroyAllWindows()