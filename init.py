# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:45:34 2019

@author: Juen
"""
import torch
import argparse
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import VideoDataset
from network_r2p1d import R2Plus1DNet
from module import msra_init
from train_net import train_model

parser = argparse.ArgumentParser(description = 'PyTorch 2.5D Action Recognition ResNet')

parser.add_argument('dataset', help = 'video dataset to be trained and validated', choices = ['ucf', 'hmdb'])
parser.add_argument('modality', help = 'modality to be trained and validated', choices = ['rgb', 'flow'])
parser.add_argument('-cl', '--clip-length', help = 'initial temporal length of each video training input', default = 8, type = int)
parser.add_argument('-sp', '--split', help = 'dataset split selected in training and evaluating model (0 to load all spilts at once)', default = 0, choices = list(range(4)), type = int)
parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-ld', '--layer-depth', help = 'depth of the resnet', default = 18, choices = [18, 34], type = int)
parser.add_argument('-ep', '--epoch', help = 'number of epochs for training process', default = 45, type = int)
parser.add_argument('-bs', '--batch-size', help = 'number of labelled sample for each batch', default = 32, type = int)
parser.add_argument('-lr', '--learning-rate', help = 'initial learning rate (alpha) for updating parameters', default = 0.01, type = float)
parser.add_argument('-ss', '--step-size', help = 'decaying lr for each [ss] epoches', default = 10, type = int)
parser.add_argument('-gm', '--lr-decay', help = 'lr decaying rate', default = 0.1, type = float)
parser.add_argument('-tm', '--test-mode', help = 'activate test mode to minimize dataset for debugging purpose', action = 'store_true', default = False)
parser.add_argument('-tc', '--test-amt', help = 'number of labelled samples to be left when test mode is activated', default = 2, type = int)
parser.add_argument('-wn', '--worker-num', help = 'number of workers for some processes (safer to set at 0; -1 set as number of device)', default = 0, type = int)
parser.add_argument('-mo', '--bn-momentum', help = 'momemntum for batch normalization', default = 0.1, type = float)
parser.add_argument('-es', '--bn-epson', help = 'epson for batch normalization', default = 1e-3, type = float)
parser.add_argument('-va', '--validation-mode', help = 'activate validation mode', action = 'store_true', default = False)

parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)
parser.add_argument('-v2', '--verbose2', help = 'activate to allow printing of loss and accuracy after each epoch', action = 'store_true', default = False)

args = parser.parse_args()
print(args)

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
num_workers = torch.cuda.device_count() if args.worker_num == -1 else args.worker_num

if args.verbose2:
    print('Device being used:', device)

# intialize the model
layer_sizes = {18 : [2, 2, 2, 2], 34 : [3, 4, 6, 3]}
num_classes = {'ucf' : 101, 'hmdb' : 51}
in_channels = {'rgb' : 3, 'flow' : 2}

# Uncomment this to test on whether the Dataloader is working
########### DATALOADER TESTING ZONE
#dataset = VideoDataset(args.dataset, args.split, 'train', args.modality, 
#                       clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt)
#                                           
#dataloader = DataLoader(dataset, shuffle=True, batch_size=2, num_workers = num_workers)
#for x, y in dataloader:
#    print(x.shape)
#x,y = next(iter(dataset))
#print(x.shape)
###########

# initialize the model
model = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                    in_channels = in_channels[args.modality], verbose = args.verbose1, 
                    bn_momentum = args.bn_momentum, bn_epson = args.bn_epson).to(device)
# initialize the model parameters according to msra_fill initialization
# DISABLED as it worsens the optimization
#msra_init(model)

# initialize loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
# decay lr by dividing alpha 0.1 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.lr_decay)
                                        
# prepare the datasets
train_dataloader = DataLoader(VideoDataset(args.dataset, args.split, 'train', args.modality, 
                                           clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                              batch_size = args.batch_size, shuffle = True, num_workers = num_workers)

val_dataloader = DataLoader(VideoDataset(args.dataset, args.split, 'test', args.modality, 
                                         clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                            batch_size = args.batch_size, num_workers = num_workers) if args.validation_mode else None

#test_dataloader = DataLoader(VideoDataset(args.dataset, args.split, 'test', args.modality, 
#                                         clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt
#                                         ,load_mode = 'video', clips_per_video = 10), 
#                            batch_size = args.batch_size, num_workers = num_workers)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

# Uncomment this to test on only few randomly selected classes
# Note: Training tends to fail when num_classess is too small
# Solved by disabling validation mode, possible cause is that no sample for the validation y label
############# MODEL TESTING ZONE
old_y = set(list(train_dataloader.dataset._labels))
num_classes = len(old_y)
y_dict = {old_label : new_label for new_label, old_label in enumerate(old_y)}
train_dataloader.dataset._labels = np.array([y_dict[i] for i in train_dataloader.dataset._labels], dtype = int)
#val_dataloader.dataset._labels = np.array([y_dict[i] for i in val_dataloader.dataset._labels], dtype = int)
#print(train_dataloader.dataset._labels, val_dataloader.dataset._labels)
model.replaceLinear(num_classes)
model.to(device)
#msra_init(model)
#############

if args.verbose2:
    if args.validation_mode:
        print('Dataset loaded:', args.dataset, args.modality, 'containing', len(train_dataloader.dataset), 'training samples', 
              'and', len(val_dataloader.dataset), 'validation samples')
    else:
        print('Dataset loaded:', args.dataset, args.modality, 'containing', len(train_dataloader.dataset), 'training samples', 
              'and', 'None' , 'validation samples')

train_model(args, device, model, dataloaders, optimizer, criterion, scheduler = scheduler)

