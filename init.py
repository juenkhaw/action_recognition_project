# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:45:34 2019

@author: Juen
"""
import argparse
import numpy as np
import itertools

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import VideoDataset
from network_r2p1d import R2Plus1DNet
from module import msra_init
from train_net import train_model
from test_net import test_model

parser = argparse.ArgumentParser(description = 'PyTorch 2.5D Action Recognition ResNet')
# training settings
parser.add_argument('dataset', help = 'video dataset to be trained and validated', choices = ['ucf', 'hmdb'])
parser.add_argument('modality', help = 'modality to be trained and validated', choices = ['rgb', 'flow', '2-stream'])
parser.add_argument('dataset_path', help = 'path link to the directory where rgb_frames and optical flows located')
parser.add_argument('-cl', '--clip-length', help = 'initial temporal length of each video training input', default = 8, type = int)
parser.add_argument('-sp', '--split', help = 'dataset split selected in training and evaluating model (0 to train/test on all split)', default = 0, choices = list(range(4)), type = int)
parser.add_argument('-ld', '--layer-depth', help = 'depth of the resnet', default = 18, choices = [18, 34], type = int)
parser.add_argument('-ep', '--epoch', help = 'number of epochs for training process', default = 45, type = int)
parser.add_argument('-bs', '--batch-size', help = 'number of labelled sample for each batch', default = 32, type = int)
parser.add_argument('-lr', '--learning-rate', help = 'initial learning rate (alpha) for updating parameters', default = 0.01, type = float)
parser.add_argument('-ss', '--step-size', help = 'decaying lr for each [ss] epoches', default = 10, type = int)
parser.add_argument('-gm', '--lr-decay', help = 'lr decaying rate', default = 0.1, type = float)
parser.add_argument('-tm', '--test-mode', help = 'activate test mode to minimize dataset for debugging purpose', action = 'store_true', default = False)
parser.add_argument('-tc', '--test-amt', help = 'number of labelled samples to be left when test mode is activated', default = 2, type = int)
parser.add_argument('-mo', '--bn-momentum', help = 'momemntum for batch normalization', default = 0.1, type = float)
parser.add_argument('-es', '--bn-epson', help = 'epson for batch normalization', default = 1e-3, type = float)
# device and parallelism settings
parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-parallel', '--parallel', help = 'activate to run on multiple gpus', action = 'store_true', default = False)
parser.add_argument('-wn', '--worker-num', help = 'number of workers for some processes (safer to set at 0; -1 set as number of device)', default = 0, type = int)
# validation settings
parser.add_argument('-va', '--validation-mode', help = 'activate validation mode', action = 'store_true', default = False)
# testing settings
parser.add_argument('-runalltest', '--run-all-test', help = 'activate to run all prediction methods to obtain multiple accuracy', action = 'store_true', default = False)
parser.add_argument('-lm', '--load-mode', help = 'load testing samples as series of clips (video) or a single clip', default = 'clip', choices = ['video', 'clip'])
parser.add_argument('-topk', '--top-acc', help = 'comapre true labels with top-k predicted labels', default = 1, type = int)
parser.add_argument('-nclip', '--clips-per-video', help = 'number of clips for testing video in video-level prediction', default = 10, type = int)
# output settings
parser.add_argument('-save', '--save', help = 'save model and accuracy', action = 'store_true', default = False)
parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)
parser.add_argument('-v2', '--verbose2', help = 'activate to allow printing of loss and accuracy after each epoch', action = 'store_true', default = False)

args = parser.parse_args()
if args.load_mode == 'clip' and not args.run_all_test:
    args.clips_per_video = 1
print(args)

# Allocate device (gpu/cpu) to be used in training and testing
device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
num_devices = torch.cuda.device_count() 
num_workers = num_devices if args.worker_num == -1 else args.worker_num

if args.verbose2:
    print(f'Device being used: {device} | device_num {num_devices} | parallelism {args.parallel}')

# intialize the model hyperparameters
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

save_content = {}

# prepare for training/testing on split selected
modalities = [args.modality] if args.modality != '2-stream' else ['rgb', 'flow']
splits = [args.split] if args.split != 0 else list(range(1, 4))

for modality, split in itertools.product(modalities, splits):
    
    if args.verbose2:
        print(f'****** Current task: dataset {args.dataset} | modality {modality} | split {split}')
    
    # initialize the model
    model = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                        in_channels = in_channels[modality], verbose = args.verbose1, 
                        bn_momentum = args.bn_momentum, bn_epson = args.bn_epson)
    # initialize the model parameters according to msra_fill initialization
    # DISABLED as it worsens the optimization
    #msra_init(model)
    
    # introduces parallelism into the model
    if args.parallel and num_devices > 1 and ('cuda' in device.type):
        """
        nn.DataParallel(module, device_ids, output_device, dim)
        module : module to be parallelized
        device_ids : default set to list of all available GPUs
        output_device : default set to device_ids[0]
        dim : deafult = 0, axis where tensors to be scattered
        """
        model = nn.DataParallel(model)
        
    model.to(device)
    
    # initialize loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
    
    # decay lr by dividing alpha 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.lr_decay)
    
    # prepare the training datasets and validation datasets (if have)
    train_dataloader = DataLoader(VideoDataset(args.dataset_path, args.dataset, split, 'train', modality, 
                                               clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                                  batch_size = args.batch_size, shuffle = True, num_workers = num_workers)
    
    val_dataloader = DataLoader(VideoDataset(args.dataset_path, args.dataset, split, 'test', modality, 
                                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                                batch_size = args.batch_size, num_workers = num_workers) if args.validation_mode else None
    
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    # Uncomment this to test on only few randomly selected classes
    # Note: Training tends to fail when num_classess is too small
    # Solved by disabling validation mode, possible cause is that no sample for the validation y label
    ############# MODEL TESTING ZONE
    #old_y = set(list(train_dataloader.dataset._labels))
    #num_classes = len(old_y)
    #y_dict = {old_label : new_label for new_label, old_label in enumerate(old_y)}
    #train_dataloader.dataset._labels = np.array([y_dict[i] for i in train_dataloader.dataset._labels], dtype = int)
    ##val_dataloader.dataset._labels = np.array([y_dict[i] for i in val_dataloader.dataset._labels], dtype = int)
    ##print(train_dataloader.dataset._labels, val_dataloader.dataset._labels)
    #model.replaceLinear(num_classes)
    #model.to(device)
    ##msra_init(model)
    #############
    
    if args.verbose2:
        if args.validation_mode:
            print('###### Dataset loaded:', ' training samples', len(train_dataloader.dataset), '| validation samples', 
                  len(val_dataloader.dataset))
        else:
            print('###### Dataset loaded:', ' training samples', len(train_dataloader.dataset), '| validation samples None')
    
    # train model
    train_loss, train_acc, train_elapsed = train_model(
            args, device, model, dataloaders, optimizer, criterion, scheduler = scheduler)
    
    # prepare for testing with method selected
    pred_levels = [args.load_mode] if not args.run_all_test else ['clip', 'video']
    top_acc = [args.top_acc] if not args.run_all_test else [1, 5]
    
    testing_content = {}
    
    for level, top in itertools.product(pred_levels, top_acc):
        
        if level == 'clip' and top == 5:
            continue
    
        # initialize testing dataset for clip/video level predictions
        test_dataloader = DataLoader(VideoDataset(args.dataset_path, args.dataset, split, 'test', modality, 
                                                 clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt,
                                                 load_mode = level, clips_per_video = args.clips_per_video if level == 'video' else 1), 
                                    batch_size = args.batch_size, num_workers = num_workers, shuffle = False)
                
        if args.verbose2:
            print(f'####### Dataset loaded: testing samples {len(test_dataloader.dataset)}')
            print(f'####### Current Testing method: prediction-level {level} | top {top}')
        
        # use the trained model to predict X_test
        predicted, test_acc, test_elapsed = test_model(
                args, device, model, test_dataloader, load_mode = level, top_acc = top)
        
        # append the testing outcomes with current prediction method to a buffer dict
        testing_content[level + '@' + str(top)] = {
                'predicted' : predicted,
                'test_acc' : test_acc,
                'test_elapsed' : test_elapsed
            }
    
    # append the model of current training/testing task to a buffer dict
    if modality not in save_content.keys():
        save_content[modality] = {}
    save_content[modality].update({
            'split' + str(split) : {
                'train_acc' : train_acc,
                'train_loss' : train_loss,
                'train_elapsed' : train_elapsed,
                'state_dict' : model.state_dict(),
                #'opt_dict': optimizer.state_dict(),
                'testing_content' : testing_content
            }
        })

# save the model and details
if args.save:
    save_path = 'fyp_r2.5_1704754_juen.pth.tar'
    torch.save({
            'args' : args,
            'content' : save_content
            }, save_path)