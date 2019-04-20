# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 23:35:43 2019

@author: Juen
"""

import argparse
import traceback

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import VideoDataset
from network_r2p1d import R2Plus1DNet
from test_net import test_stream
from train_net import train_stream

parser = argparse.ArgumentParser(description = 'R(2+1)D Stream Network')

# training settings
parser.add_argument('dataset', help = 'video dataset to be trained and validated', choices = ['ucf', 'hmdb'])
parser.add_argument('modality', help = 'modality to be trained and validated', choices = ['rgb', 'flow', '2-stream'])
parser.add_argument('dataset_path', help = 'path link to the directory where rgb_frames and optical flows located')
parser.add_argument('-cl', '--clip-length', help = 'initial temporal length of each video training input', default = 8, type = int)
parser.add_argument('-sp', '--split', help = 'dataset split selected in training and evaluating model', default = 1, choices = list(range(1, 4)), type = int)
parser.add_argument('-ld', '--layer-depth', help = 'depth of the resnet', default = 18, choices = [18, 34], type = int)
parser.add_argument('-ep', '--epoch', help = 'number of epochs for training process', default = 45, type = int)
parser.add_argument('-bs', '--batch-size', help = 'number of labelled sample for each batch', default = 32, type = int)
parser.add_argument('-sbs', '--subbatch-size', help = 'number of labelled sample for each sub-batch', default = 8, type = int)

parser.add_argument('-train', '--train', help = 'activate to train the model', action = 'store_true', default = False)
parser.add_argument('-loadmodel', '--load-model', help = 'path to the pretrained model state_dict', default = None, type = str)
parser.add_argument('-modelname', '--model-name', help = 'path to the model state_dict in the model package', default = None, type = str)

# debugging mode settings
parser.add_argument('-tm', '--test-mode', help = 'activate test mode to minimize dataset for debugging purpose', action = 'store_true', default = False)
parser.add_argument('-tc', '--test-amt', help = 'number of labelled samples to be left when test mode is activated', default = 2, type = int)

# device and parallelism settings
parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-parallel', '--parallel', help = 'activate to run on multiple gpus', action = 'store_true', default = False)

# testing settings
parser.add_argument('-test', '--test', help = 'activate to evaluate the model', action = 'store_true', default = False)
parser.add_argument('-tbs', '--test-batch-size', help = 'number of sample in each testing batch', default = 32, type = int)
parser.add_argument('-stbs', '--sub-test-batch-size', help = 'number of clips in each testing sub-batch', default = 8, type = int)

# output settings
parser.add_argument('-save', '--save', help = 'save model and accuracy', action = 'store_true', default = False)
parser.add_argument('-saveintv', '--save-interval', help = 'save model after running N epochs', default = 5, type = int)
parser.add_argument('-savename', '--savename', help = 'name of the output file', default = 'save', type = str)
parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)
parser.add_argument('-v2', '--verbose2', help = 'activate to allow printing of loss and accuracy after each epoch', action = 'store_true', default = False)

# parse arguments and print it out
args = parser.parse_args()
print('******* ARGUMENTS *******\n', args ,'\n*************************\n')

# SETTINGS OF DEVICES ========================================
gpu_name = 'cuda:0'
all_gpu = ['cuda:0']
device = torch.device(gpu_name if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
num_devices = torch.cuda.device_count() 

# printing device info
if args.verbose2:
    print('######## DEVICE INFO ##########', 
          '\nGatherer Unit =', device, '\nWorking Unit =', all_gpu,
          '\nDevice Count =', num_devices, '\nParallelism =', args.parallel,
          '\n###############################')
          
# intitalize model properties
layer_sizes = {18 : [2, 2, 2, 2], 34 : [3, 4, 6, 3]}
num_classes = {'ucf' : 101, 'hmdb' : 51}
in_channels = {'rgb' : 3, 'flow' : 1}

# intialize save content
save_content = {}
save_content['args'] = args

# define and intialize the network
model = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                in_channels = in_channels[args.modality], verbose = args.verbose1, 
                                bn_momentum = 0.1, bn_epson = 1e-3, endpoint = ['FC']).to(device)

# read the model state if model path is defined
if args.load_model is not None:
        
    print('\n********* LOADING STATE ***********', 
          '\nModel Path =', args.load_model, '\nModel State =', args.model_name)
    
    assert(args.model_name is not None)
    model_state = torch.load(args.load_model)[args.modality][args.model_name]['state_dict']
    
    # load the state model into the network
    model.load_state_dict(model_state)
    del model_state
    
    print('************* LOADED **************')
    
try:
    if args.verbose2:
        print('\n************ TRAINING *************', 
              '\nDataset =', args.dataset, '\nModality =', args.modality,
              '\nSplit =', args.split)
    
    # applying parallelism into the model
    # assuming 4 GPUs scnario
    if args.parallel and num_devices > 1 and ('cuda' in device.type):
        """
        nn.DataParallel(module, device_ids, output_device, dim)
        module : module to be parallelized
        device_ids : default set to list of all available GPUs
        the first in the device_ids list would be cuda:0 as default, could set other gpu as host
        output_device : default set to device_ids[0]
        dim : deafult = 0, axis where tensors to be scattered
        """
        
        # SETTINGS on working GPUs ===============================
        model = nn.DataParallel(model, all_gpu, gpu_name)
    
    # execute training
    if args.train:
        
        # define criterion, optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = 1e-2)
        # trying on dynamic scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, threshold = 1e-4, min_lr = 1e-6)
        
        # preparing the training and validation dataset
        train_dataloader = DataLoader(
                VideoDataset(args.dataset_path, args.dataset, args.split, 'train', args.modality, 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                batch_size = args.batch_size, shuffle = True)
        val_dataloader = DataLoader(
                VideoDataset(args.dataset_path, args.dataset, args.split, 'validation', args.modality, 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                batch_size = args.batch_size, shuffle = False)
                
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        
        if args.verbose2:
            print('Training Set =', len(train_dataloader.dataset), 
                  '\nValidation Set =', len(val_dataloader.dataset),
                  '\n***********************************')
            
        # train
        train_stream(args, device, model, dataloaders, optimizer, criterion, scheduler)
    
    # execute testing
    if args.test:
        
        if args.verbose2:
            print('\n************ TESTING **************', 
                  '\nDataset =', args.dataset, '\nModality =', args.modality,
                  '\nSplit =', args.split)
        
        # preparing testing dataset
        test_dataloader = DataLoader(
                VideoDataset(args.dataset_path, args.dataset, args.split, 'test', args.modality, 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt),
                batch_size = args.test_batch_size, shuffle = False)
                
        if args.verbose2:
            print('Testing Set =', len(test_dataloader.dataset),
                  '\n***********************************')
            
        # testing
        test_stream(args, device, model, test_dataloader)
    
except Exception:
    print(traceback.format_exc())
    
else:
    print('Everything went well \\\\^o^//')