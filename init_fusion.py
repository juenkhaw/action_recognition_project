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

from dataset import TwoStreamDataset
from network_r2p1d import R2Plus1DNet
from fusion_network import FusionNet
from train_net import train_pretrained_stream, save_training_model

parser = argparse.ArgumentParser(description = 'R(2+1)D Fusion Network')

# mandatory settings
parser.add_argument('dataset', help = 'video dataset to be trained and validated', choices = ['ucf', 'hmdb'])
parser.add_argument('dataset_path', help = 'path link to the directory where rgb_frames and optical flows located')
parser.add_argument('fusion', help = 'Fusion method to be used', choices = ['average', 'modality-1-layer'])
parser.add_argument('archi', help = 'architecture used on complying streams and fusion networks', choices = ['pref', 'e2e'])

# network and optimizer settings
parser.add_argument('-train', '--train', help = 'activate to train the model', action = 'store_true', default = False)
parser.add_argument('-cl', '--clip-length', help = 'initial temporal length of each video training input', default = 8, type = int)
parser.add_argument('-sp', '--split', help = 'dataset split selected in training and evaluating model', default = 1, choices = list(range(1, 4)), type = int)
parser.add_argument('-ld', '--layer-depth', help = 'depth of the resnet', default = 18, choices = [18, 34], type = int)
parser.add_argument('-ep', '--epoch', help = 'number of epochs for training process', default = 45, type = int)
parser.add_argument('-bs', '--batch-size', help = 'number of labelled sample for each batch', default = 32, type = int)
parser.add_argument('-sbs', '--subbatch-size', help = 'number of labelled sample for each sub-batch', default = 8, type = int)

# model state loading settings
parser.add_argument('-loadstream', '--load-stream', help = 'paths to the pretrained stream model state_dict (rgb, flow)', nargs = '+', default = [])
parser.add_argument('-loadfusion', '--load-fusion', help = 'path to the pretrained fusion network model state_dict (pref : fusion network only, e2e : streams + fusion network)', 
                    nargs = '+', default = [])

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
print('******* ARGUMENTS *******', args ,'\n*************************\n')

assert(args.train or args.test)

# pretrained fusion is not legit for avearging fusion
# and ensure that streams are preloaded with model state
if args.fusion == 'average':
    assert(args.archi == 'pref')
    assert(not args.train)
    assert(len(args.load_stream) == 2)

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
stream_endp = {'average' : ['SCORES'], 'modality-1-layer' : ['AP', 'FC']}

# intialize save content
save_content = {}
save_content['args'] = args

# define and intialize the network
rgbnet = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                in_channels = in_channels['rgb'], verbose = args.verbose1, 
                                bn_momentum = 0.1, bn_epson = 1e-3, endpoint = stream_endp[args.fusion]).to(device)
flownet = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                in_channels = in_channels['flow'], verbose = args.verbose1, 
                                bn_momentum = 0.1, bn_epson = 1e-3, endpoint = stream_endp[args.fusion]).to(device)
fusionnet = FusionNet(fusion = args.fusion).to(device)

# ensure there are stream models input if training mode is set to pretrained streams
if args.archi == 'pref':
    assert(len(args.load_stream) == 2)

# load the stream network model state that is completed training
# FOR TESTING ONLY
if (args.load_stream is not [] or args.load_fusion is not []) and (not args.train and args.test):
        
    print('\n********* LOADING STATE ***********', 
          '\nFusion Mode =', 'End to End' if args.archi == 'e2e' else 'Pretrained Streams'
          '\nStream Paths =', args.load_stream, '\nFusion Paths =', args.load_fusion)
    
    # validation on number of model passed in
    if args.archi == 'pref':
        
        assert(len(args.load_stream) == 2)
        if args.fusion != 'average':
            assert(len(args.load_fusion) == 1)
        
        rgb_state = torch.load(args.load_stream[0])['train']['state_dict']
        flow_state = torch.load(args.load_stream[1])['train']['state_dict']
        
        rgbnet.load_state_dict(rgb_state)
        flownet.load_state_dict(flow_state)
        
        del rgb_state
        del flow_state
        
        if args.fusion != 'average':
            fusion_state = torch.load(args.load_fusion[0])['train']['state_dict']
            fusionnet.load_state_dict(fusion_state)
            
            del fusion_state
        
    elif args.archi == 'e2e':
        assert(len(args.load_fusion) == 1)
        
        state = torch.load(args.load_fusion[0])['train']
        
        rgbnet.load_state_dict(state['rgb'])
        flownet.load_state_dict(state['flow'])
        fusionnet.load_state_dict(state['fusion'])
        
        del state
    
    print('************* LOADED **************')
    
try:    
    # applying parallelism into the model
    # assuming 4 GPUs scnario
    if args.parallel  and ('cuda' in device.type):
        """
        nn.DataParallel(module, device_ids, output_device, dim)
        module : module to be parallelized
        device_ids : default set to list of all available GPUs
        the first in the device_ids list would be cuda:0 as default, could set other gpu as host
        output_device : default set to device_ids[0]
        dim : deafult = 0, axis where tensors to be scattered
        """
        
        # SETTINGS on working GPUs ===============================
        rgbnet = nn.DataParallel(rgbnet, [int(i[len(i) - 1]) for i in all_gpu], gpu_name[len(gpu_name) - 1])
        flownet = nn.DataParallel(flownet, [int(i[len(i) - 1]) for i in all_gpu], gpu_name[len(gpu_name) - 1])
        fusionnet = nn.DataParallel(fusionnet, [int(i[len(i) - 1]) for i in all_gpu], gpu_name[len(gpu_name) - 1])
    
    # execute training
    if args.train:
        
        if args.verbose2:
            print('\n************ TRAINING *************', 
                  '\nDataset =', args.dataset, '\nFusion =', args.fusion,
                  '\nSplit =', args.split)
        
        # define criterion, optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        
        if args.archi == 'e2e':
            rgb_optimizer = optim.SGD(rgbnet.parameters(), lr = 1e-2)
            flow_optimizer = optim.SGD(flownet.parameters(), lr = 1e-2)
            
            rgb_scheduler = optim.lr_scheduler.ReduceLROnPlateau(rgb_optimizer, patience = 10, threshold = 1e-4, min_lr = 1e-6)
            flow_scheduler = optim.lr_scheduler.ReduceLROnPlateau(flow_optimizer, patience = 10, threshold = 1e-4, min_lr = 1e-6)
        
        if args.fusion is not 'average':
            fusion_optimizer = optim.SGD(fusionnet.parameters(), lr = 1e-2)
            fusion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fusion_optimizer, patience = 10, threshold = 1e-4, min_lr = 1e-6)
            
        # preparing the training and validation dataset
        train_dataloader = DataLoader(
                TwoStreamDataset(args.dataset_path, args.dataset, args.split, 'train', 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                batch_size = args.batch_size, shuffle = True)
        val_dataloader = DataLoader(
                TwoStreamDataset(args.dataset_path, args.dataset, args.split, 'validation', 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                batch_size = args.batch_size, shuffle = False)
                
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        
        if args.verbose2:
            print('Training Set =', len(train_dataloader.dataset), 
                  '\nValidation Set =', len(val_dataloader.dataset),
                  '\n***********************************')
            
        # train
        if args.archi == 'pref':
            losses, accs, train_elapsed = train_pretrained_stream(args, device, 
                                                                  {'rgb':rgbnet,'flow':flownet,'fusion':fusionnet}, 
                                                                  dataloaders, fusion_optimizer, criterion, 
                                                                  fusion_scheduler, None)
#        elif args.train_option == 'pref':
#            losses, accs, train_elapsed = train_fusion(args, device, 
#                                                   {'rgb':rgbnet,'flow':flownet,'fusion':fusionnet}, 
#                                                   dataloaders, 
#                                                   {'fusion':fusion_optimizer}, 
#                                                   criterion, 
#                                                   {'fusion':fusion_scheduler}, 
#                                                   save_content)
        
#        if args.save:
#            save_training_model(args, 'train', save_content,  
#                                    accuracy = accs,
#                                    losses = losses,
#                                    train_elapsed = train_elapsed,
#                                    state_dict = model.state_dict(),
#                                    opt_dict = optimizer.state_dict(),
#                                    sch_dict = scheduler.state_dict() if scheduler is not None else {},
#                                    epoch = args.epoch
#                                    )
    # execute testing
    if args.test:
        
        if args.verbose2:
            print('\n************ TESTING **************', 
                  '\nDataset =', args.dataset, '\nFusion =', args.fusion,
                  '\nSplit =', args.split)
        
        # preparing testing dataset
        test_dataloader = DataLoader(
                TwoStreamDataset(args.dataset_path, args.dataset, args.split, 'test',
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt),
                batch_size = args.test_batch_size, shuffle = False)
                
        if args.verbose2:
            print('Testing Set =', len(test_dataloader.dataset),
                  '\n***********************************')
#            
#        # testing
#        all_scores, test_acc, test_elapsed = test_stream(args, device, model, test_dataloader)
#        
#        if args.save:
#            save_training_model(args, 'test', save_content, 
#                                    scores = all_scores,
#                                    accuracy = test_acc,
#                                    test_elapsed = test_elapsed
#                                    )
except Exception:
    print(traceback.format_exc())
    
else:
    print('Everything went well \\\\^o^//')