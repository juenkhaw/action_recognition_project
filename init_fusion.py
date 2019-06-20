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
from train_net import train_pref_fusion, save_training_model, mem_state
from test_net import test_pref_fusion

parser = argparse.ArgumentParser(description = 'R(2+1)D Fusion Network')

# mandatory settings
parser.add_argument('dataset', help = 'video dataset to be trained and validated', choices = ['ucf', 'hmdb'])
parser.add_argument('dataset_path', help = 'path link to the directory where rgb_frames and optical flows located')
parser.add_argument('fusion', help = 'Fusion method to be used', 
                    choices = ['average', 'vanilla-ld3', 'class-ld3', 'vanilla-ld3-2', 'class-ld3-2', 'activation-ld3'])

parser.add_argument('-lr', '--lr', help = 'learning rate', default = 1e-2, type = float)
parser.add_argument('-momentum', '--momentum', help = 'momentum magnitude', default = 0.1, type = float)
parser.add_argument('-l2wd', '--l2wd', help = 'L2 weight decaying regularizer', default = 1e-2, type = float)
parser.add_argument('-wdloss', '--wdloss', help = 'Ratio applying weight diff loss', default = 0, type = float)

# network and optimizer settings
parser.add_argument('-train', '--train', help = 'activate to train the model', action = 'store_true', default = False)
parser.add_argument('-cl', '--clip-length', help = 'initial temporal length of each video training input', default = 16, type = int)
parser.add_argument('-sp', '--split', help = 'dataset split selected in training and evaluating model', default = 1, choices = list(range(1, 4)), type = int)
parser.add_argument('-ld', '--layer-depth', help = 'depth of the resnet', default = 34, choices = [18, 34], type = int)
parser.add_argument('-ep', '--epoch', help = 'number of epochs for training process', default = 50, type = int)
parser.add_argument('-bs', '--batch-size', help = 'number of labelled sample for each batch', default = 32, type = int)
parser.add_argument('-sbs', '--subbatch-size', help = 'number of labelled sample for each sub-batch', default = 8, type = int)
parser.add_argument('-vsbs', '--val-subbatch-size', help = 'number of labelled sample for each validation sub-batch', default = 8, type = int)
parser.add_argument('-meansub', '--meansub', help = 'activates mean substraction on flows', action = 'store_true', default = False)

# model state loading settings
parser.add_argument('-loadstream', '--load-stream', help = 'paths to the pretrained stream model state_dict (rgb, flow)', nargs = '+', default = [], type = str)
parser.add_argument('-loadfusion', '--load-fusion', help = 'path to the pretrained fusion network model state_dict', nargs = '+', default = [], type = str)
parser.add_argument('-resume', '--resume', help = 'indicating this session is continuing training from last session', action = 'store_true', default = False)

# debugging mode settings
parser.add_argument('-tm', '--test-mode', help = 'activate test mode to minimize dataset for debugging purpose', default = 'none', choices = ['none', 'peek', 'distributed'])
parser.add_argument('-tc', '--test-amt', help = 'number of labelled samples to be left when test mode is activated', nargs = '+', default = [2, 1, 1])

# device and parallelism settings
parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-parallel', '--parallel', help = 'activate to run on multiple gpus', action = 'store_true', default = False)

# testing settings
parser.add_argument('-test', '--test', help = 'activate to evaluate the model', action = 'store_true', default = False)
parser.add_argument('-tbs', '--test-batch-size', help = 'number of sample in each testing batch', default = 32, type = int)
parser.add_argument('-stbs', '--sub-test-batch-size', help = 'number of clips in each testing sub-batch', default = 8, type = int)

# output settings
parser.add_argument('-save', '--save', help = 'save model and accuracy', action = 'store_true', default = False)
parser.add_argument('-savename', '--savename', help = 'name of the output file', default = 'save', type = str)
parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)
parser.add_argument('-v2', '--verbose2', help = 'activate to allow printing of loss and accuracy after each epoch', action = 'store_true', default = False)

# parse arguments and print it out
args = parser.parse_args()
print('******* ARGUMENTS *******', args ,'\n*************************\n')

assert(args.train or args.test or args.resume)
assert(len(args.load_stream) == 2)

# pretrained fusion is not legit for avearging fusion
# and ensure that streams are preloaded with model state
if args.fusion == 'average':
    assert(not args.train)
    
# arguments on debugging mode arguments
if args.test_mode == 'peek':
    assert(len(args.test_amt) == 1)
elif args.test_mode == 'distributed':
    assert(len(args.test_amt) == 3)

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
in_channels = {'rgb' : 3, 'flow' : 2}
stream_endp = {'average' : ['SCORES'], 
               'vanilla-ld3' : ['AP', 'FC'], 
               'vanilla-ld3-2' : ['AP', 'FC'], 
               'class-ld3' : ['AP', 'FC'], 
               'class-ld3-2' : ['AP', 'FC'], 
               'activation-ld3' : ['AP']
               }

# intialize save content
save_content = {}
save_content['args'] = args
    
print('\n********* LOADING MODEL ***********', 
          '\nStream State Path =', args.load_stream, 
          '\nFusion State Path =', args.load_fusion,
          '\nModel Depth =', args.layer_depth, 
          '\nClip Length =', args.clip_length,
          '\nTasks =', 'TRAIN' if args.train else '', 'TEST' if args.test else '', 'TRAIN(RESUME)' if args.resume else '')

# define and intialize the network
rgbnet = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                in_channels = in_channels['rgb'], verbose = args.verbose1, 
                                bn_momentum = 0.1, bn_epson = 1e-3, endpoint = stream_endp[args.fusion]).to(device)
flownet = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                in_channels = in_channels['flow'], verbose = args.verbose1, 
                                bn_momentum = 0.1, bn_epson = 1e-3, endpoint = stream_endp[args.fusion]).to(device)
fusionnet = FusionNet(fusion = args.fusion).to(device)

# load the stream network model state that is completed training
# FOR TESTING ONLY
if args.load_stream is not [] or args.load_fusion is not []:
        
    print('Fusion Mode =', 'Frozen Streams')
    
    # validation on number of model passed in
        
    assert(len(args.load_stream) == 2)
#        if args.fusion != 'average':
#            assert(len(args.load_fusion) == 1)
    
    rgb_state = torch.load(args.load_stream[0])['train']['best']['state_dict']
    flow_state = torch.load(args.load_stream[1])['train']['best']['state_dict']
#        rgb_state = torch.load(args.load_stream[0])['train']['best']['model_state']
#        flow_state = torch.load(args.load_stream[1])['train']['best']['model_state']
    
    rgbnet.load_state_dict(rgb_state)
    flownet.load_state_dict(flow_state)
    
    del rgb_state
    del flow_state
    
    if args.fusion != 'average' and len(args.load_fusion) > 0:
        
        if args.resume or args.train:
            fusion_state = torch.load(args.load_fusion[0])['train']
            save_content['train'] = fusion_state
        else:
            fusion_state = torch.load(args.load_fusion[0])['train']['best']
            
        fusionnet.load_state_dict(fusion_state['state_dict'])
        
        del fusion_state
        
    torch.cuda.empty_cache()
        
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
    if args.train or args.resume:
        
        if args.verbose2:
            print('\n************ TRAINING *************', 
                  '\nDataset =', args.dataset, '\nFusion =', args.fusion,
                  '\nSplit =', args.split)
        
        # define criterion, optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        
        if args.fusion is not 'average':
            #fusion_optimizer = optim.RMSprop(fusionnet.parameters(), lr = 1e-3, alpha = 0.9)
            fusion_optimizer = optim.SGD(fusionnet.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.l2wd)
            fusion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fusion_optimizer, patience = 10, threshold = 1e-4, min_lr = 1e-7)
            
        # preparing the training and validation dataset
        train_dataloader = DataLoader(
                TwoStreamDataset(args.dataset_path, args.dataset, args.split, 'train', mean_sub = args.meansub, 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                batch_size = args.batch_size, shuffle = True)
        val_dataloader = DataLoader(
                TwoStreamDataset(args.dataset_path, args.dataset, args.split, 'validation', mean_sub = args.meansub, 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                batch_size = args.val_subbatch_size, shuffle = False)
                
        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        
        torch.cuda.empty_cache()
        if args.verbose2:
            print('Training Set =', len(train_dataloader.dataset), 
                  '\nValidation Set =', len(val_dataloader.dataset),
                  '\n***********************************')
            
        # train
        train_pref_fusion(args, device, {'rgb':rgbnet,'flow':flownet,'fusion':fusionnet}, 
                          dataloaders, fusion_optimizer, criterion, fusion_scheduler, save_content)
        
    mem_state(0)
        
    # execute testing
    if args.test:
        
        torch.cuda.empty_cache()
        if args.verbose2:
            print('\n************ TESTING **************', 
                  '\nDataset =', args.dataset, '\nFusion =', args.fusion,
                  '\nSplit =', args.split)
        
        # preparing testing dataset
        test_dataloader = DataLoader(
                TwoStreamDataset(args.dataset_path, args.dataset, args.split, 'test', mean_sub = args.meansub, 
                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt),
                batch_size = args.test_batch_size, shuffle = False)
                
        if args.verbose2:
            print('Testing Set =', len(test_dataloader.dataset),
                  '\n***********************************')
            
        # testing
        all_scores, all_weights, test_acc, test_elapsed = test_pref_fusion(args, device, 
                                                                           {'rgb':rgbnet,'flow':flownet,'fusion':fusionnet}, 
                                                                           test_dataloader)
        
        if args.save:
            save_training_model(args, 'test', save_content, 
                                    scores = all_scores,
                                    weights = all_weights, 
                                    accuracy = test_acc,
                                    test_elapsed = test_elapsed
                                    )
    mem_state(0)
    
except Exception:
    print(traceback.format_exc())
    
else:
    print('Everything went well \\\\^o^//')