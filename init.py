# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:45:34 2019

@author: Juen
"""
import argparse
import itertools
import traceback

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import VideoDataset, TwoStreamDataset
from network_r2p1d import R2Plus1DNet
from fusion_network import FusionNet, FusionNet2
from train_net import train_model, save_training_model
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
parser.add_argument('-sbs', '--subbatch-size', help = 'number of labelled sample for each sub-batch', default = 8, type = int)
parser.add_argument('-lr', '--learning-rate', help = 'initial learning rate (alpha) for updating parameters', default = 0.01, type = float)
parser.add_argument('-ss', '--step-size', help = 'decaying lr for each [ss] epoches', default = 10, type = int)
parser.add_argument('-gm', '--lr-decay', help = 'lr decaying rate', default = 0.1, type = float)
parser.add_argument('-mo', '--bn-momentum', help = 'momemntum for batch normalization', default = 0.1, type = float)
parser.add_argument('-es', '--bn-epson', help = 'epson for batch normalization', default = 1e-3, type = float)
# fusion settings
parser.add_argument('-fusion', '--fusion', help = 'Fusion method to be used', default = 'none', choices = ['none', 'average', 'modality-wf'])
# debugging mode settings
parser.add_argument('-tm', '--test-mode', help = 'activate test mode to minimize dataset for debugging purpose', action = 'store_true', default = False)
parser.add_argument('-tc', '--test-amt', help = 'number of labelled samples to be left when test mode is activated', default = 2, type = int)
# pre-training settings
parser.add_argument('-train', '--train', help = 'activate to train the model', action = 'store_true', default = False)
parser.add_argument('-loadmodel', '--load-model', help = 'path to the pretrained model state_dict', default = None, type = str)
# device and parallelism settings
parser.add_argument('-dv', '--device', help = 'device chosen to perform training', default = 'gpu', choices = ['gpu', 'cpu'])
parser.add_argument('-parallel', '--parallel', help = 'activate to run on multiple gpus', action = 'store_true', default = False)
parser.add_argument('-wn', '--worker-num', help = 'number of workers for some processes (safer to set at 0; -1 set as number of device)', default = 0, type = int)
# validation settings
parser.add_argument('-va', '--validation-mode', help = 'activate validation mode', action = 'store_true', default = False)
# testing settings
parser.add_argument('-test', '--test', help = 'activate to evaluate the model', action = 'store_true', default = False)
parser.add_argument('-tbs', '--test-batch-size', help = 'number of sample in each testing batch', default = 1, type = int)
parser.add_argument('-stbs', '--sub-test-batch-size', help = 'number of clips in each testing sub-batch', default = 8, type = int)
parser.add_argument('-runalltest', '--run-all-test', help = 'activate to run all prediction methods to obtain multiple accuracy', action = 'store_true', default = False)
parser.add_argument('-lm', '--load-mode', help = 'load testing samples as series of clips (video) or a single clip', default = 'video', choices = ['video'])
parser.add_argument('-topk', '--top-acc', help = 'comapre true labels with top-k predicted labels', default = 1, type = int)
parser.add_argument('-nclip', '--clips-per-video', help = 'number of clips for testing video in video-level prediction', default = 10, type = int)
# output settings
parser.add_argument('-save', '--save', help = 'save model and accuracy', action = 'store_true', default = False)
parser.add_argument('-saveintv', '--save-interval', help = 'save model after running N epochs', default = 5, type = int)
parser.add_argument('-savename', '--savename', help = 'name of the output file', default = 'save', type = str)
parser.add_argument('-v1', '--verbose1', help = 'activate to allow reporting of activation shape after each forward propagation', action = 'store_true', default = False)
parser.add_argument('-v2', '--verbose2', help = 'activate to allow printing of loss and accuracy after each epoch', action = 'store_true', default = False)
# special case
parser.add_argument('-mwfpretrain', '--mwfpretrain', help = 'apply pretrained model on stream networks', action = 'store_true', default = False)

args = parser.parse_args()

# ensure there is no fusion method if it is only to be trained on single modality
if args.modality != '2-stream':
    assert(args.fusion == 'none')

print(args)

# Allocate device (gpu/cpu) to be used in training and testing
# CRITICAL SETTINGS OF DEFAULT GPU HOLDING ALL THE TENSORS
gpu_name = 'cuda:0'
device = torch.device(gpu_name if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
num_devices = torch.cuda.device_count() 
num_workers = num_devices if args.worker_num == -1 else args.worker_num

if args.verbose2:
    print('###### Device being used:', device, '| device_num', num_devices, '| parallelism', args.parallel)
    print('###### Fusion method:', args.fusion)

# intialize the model hyperparameters
layer_sizes = {18 : [2, 2, 2, 2], 34 : [3, 4, 6, 3]}
num_classes = {'ucf' : 101, 'hmdb' : 51}
#in_channels = {'rgb' : 3, 'flow' : 2}
in_channels = {'rgb' : 3, 'flow' : 1}
fusion_endpoint = {'average' : 'FC', 'modality-wf' : 'AP'}

save_content = {}

# prepare for training/testing on split selected
if args.modality != '2-stream':
    modalities = [args.modality]
    network = R2Plus1DNet
    datasetClass = VideoDataset
else:
    if args.fusion != 'none':
        modalities = ['2-stream']
        network = FusionNet2
        datasetClass = TwoStreamDataset
    else:
        modalities = ['rgb', 'flow']
        network = R2Plus1DNet
        datasetClass = VideoDataset

splits = [args.split] if args.split != 0 else list(range(1, 4))

# prepare pretrained model
if args.load_model is not None:
    content = torch.load(args.load_model, map_location = {'cuda:2': gpu_name})['content']

try:

    for modality, split in itertools.product(modalities, splits):
        
        if args.verbose2:
            #print(f'****** Current task: dataset {args.dataset} | modality {modality} | split {split}')
            print('****** Current task: dataset', args.dataset,'| modality', modality,'| split', split)
        
        # initialize the model
        if network == R2Plus1DNet:
            model = network(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                in_channels = in_channels[modality], verbose = args.verbose1, 
                                bn_momentum = args.bn_momentum, bn_epson = args.bn_epson)
        else:
            model = network(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                                fusion = args.fusion, endpoint = fusion_endpoint[args.fusion], 
                                verbose = args.verbose1, bn_momentum = args.bn_momentum, bn_epson = args.bn_epson, 
                                load_pretrained_stream = args.mwfpretrain, 
                                load_fusion_state = args.load_model is not None)
        
        # initialize the model parameters according to msra_fill initialization
        # DISABLED as it worsens the optimization
        #msra_init(model)
        
        # introduces parallelism into the model
        # ASSUME the remote machine has 4 GPUs
        if args.parallel and num_devices > 1 and ('cuda' in device.type):
            """
            nn.DataParallel(module, device_ids, output_device, dim)
            module : module to be parallelized
            device_ids : default set to list of all available GPUs
            the first in the device_ids list would be cuda:0 as default, could set other gpu as host
            output_device : default set to device_ids[0]
            dim : deafult = 0, axis where tensors to be scattered
            """
            model = nn.DataParallel(model, ['cuda:0', 'cuda:1', 'cuda:3'], 'cuda:2')
        
        # move model to computing devices
        model.to(device)
        
        # if pretrained model is available, load it
        if args.load_model is not None:
            print('##### Loading pre-trained model', args.load_model + '/content/' + modality + '/split' + str(split) + '/state_dict')
            model.load_state_dict(content[modality]['split'+str(split)]['state_dict'],
                    strict = False)
            
        if args.train:
        
            # initialize loss function, optimizer, and scheduler
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)
            
            # decay lr by dividing alpha 0.1 every 10 epochs
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.lr_decay)
            
            # prepare the training datasets and validation datasets (if have)
            if datasetClass == VideoDataset:
                train_dataloader = DataLoader(datasetClass(args.dataset_path, args.dataset, split, 'train', modality, 
                                                           clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                                              batch_size = args.batch_size, shuffle = True, num_workers = num_workers)
                
                val_dataloader = DataLoader(datasetClass(args.dataset_path, args.dataset, split, 'test', modality, 
                                                         clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                                            batch_size = args.batch_size, num_workers = num_workers) if args.validation_mode else None
            else:
                train_dataloader = DataLoader(TwoStreamDataset(args.dataset_path, args.dataset, split, 'train', 
                                                       clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                                          batch_size = args.batch_size, shuffle = True, num_workers = num_workers)
                
                val_dataloader = DataLoader(TwoStreamDataset(args.dataset_path, args.dataset, split, 'test', 
                                                     clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt), 
                                        batch_size = args.batch_size, num_workers = num_workers) if args.validation_mode else None
            
            dataloaders = {'train': train_dataloader, 'val': val_dataloader}

            if args.verbose2:
                if args.validation_mode:
                    print('###### Dataset loaded:', ' training samples', len(train_dataloader.dataset), '| validation samples', 
                          len(val_dataloader.dataset))
                    
                else:
                    print('###### Dataset loaded:', ' training samples', len(train_dataloader.dataset), '| validation samples None')
            
            # train model
            if args.save:
                train_loss, train_acc, train_elapsed = train_model(
                        args, device, model, dataloaders, optimizer, criterion, scheduler = scheduler, 
                        pretrained_content = content if args.load_model is not None else None, 
                        modality = modality, split = split, save_content = save_content)
                
                save_training_model(args, save_content, modality, split, 
                                    train_acc = train_acc,
                                    train_loss = train_loss,
                                    train_elapsed = train_elapsed,
                                    state_dict = model.state_dict(),
                                    stream_weight = model.stream_weights if isinstance(model, FusionNet2) else None,
                                    opt_dict = optimizer.state_dict(),
                                    sch_dict = scheduler.state_dict() if scheduler is not None else {},
                                    epoch = args.epoch
                                    )
                
            else:
                train_loss, train_acc, train_elapsed = train_model(
                        args, device, model, dataloaders, optimizer, criterion, scheduler = scheduler,
                        pretrained_content = content if args.load_model is not None else None, 
                        modality = modality, split = split, save_content = save_content)
            
        testing_content = {}
        
        if args.test:
            
            # prepare for testing with method selected
            # removed clip-level prediction if -runalltest
            pred_levels = [args.load_mode] if not args.run_all_test else ['video']
            top_acc = [args.top_acc] if not args.run_all_test else [1, 5]
            
            # load the stream weightages if it is a fusion network
            if args.load_model is not None and isinstance(model, FusionNet2) and args.fusion is not 'average':
                model.stream_weights = content[modality]['split'+str(split)]['stream_weight']
            
            # initialize testing dataset for clip/video level predictions
            if datasetClass == VideoDataset:
                test_dataloader = DataLoader(datasetClass(args.dataset_path, args.dataset, split, 'test', modality, 
                                                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt,
                                                             load_mode = 'video', clips_per_video = args.clips_per_video), 
                                                batch_size = args.test_batch_size, num_workers = num_workers, shuffle = False)
            else:
                test_dataloader = DataLoader(datasetClass(args.dataset_path, args.dataset, split, 'test', 
                                                             clip_len = args.clip_length, test_mode = args.test_mode, test_amt = args.test_amt,
                                                             load_mode = 'video', clips_per_video = args.clips_per_video), 
                                                batch_size = args.test_batch_size, num_workers = num_workers, shuffle = False)
    
            for top in top_acc:
                    
                if args.verbose2:
#                    print(f'####### Dataset loaded: testing samples {len(test_dataloader.dataset)}')
#                    print(f'####### Current Testing method: prediction-level {level} | top {top}')
                    print('####### Dataset loaded: testing samples ',len(test_dataloader.dataset))
                    print('####### Current Testing method: prediction-level ','video',' | top' ,top)
        
                # use the trained model to predict X_test
                model.eval()
                predicted, test_acc, test_elapsed = test_model(
                        args, device, model, test_dataloader, load_mode = 'video', top_acc = top)
                
                # append the testing outcomes with current prediction method to a buffer dict
                testing_content['video' + '@' + str(top)] = {
                        'predicted' : predicted,
                        'test_acc' : test_acc,
                        'test_elapsed' : test_elapsed
                    }
            
        # append the model of current training/testing task to a buffer dict
        if modality not in save_content.keys():
            save_content[modality] = {}
            
        if ('split' + str(split)) not in save_content[modality]:
            save_content[modality]['split' + str(split)] = {}
            
        save_content[modality]['split' + str(split)].update(testing_content)
        
        # saving complete contents
        if args.save:
            save_path = args.savename + '.pth.tar'
            print('Saving all content to', save_path)
            torch.save({
                    'args' : args,
                    'content' : save_content
                    }, save_path)
                    
except Exception:
    print(traceback.format_exc())
    
else:
    print('Everything went well \\\\^o^//')
