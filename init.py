# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 21:45:34 2019

@author: Juen
"""
import torch
import time
import argparse
import numpy as np

from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import VideoDataset
from network_r2p1d import R2Plus1DNet

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

model = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes[args.dataset], device, 
                    in_channels = in_channels[args.modality], verbose = args.verbose1).to(device)

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
                            batch_size = args.batch_size, num_workers = num_workers)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

############# MODEL TESTING ZONE
#old_y = set(np.append(train_dataloader.dataset._labels, val_dataloader.dataset._labels))
#num_classes = len(old_y)
#y_dict = {old_label : new_label for new_label, old_label in enumerate(old_y)}
#train_dataloader.dataset._labels = np.array([y_dict[i] for i in train_dataloader.dataset._labels], dtype = int)
#val_dataloader.dataset._labels = np.array([y_dict[i] for i in val_dataloader.dataset._labels], dtype = int)
#print(train_dataloader.dataset._labels, val_dataloader.dataset._labels)
#model = R2Plus1DNet(layer_sizes[args.layer_depth], num_classes, device, 
#                    in_channels = in_channels[args.modality], verbose = False).to(device)
#############

if args.verbose2:
    print('Dataset loaded:', args.dataset, 'containing', len(train_dataloader.dataset), 'training samples', 
      'and', len(val_dataloader.dataset), 'validation samples')


# to record time elapsed
start = time.time()
epoch = 0

if args.verbose2:
    print('Starting to train model.......')

for epoch in range(epoch, args.epoch):
        
    for phase in ['train', 'val']:
        
        # reset the loss and accuracy
        current_loss = 0
        current_correct = 0
        
        # initialize the model with respective mode
        if phase == 'train':
            scheduler.step()
            model.train()
        else:
            model.eval()
            
        #print(len(dataloaders[phase].dataset))
                
        # for each mini batch of dataset
        for inputs, labels in dataloaders[phase]:
            
            #print(inputs.shape, labels.shape)
                        
            # retrieve the inputs and labels and send to respective computing devices
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            
            # with enabling gradient descent on parameters during training phase
            with torch.set_grad_enabled(phase == 'train'):
                # compute the final scores
                outputs = model(inputs)
                # transforming outcome from a series of scores to a single scalar index
                # indicating the index where the highest score held
                _, preds = torch.max(outputs, 1)
                # compute the loss
                loss = criterion(outputs, labels)
                
                # update parameters if it is training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # accumulate loss and true positive for the current batch
                current_loss += loss.item() * inputs.size(0)
                current_correct += torch.sum(preds == labels.data)
            
            # compute the loss and accuracy for the current batch
            epoch_loss = current_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(current_correct) / len(dataloaders[phase].dataset)
            
            if args.verbose2:
                print(f'Epoch {epoch} {phase} Loss = {epoch_loss:.4f} Accuracy = {epoch_acc:.2f}')

# display the time elapsed
time_elapsed = time.time() - start    
print(f"Training complete in {int(time_elapsed//3600)}h {int((time_elapsed%3600)//60)}m {int(time_elapsed %60)}s")