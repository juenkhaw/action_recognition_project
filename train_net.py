# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:50:07 2019

@author: Juen
"""
from __future__ import print_function
import time

import torch

from network_r2p1d import R2Plus1DNet
from fusion_network import FusionNet, FusionNet2

def generate_subbatches(sbs, *tensors):
    """
    Generate list of subbtaches from a batch of sample data
    
    Inputs:
        tensors : series of tensor batch to be partitioned
        
    Returns:
        subbatches : list of partitioned subbateches
    
    """
    
    #print(tensors.__class__)
    # engulf tensor into a list if there is only one tensor passed in
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
        
    subbatches = []
    for i in range(len(tensors)):
        subbatch = []
        part_num = tensors[i].shape[0] // sbs
        # if subbatch size is lower than the normal batch size
        if sbs < tensors[i].shape[0]:
            # partitioning batch with subbatch size
            subbatch = [tensors[i][j * sbs : j * sbs + sbs] for j in range(part_num)]
            # if there is any remainder in the batch
            if part_num * sbs < tensors[i].shape[0]:
                subbatch.append(tensors[i][part_num * sbs : ])
            subbatches.append(subbatch)
        else:
            subbatches.append([tensors[i]])
    
    return subbatches if len(tensors) > 1 else subbatches[0]

def save_training_model(args, save_content, modality, split, **contents):
    """
    Save the model state after interval of running epochs
    
    Inputs:
        args : Program arguments
        save_content : Accumulated save content
        modality, split : Recording purpose
        contetns : New contents with key and values to be added into the save_content
        
    Returns:
        None
    """
    
    # create a new entry with modality as key if it is yet existed
    if modality not in save_content.keys():
        save_content[modality] = {}
    
    # update the content of current modality/split
    save_content[modality].update({
        'split' + str(split) : contents
    })

    save_path = args.savename + '.pth.tar'
    #print('Saving at epoch', epoch + 1)
    torch.save({
            'args' : args,
            'content' : save_content
            }, save_path)
    

def train_model(args, device, model, dataloaders, optimizer, criterion, scheduler = None, 
                pretrained_content = None, modality = None, split = None, save_content = None):
    """
    This function trains (and validates if available) network with appointed training set, optmizer, criterion
    and scheduler (if available), and save the model after running for an interval of epochs, as well as resuming
    the training with intermediate model state
    
    Inputs:
        args : arguments dict passed from main function
        device : device id to be used in training
        model : model object to be trained
        dataloaders : dataloader dict containing dataloaders for training (and validation) datasets
        optimizer : optimizer object for parameters learning
        criterion : criterion object for computing loss
        scheduler : scheduler object for learning rate decay
        pretrained_content : an intermediate model state previously saved
        modality, split : recording purpose
        save_content : current content to be saved
        
    Outputs:
        train_loss : list of training loss for each epoch
        train_acc : list of training accuracy for each epoch
        time_elapsed : time taken in training
    """
    
    # loading those forsaken guys if halfway-pretrained model is identified
    if pretrained_content is not None:
        train_loss = pretrained_content[modality]['split'+str(split)]['train_loss']
        train_acc = pretrained_content[modality]['split'+str(split)]['train_acc']
        start = time.time() - pretrained_content[modality]['split'+str(split)]['train_elapsed']
        epoch = pretrained_content[modality]['split'+str(split)]['epoch']
        # loading model is already done in main function
        optimizer.load_state_dict(pretrained_content[modality]['split'+str(split)]['opt_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(pretrained_content[modality]['split'+str(split)]['sch_dict'])
        if pretrained_content[modality]['split'+str(split)]['stream_weight'] is not None:
            model.stream_weights = pretrained_content[modality]['split'+str(split)]['stream_weight']
            
    # else initializing them with emptiness
    else:
        train_loss = []
        train_acc = []
        start = time.time()
        epoch = 0
    
    if args.verbose2:
        print('Starting to train model.......')
    
    for epoch in range(epoch, args.epoch):
            
        for phase in ['train', 'val'] if args.validation_mode else ['train']:
            
            batch = 0
            total_batch = len(dataloaders[phase].dataset) // args.batch_size
            
            # reset the loss and accuracy
            current_loss = 0
            current_correct = 0
            
            # initialize the model with respective mode
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()
            else:
                model.eval()
                
            #print(len(dataloaders[phase].dataset))
            
            if isinstance(model, R2Plus1DNet):
            
                # for each mini batch of dataset
                for inputs, labels in dataloaders[phase]:
                    
                    #print(inputs.shape, labels.shape)
                    print('Current batch', str(batch), '/', str(total_batch), end = '\r')
                    batch += 1
                                
                    # retrieve the inputs and labels and send to respective computing devices
                    inputs = inputs.to(device)
                    labels = labels.long().to(device)
                    optimizer.zero_grad()
                    
                    # partioning each batch into subbatches to fit into memory
                    sub_inputs, sub_labels = generate_subbatches(args.subbatch_size, inputs, labels)
                    
                                        
                    # with enabling gradient descent on parameters during training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = torch.tensor([], dtype = torch.float).to(device)
                        for sb in range(len(sub_inputs)):
                            #print(sub_inputs[sb].shape)
                            # compute the final scores
                            #print(sub_inputs[sb].shape)
                            outputs = model(sub_inputs[sb])
                            
                            # transforming outcome from a series of scores to a single scalar index
                            # indicating the index where the highest score held
                            _, preds = torch.max(outputs['SCORES'], 1)
                            
                            # compute the loss
                            loss = criterion(outputs['SCORES'], sub_labels[sb])
                            
                            loss.backward()
                            
                            # accumulate loss and true positive for the current subbatch
                            current_loss += loss.item() * sub_inputs[sb].size(0)
                            current_correct += torch.sum(preds == sub_labels[sb].data)
                        
                        
                        # update parameters if it is training phase
                        if phase == 'train':
                            optimizer.step()
                        
            else:
                
                # for each mini batch of dataset
                for rgbX, flowX, labels in dataloaders['train']:
                
                    #print(inputs.shape, labels.shape)
                    print('Current batch', str(batch), '/', str(total_batch), end = '\r')
                    batch += 1
                                
                    # retrieve the inputs and labels and send to respective computing devices
                    rgbX = rgbX.to(device)
                    flowX = flowX.to(device)
                    labels = labels.long().to(device)
                    optimizer.zero_grad()
                    
                    # partioning each batch into subbatches to fit into memory
                    sub_rgbX, sub_flowX, sub_labels = generate_subbatches(args.subbatch_size, rgbX, flowX, labels)
                    
                    # with enabling gradient descent on parameters during training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = torch.tensor([], dtype = torch.float).to(device)
                        for sb in range(len(sub_rgbX)):
                            
                            # compute the final scores
                            outputs = model(sub_rgbX[sb], sub_flowX[sb])
                            
                            # transforming outcome from a series of scores to a single scalar index
                            # indicating the index where the highest score held
                            _, preds = torch.max(outputs['FUSION_SCORES'], 1)
#                            _, preds1 = torch.max(outputs['RGB_SCORES'], 1)
#                            _, preds2 = torch.max(outputs['FLOW_SCORES'], 1)
#                            
#                            print(preds, preds1, preds2)
#                            print(list(model.rgb_net.parameters())[0].requires_grad)
                            
                            # compute the loss
                            if not model.pretrained_streams:
                                rgb_loss = criterion(outputs['RGB_SCORES'], sub_labels[sb])
                                flow_loss = criterion(outputs['FLOW_SCORES'], sub_labels[sb])
                                
                            fusion_loss = criterion(outputs['FUSION_SCORES'], sub_labels[sb])
                            
                            # accumulate loss and true positive for the current subbatch
                            current_loss += fusion_loss.item() * sub_rgbX[sb].size(0)
                            current_correct += torch.sum(preds == sub_labels[sb].data)
                            
                            if not model.pretrained_streams:
                                # backprop on stream network first
                                rgb_loss.backward(retain_graph = True)
                                flow_loss.backward(retain_graph = True)
                                
                                # freeze the stream before backprop on fusion network
                                model.freeze_stream()
                                fusion_loss.backward()
                                model.freeze_stream(unfreeze = True)
                                
                            else:
                                fusion_loss.backward()
#                                                        
#                            _, preds = torch.max(outputs, 1)
#                            
#                            # compute the loss
#                            loss = criterion(outputs, sub_labels[sb])
#                            
#                            loss.backward()
#                            
#                            # accumulate loss and true positive for the current subbatch
#                            current_loss += loss.item() * sub_rgbX[sb].size(0)
#                            current_correct += torch.sum(preds == sub_labels[sb].data)
                        
                        # update parameters if it is training phase
                        if phase == 'train':
                            optimizer.step()
                
            # compute the loss and accuracy for the current batch
            epoch_loss = current_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(current_correct) / len(dataloaders[phase].dataset)
            
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            
            if args.verbose2:
                #print(f'Epoch {epoch} | Phase {phase} | Loss {epoch_loss:.4f} | Accuracy {epoch_acc:.2f}')
                print('Epoch %d | Phase %s | Loss %.4f | Accuracy %.4f' % (epoch + 1, phase, epoch_loss, epoch_acc))
        
        # time to save these poor guys
        # dun wanna losing them again
        if (epoch + 1) % args.save_interval == 0 and args.save:
            
            save_training_model(args, save_content, modality, split, 
                                train_acc = train_acc,
                                train_loss = train_loss,
                                train_elapsed = time.time() - start,
                                state_dict = model.state_dict(),
                                stream_weight = model.stream_weights if isinstance(model, FusionNet2) else None, 
                                opt_dict = optimizer.state_dict(),
                                sch_dict = scheduler.state_dict() if scheduler is not None else {},
                                epoch = epoch + 1)
            
    # display the time elapsed
    time_elapsed = time.time() - start    
    #print(f"Training completein {int(time_elapsed//3600)}h {int((time_elapsed%3600)//60)}m {int(time_elapsed %60)}s")
    print("Training completein %d h %d m %d s" % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)))
    
    return train_loss, train_acc, time_elapsed
