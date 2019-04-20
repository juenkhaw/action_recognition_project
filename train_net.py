# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:42:57 2019

@author: Juen
"""

import time
from math import ceil

import torch

from network_r2p1d import R2Plus1DNet

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

def save_training_model(args, key, save_content, **contents):
    """
    Save the model state after interval of running epochs
    
    Inputs:
        args : Program arguments
        save_content : Accumulated save content
        contetns : New contents with key and values to be added into the save_content
        
    Returns:
        None
    """
    
    # create/update the current training contents
    save_content[key] = contents   
    
    save_path = args.savename + '.pth.tar'
    #print('Saving at epoch', epoch + 1)
    torch.save(save_content, save_path)

def train_stream(args, device, model, dataloaders, optimizer, criterion, scheduler = None, save_content = None):
    
    subbatch_sizes = {'train' : args.subbatch_size, 'val' : args.sub_test_batch_size}
    
    assert(save_content is not None)
    
    # load the model state that is to be continued for training
    if args.load_model is not None and args.train:
        
        content = torch.load(args.load_model)['train']
        
        print('\n********* LOADING STATE ***********', 
              '\nModel Path =', args.load_model, '\nLast Epoch =', content['epoch'])
        
        assert(content['epoch'] < args.epoch)
        
        # load the state model into the network and other modules
        model.load_state_dict(content['state_dict'])
        optimizer.load_state_dict(content['opt_dict'])
        scheduler.load_state_dict(content['sch_dict'])
        epoch = content['epoch']
        losses = content['losses']
        accs = content['accuracy']
        start = content['train_elapsed']
        
        del content
        
        print('************* LOADED **************')
    
    else:
        losses = {'train' : [], 'val': []}
        accs = {'train': [], 'val' : []}
        epoch = 0
        start = time.time()
    
    for epoch in range(epoch, args.epoch):
        
        for phase in ['train', 'val']:
            
            batch = 1
            total_batch = int(ceil(len(dataloaders[phase].dataset) / args.batch_size))
            
            # reset the loss and accuracy
            current_loss = 0
            current_correct = 0
            
            # put the model into appropriate mode
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            # for each mini batch of dataset
            for inputs, labels in dataloaders[phase]:
                
                print('Phase', phase, '| Current batch', str(batch), '/', str(total_batch), end = '\r')
                batch += 1
                
                # place the input and label into memory of gatherer unit
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                
                # format validation input volume
                if phase == 'val':
                    inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], 
                             inputs.shape[4], inputs.shape[5])
        
                # partioning each batch into subbatches to fit into memory
                sub_inputs, sub_labels = generate_subbatches(subbatch_sizes[phase], inputs, labels)
            
                # with enabling gradient descent on parameters during training phase
                with torch.set_grad_enabled(phase == 'train'):
                
                    outputs = torch.tensor([], dtype = torch.float).to(device)
                    sb = 0
                
                    for sb in range(len(sub_inputs)):
                    
                        output = model(sub_inputs[sb])
                        
                        if phase == 'train':
                            # transforming outcome from a series of scores to a single scalar index
                            # indicating the index where the highest score held
                            _, preds = torch.max(output['FC'], 1)
                            
                            # compute the loss
                            loss = criterion(output['FC'], sub_labels[sb])
                            
                            # accumulate gradients on parameters
                            loss.backward()
                            
                            # accumulate loss and true positive for the current subbatch
                            current_loss += loss.item() * sub_inputs[sb].size(0)
                            current_correct += torch.sum(preds == sub_labels[sb].data)
                            
                        else:
                            # append the validation result until all subbatches are tested on
                            outputs = torch.cat((outputs, output['FC']))
                        
                    # avearging over validation results and compute val loss
                    if phase == 'val':
                        
                        outputs = torch.mean(torch.reshape(outputs, (labels.shape[0], 10, -1)), dim = 1)
                        current_loss += criterion(outputs, labels) * labels.shape[0]
                        
                        _, preds = torch.max(outputs, 1)
                        current_correct += torch.sum(preds == labels.data)
            
                    # update parameters
                    if phase == 'train':
                        optimizer.step()
            
            # compute the loss and accuracy for the current batch
            epoch_loss = current_loss / len(dataloaders['train'].dataset)
            epoch_acc = float(current_correct) / len(dataloaders['train'].dataset)
            
            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            
            # get current learning rate
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            
            # step on reduceLRonPlateau with val acc
            if scheduler is not None and phase == 'val':
                scheduler.step(epoch_acc)
            
        if args.verbose2:
            #print(f'Epoch {epoch} | Phase {phase} | Loss {epoch_loss:.4f} | Accuracy {epoch_acc:.2f}')
            print('Epoch %d | lr %.1E | TrainLoss %.4f | ValLoss %.4f | TrainAcc %.4f | ValAcc %.4f' % 
                  (epoch + 1, lr, losses['train'][epoch], losses['val'][epoch], 
                   accs['train'][epoch], accs['val'][epoch]))
            
        # time to save these poor guys
        # dun wanna losing them again
        if (epoch + 1) % args.save_interval == 0 and args.save:
            
            save_training_model(args, 'train', save_content,  
                                    accuracy = accs,
                                    losses = losses,
                                    train_elapsed = time.time() - start,
                                    state_dict = model.state_dict(),
                                    opt_dict = optimizer.state_dict(),
                                    sch_dict = scheduler.state_dict() if scheduler is not None else {},
                                    epoch = epoch + 1
                                    )
            
    # display the time elapsed
    time_elapsed = time.time() - start
    if args.verbose2:
        print('\n\n+++++++++ TRAINING RESULT +++++++++++',
              '\nElapsed Time = %d h %d m %d s' % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)), 
              '\n+++++++++++++++++++++++++++++++++++++')
    #print(f"Training completein {int(time_elapsed//3600)}h {int((time_elapsed%3600)//60)}m {int(time_elapsed %60)}s")
    #print("Training completein %d h %d m %d s" % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)))
    
    return losses, accs, time_elapsed