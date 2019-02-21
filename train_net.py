# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:50:07 2019

@author: Juen
"""

import time
import torch

def train_model(args, device, model, dataloaders, optimizer, criterion, scheduler = None):
    """
    This function trains (and validates if available) network with appointed training set, optmizer, criterion
    and scheduler (if available)
    
    Inputs:
        args : arguments dict passed from main function
        device : device id to be used in training
        model : model object to be trained
        dataloaders : dataloader dict containing dataloaders for training (and validation) datasets
        optimizer : optimizer object for parameters learning
        criterion : criterion object for computing loss
        scheduler : scheduler object for learning rate decay
        
    Outputs:
        train_loss : list of training loss for each epoch
        train_acc : list of training accuracy for each epoch
        time_elapsed : time taken in training
    """
    
    train_loss = []
    train_acc = []
    start = time.time()
    epoch = 0
    
    if args.verbose2:
        print('Starting to train model.......')
    
    for epoch in range(epoch, args.epoch):
            
        for phase in ['train', 'val'] if args.validation_mode else ['train']:
            
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
            
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            
            if args.verbose2:
                #print(f'Epoch {epoch} | Phase {phase} | Loss {epoch_loss:.4f} | Accuracy {epoch_acc:.2f}')
                print('Epoch %d | Phase %s | Loss %.4f | Accuracy %.2f' % (epoch, phase, epoch_loss, epoch_acc))
    
    # display the time elapsed
    time_elapsed = time.time() - start    
    #print(f"Training completein {int(time_elapsed//3600)}h {int((time_elapsed%3600)//60)}m {int(time_elapsed %60)}s")
    print("Training completein %d h %d m %d s" % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)))
    
    return train_loss, train_acc, time_elapsed