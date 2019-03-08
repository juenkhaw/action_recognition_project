# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:26:48 2019

@author: Juen
"""
from __future__ import print_function
import time
import numpy as np

import torch

from network_r2p1d import R2Plus1DNet
from train_net import generate_subbatches

def test_model(args, device, model, test_dataloader, load_mode, top_acc):
    """
    This function evaluates trained model on testing dataset
    
    Inputs:
        args : arguments dict passed from main function
        device : device id to be used in training
        model : model object to be trained
        test_dataloader : dataloader object containing testing dataset
        load_mode : [clip/video] performs clip/video level prediction
        top_acc : top N predicted labels to be considered as predicted results
        
    Outputs:
        predicted : list of predicted labels for each testing batch
        test_acc : Testing accuracy
        time_elapsed : Time taken in testing
    """
    predicted = []
    start = time.time()
    
    if args.verbose2:
        print('Starting to test model.......')
        
    test_correct = 0
    test_acc = 0
    
    batch = 0
    total_batch = len(test_dataloader.dataset) // args.test_batch_size
    
    if isinstance(model, R2Plus1DNet):
    
        for inputs, labels in test_dataloader:
            print('Current batch', str(batch), '/', str(total_batch), end = '\r')
            batch += 1
            #print(inputs.shape, labels.shape)
            # if loading series of clip, reshaping the inputs tensor to fit into the model
            # from [sample, clips, channel, frame, h, w] to [sample * clips, -1]
            current_batch_size = inputs.shape[0]
            if load_mode == 'video':
                clips_per_video = inputs.shape[1]
                inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], 
                                     inputs.shape[4], inputs.shape[5])
            
            inputs = inputs.to(device)
            
            # reshaping labels tensor
            labels = np.array(labels).reshape(len(labels), 1)
            
            # partioning each batch into subbatches to fit into memory
            sub_inputs = generate_subbatches(args.sub_test_batch_size, inputs)[0]
            
            with torch.set_grad_enabled(False):
                outputs = torch.tensor([], dtype = torch.float).to(device)
                for sb in range(len(sub_inputs)):
                    #print(sub_inputs[sb].shape)
                    # use model to predict scores
                    # copy the tensor to host memory before converting to np array
                    output = model(sub_inputs[sb])
                    outputs = torch.cat((outputs, output))
                
            outputs = outputs.cpu().detach().numpy()
            
            # average the scores for each classes across all clips that belong to the same video
            averaged_score = np.average(np.array(np.split(outputs, current_batch_size)), axis = 1)
            
            # retrieve the label index with the top-N scores
            top_k_indices = np.argsort(averaged_score, axis = 1)[:, ::-1][:, :top_acc]
            predicted.extend(top_k_indices)
            
            # compute number of matches between predicted labels and true labels
            test_correct += np.sum(top_k_indices == np.array(labels))
            
    else:
            
        for rgbX, flowX, labels in test_dataloader:
            print('Current batch', str(batch), '/', str(total_batch), end = '\r')
            batch += 1
            #print(inputs.shape, labels.shape)
            # if loading series of clip, reshaping the inputs tensor to fit into the model
            # from [sample, clips, channel, frame, h, w] to [sample * clips, -1]
            current_batch_size = rgbX.shape[0]
            if load_mode == 'video':
                clips_per_video = rgbX.shape[1]
                rgbX = rgbX.view(-1, rgbX.shape[2], rgbX.shape[3], 
                                     rgbX.shape[4], rgbX.shape[5])
                flowX = flowX.view(-1, flowX.shape[2], flowX.shape[3], 
                                     flowX.shape[4], flowX.shape[5])
            
            rgbX = rgbX.to(device)
            flowX = flowX.to(device)
            
            # reshaping labels tensor
            labels = np.array(labels).reshape(len(labels), 1)
            
            # partioning each batch into subbatches to fit into memory
            sub_rgbX, sub_flowX = generate_subbatches(args.sub_test_batch_size, rgbX, flowX)
            
            with torch.set_grad_enabled(False):
                outputs = torch.tensor([], dtype = torch.float).to(device)
                for sb in range(len(sub_rgbX)):
                    #print(sub_rgbX[sb].shape, sub_flowX[sb].shape)
                    # use model to predict scores
                    # copy the tensor to host memory before converting to np array
                    output = model(sub_rgbX[sb], sub_flowX[sb])
                    outputs = torch.cat((outputs, output))
                
            outputs = outputs.cpu().detach().numpy()
            
            # average the scores for each classes across all clips that belong to the same video
            averaged_score = np.average(np.array(np.split(outputs, current_batch_size)), axis = 1)
            
            # retrieve the label index with the top-N scores
            top_k_indices = np.argsort(averaged_score, axis = 1)[:, ::-1][:, :top_acc]
            predicted.extend(top_k_indices)
            
            # compute number of matches between predicted labels and true labels
            test_correct += np.sum(top_k_indices == np.array(labels))
    
    # compute accuracy over predictions on current batch
    test_acc = float(test_correct) / len(test_dataloader.dataset)
    
    # display the time elapsed in testing
    time_elapsed = time.time() - start
    #print(f"Testing complete in {int(time_elapsed//3600)}h {int((time_elapsed%3600)//60)}m {int(time_elapsed %60)}s")
    print('\nTesting acc %.4f' % (test_acc))
    print("Testing complete in %d h %d m %d s" % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)))

    return predicted, test_acc, time_elapsed
