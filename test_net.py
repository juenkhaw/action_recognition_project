# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:46:45 2019

@author: Juen
"""
from __future__ import print_function
import time
import numpy as np
from math import ceil

import torch

from network_r2p1d import R2Plus1DNet
from train_net import generate_subbatches

def test_stream(args, device, model, test_dataloader, mode = 'test'):
    
    all_scores = []
    test_correct = [0, 0]
    test_acc = [0, 0]
    
    batch = 1
    total_batch = int(ceil(len(test_dataloader.dataset) / test_dataloader.batch_size))
    
    start = time.time()
    
    # put model into evaluation mode
    model.eval()
    
    # we want softmax scores, not activations
    model._endpoint = ['SCORES']
    
    for inputs, labels in test_dataloader:
        print('Phase test | Current batch =', str(batch), '/', str(total_batch), end = '\r')
        batch += 1
        
        # getting dimensions of input
        batch_size = inputs.shape[0]
        clips_per_video = inputs.shape[1]
        inputs = inputs.view(-1, inputs.shape[2], inputs.shape[3], 
                             inputs.shape[4], inputs.shape[5])
        
        # moving inputs to gatherer unit
        inputs = inputs.to(device)
            
        # reshaping labels tensor
        labels = np.array(labels).reshape(len(labels), 1)
        
        # partioning each batch into subbatches to fit into memory
        sub_inputs = generate_subbatches(args.sub_test_batch_size, inputs)
        
        # with gradient disabled, perform testing on each subbatches
        with torch.set_grad_enabled(False):
            
            outputs = torch.tensor([], dtype = torch.float).to(device)
            
            for sb in range(len(sub_inputs)):

                output = model(sub_inputs[sb])
                outputs = torch.cat((outputs, output['SCORES']))
        
        # detach the predicted scores         
        outputs = outputs.cpu().detach().numpy()
            
        # average the scores for each classes across all clips that belong to the same video
        averaged_score = np.average(np.array(np.split(outputs, batch_size)), axis = 1)
        
        # concet into all scores
        if all_scores == []:
            all_scores = averaged_score
        else:
            all_scores = np.concatenate((all_scores, averaged_score), axis = 0)
        
        # retrieve the label index with the top-5 scores
        top_k_indices = np.argsort(averaged_score, axis = 1)[:, ::-1][:, :5]
        
        # compute number of matches between predicted labels and true labels
        test_correct[0] += np.sum(top_k_indices[:, 0] == np.array(labels).ravel())
        test_correct[1] += np.sum(top_k_indices == np.array(labels))
    
    # compute accuracy over predictions on current batch
    test_acc[0] = float(test_correct[0]) / len(test_dataloader.dataset)
    test_acc[1] = float(test_correct[1]) / len(test_dataloader.dataset)
    
    # display the time elapsed in testing
    time_elapsed = time.time() - start
    if args.verbose2 and mode == 'test':
        print('\n\n+++++++++ TESTING RESULT +++++++++++',
              '\nElapsed Time = %d h %d m %d s' % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)),
              '\nTop-1 Accuracy = %.4f' % (test_acc[0]),
              '\nTop-5 Accuracy = %.4f' % (test_acc[1]),
              '\n++++++++++++++++++++++++++++++++++++')
    #print('\nTesting acc %.4f %.4f' % test_acc[0], test_acc[1])
    #print("Testing complete in %d h %d m %d s" % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)))
    
    return all_scores, test_acc, time_elapsed