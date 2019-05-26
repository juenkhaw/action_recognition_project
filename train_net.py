# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:42:57 2019

@author: Juen
"""

import time
from math import ceil
import gc

import torch

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

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

def train_stream(args, device, model, dataloaders, optimizer, criterion, scheduler, save_content):
    
    subbatch_sizes = {'train' : args.subbatch_size, 'val' : args.val_subbatch_size}
    subbatch_count = {'train' : args.batch_size, 'val' : args.val_subbatch_size}
    best_model = {'epoch' : 0, 
                  'model_state' : None, 
                  'train_loss' : float('inf'), 
                  'val_loss' : float('inf')}
        
    # load the model state that is to be continued for training
    if args.load_model is not None and not args.pretrain:
        
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
        start = time.time()
        prev_elapsed = content['train_elapsed']
        
        del content
        torch.cuda.empty_cache()
        
        print('************* LOADED **************')
    
    else:
        losses = {'train' : [], 'val': []}
        accs = {'train': [], 'val' : []}
        epoch = 0
        start = time.time()
        prev_elapsed = 0
    
    for epoch in range(epoch, args.epoch):
        
        for phase in ['train', 'val']:
            
            batch = 1
            total_batch = int(ceil(len(dataloaders[phase].dataset) / subbatch_count[phase]))
            
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
                
                #print('Phase', phase, '| Current batch', str(batch), '/', str(total_batch), end = '\r')
                print('Phase', phase, '| Current batch', str(batch), '/', str(total_batch), end = '\n')
                batch += 1
                
                # place the input and label into memory of gatherer unit
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
        
                # partioning each batch into subbatches to fit into memory
                if phase == 'train':
                    sub_inputs, sub_labels = generate_subbatches(subbatch_sizes[phase], inputs, labels)
                else:
                    sub_inputs = [inputs]
                    sub_labels = [labels]
                
                del inputs
                torch.cuda.empty_cache()
            
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
                        
                        current_loss += criterion(outputs, labels).item() * labels.shape[0]
                        
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
        
        if losses['val'][epoch] <= best_model['val_loss']:
            #if losses['train'][epoch] <= best_model['train_loss']:
            best_model = {'epoch' : epoch + 1, 
                  'model_state' : model.state_dict(), 
                  'train_loss' : losses['train'][epoch], 
                  'val_loss' : losses['val'][epoch]}
            
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
                                    epoch = epoch + 1,
                                    best = best_model
                                    )
            
    # display the time elapsed
    time_elapsed = time.time() - start + prev_elapsed
    if args.verbose2:
        print('\n\n+++++++++ TRAINING RESULT +++++++++++',
              '\nElapsed Time = %d h %d m %d s' % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)), 
              '\n+++++++++++++++++++++++++++++++++++++')
    #print(f"Training completein {int(time_elapsed//3600)}h {int((time_elapsed%3600)//60)}m {int(time_elapsed %60)}s")
    #print("Training completein %d h %d m %d s" % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)))
    
    return losses, accs, time_elapsed, best_model

def train_pref_fusion(args, device, models, dataloaders, optimizer, criterion, scheduler, save_content):
    
    subbatch_sizes = {'train' : args.subbatch_size, 'val' : args.val_subbatch_size}
    subbatch_count = {'train' : args.batch_size, 'val' : args.val_subbatch_size}
    
    losses = {'train' : [], 'val': []}
    accs = {'train' : [], 'val': []}
    epoch = 0
    start = time.time()
    prev_elapsed = 0
    
    best_model = {'epoch' : 0, 
                  'model_state' : None, 
                  'train_loss' : float('inf'), 
                  'val_loss' : float('inf')}
    
    # LOAD INTERMEDIATE STATE
    if (args.load_stream is not [] or args.load_fusion is not []) and not args.pretrain:
        
        print('\n********* LOADING STATE ***********', 
          '\nFusion Mode =', 'End to End' if args.archi == 'e2e' else 'Pretrained Streams'
          '\nStream Paths =', args.load_stream, '\nFusion Paths =', args.load_fusion)
        
        # load fully trained stream states
        assert(len(args.load_stream) == 2)
        
        rgb_state = torch.load(args.load_stream[0])['train']['state_dict']
        flow_state = torch.load(args.load_stream[1])['train']['state_dict']
        
        models['rgb'].load_state_dict(rgb_state)
        models['flow'].load_state_dict(flow_state)
        
        del rgb_state
        del flow_state
        torch.cuda.empty_cache()
        
        # load intermediate state of fusion network if there is
        if args.load_fusion != []:
            assert(len(args.load_fusion) == 1)
            
            fusion_state = torch.load(args.load_fusion[0])['train']
            models['fusion'].load_state_dict(fusion_state['state_dict'])
            
            optimizer.load_state_dict(fusion_state['opt_dict'])
            scheduler.load_state_dict(fusion_state['sch_dict'])
            
            epoch = fusion_state['epoch']
            losses = fusion_state['losses']
            accs = fusion_state['accuracy']
            start = fusion_state['train_elapsed']
            start = time.time()
            prev_elapsed = fusion_state['train_elapsed']
            
            del fusion_state
            torch.cuda.empty_cache()
            
        print('************* LOADED **************')
    
    # freeze the streams for all the time
    models['rgb'].freezeAll()
    models['flow'].freezeAll()
    models['rgb'].eval()
    models['flow'].eval()
    
    # ensure that the target epoch is higher from pretrained state
    assert(epoch < args.epoch)
    
    for epoch in range(epoch, args.epoch):
        
        for phase in ['train', 'val']:
            
            batch = 1
            total_batch = int(ceil(len(dataloaders[phase].dataset) / subbatch_count[phase]))
            
            # reset the loss and accuracy
            current_loss = 0
            current_correct = 0
            
            if phase == 'train':
                models['fusion'].train()
            else:
                models['fusion'].eval()
                
            # for each mini batch of dataset
            for rgbX, flowX, labels in dataloaders[phase]:
                
                #print('Phase', phase, '| Current batch', str(batch), '/', str(total_batch), end = '\r')
                print('Phase', phase, '| Current batch', str(batch), '/', str(total_batch), end = '\n')
                batch += 1
                
                # place the input and label into memory of gatherer unit
                rgbX = rgbX.to(device)
                flowX = flowX.to(device)
                labels = labels.long().to(device)
                
#                # format validation input volume
#                if phase == 'val':
#                    rgbX = rgbX.view(-1, rgbX.shape[2], rgbX.shape[3], 
#                             rgbX.shape[4], rgbX.shape[5])
#                    flowX = flowX.view(-1, flowX.shape[2], flowX.shape[3], 
#                             flowX.shape[4], flowX.shape[5])
                    
                # partioning each batch into subbatches to fit into memory
                sub_rgbX, sub_flowX, sub_labels = generate_subbatches(subbatch_sizes[phase], rgbX, flowX, labels)
                assert(len(sub_rgbX) == len(sub_flowX))
                
                del rgbX
                del flowX
                torch.cuda.empty_cache()
                
                # clear out the gradient in parameters
                optimizer.zero_grad()
                
                # with enabling gradient descent on parameters during training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = torch.tensor([], dtype = torch.float).to(device)
                    sb = 0
                    
                    for sb in range(len(sub_rgbX)):
                        
                        torch.cuda.empty_cache()
                        
                        rgb_out = models['rgb'](sub_rgbX[sb])
                        flow_out = models['flow'](sub_flowX[sb])
                        fusion_out = models['fusion'](rgb_out, flow_out)
                        
                        if phase == 'train':
                            
                            # transforming outcome from a series of scores to a single scalar index
                            # indicating the index where the highest score held
                            _, preds = torch.max(fusion_out['FC'], 1)
                            
                            fusion_loss = criterion(fusion_out['FC'], sub_labels[sb])
                            
                            # accumulate loss and true positive for the current subbatch
                            current_loss += fusion_loss.item() * sub_rgbX[sb].size(0)
                            current_correct += torch.sum(preds == sub_labels[sb].data)
                            
                            # direct backprop in fusion network as stream is always froze
                            fusion_loss.backward()
                            
                        else:
                            # append the validation result until all subbatches are tested on
                            outputs = torch.cat((outputs, fusion_out['FC']))
                    
                # avearging over validation results and compute val loss
                if phase == 'val':
                    
                    current_loss += criterion(outputs, labels).item() * labels.shape[0]
                        
                    _, preds = torch.max(outputs, 1)
                    current_correct += torch.sum(preds == labels.data)
                    
                else: # update parameters when phase == train
                    optimizer.step()
                    
            # compute the loss and accuracy for the current batch
            lr = {}

            epoch_loss = current_loss / len(dataloaders['train'].dataset)
            epoch_acc = float(current_correct) / len(dataloaders['train'].dataset)
            
            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            
            # get current learning rate
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                
            # step on reduceLRonPlateau with val acc
            if phase == 'val':
                scheduler.step(epoch_acc)
                
        if losses['val'][epoch] <= best_model['val_loss']:
            #if losses['train'][epoch] <= best_model['train_loss']:
            best_model = {'epoch' : epoch + 1, 
                  'model_state' : models['fusion'].state_dict(), 
                  'train_loss' : losses['train'][epoch], 
                  'val_loss' : losses['val'][epoch]}
                
        if args.verbose2:
            print('Epoch %d | Network %s | lr %.1E | TrainLoss %.4f | ValLoss %.4f | TrainAcc %.4f | ValAcc %.4f' % 
                  (epoch + 1, 'Fusion', lr, losses['train'][epoch], losses['val'][epoch], 
                   accs['train'][epoch], accs['val'][epoch]))
            
    # display the time elapsed
    time_elapsed = time.time() - start + prev_elapsed
    if args.verbose2:
        print('\n\n+++++++++ TRAINING RESULT +++++++++++',
              '\nElapsed Time = %d h %d m %d s' % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)), 
              '\n+++++++++++++++++++++++++++++++++++++')
        
    return losses, accs, time_elapsed
    
def train_e2e_fusion(args, device, models, dataloaders, optimizers, criterions, schedulers, save_content):
    
    subbatch_sizes = {'train' : args.subbatch_size, 'val' : args.sub_test_batch_size}
    
    # LOAD INTERMEDIATE STATE

#def train_fusion(args, device, models, dataloaders, optimizers, criterion, schedulers = None, save_content = None):
#    
#    subbatch_sizes = {'train' : args.subbatch_size, 'val' : args.sub_test_batch_size}
#    
#    assert(save_content is not None)
#    
#    if args.load_model is not None:
#        
#        content = None
#    
#        if args.train_option == 'pref' and len(args.load_model) == 2:
#            
#            content = torch.load(args.load_model[1])['train']
#            
#            models['fusion'].load_state_dict(content['state_dict'])
#            optimizers['fusion'].load_state_dict(content['opt_dict'])
#            if schedulers is not None:
#                schedulers['fusion'].load_state_dict(content['sch_dict'])
#                    
#        elif args.train_option == 'tranf' and len(args.load_model) == 1:
#            
#            content = torch.load(args.load_model[0])['train']
#            
#            models['rgb'].load_state_dict(content['rgb']['state_dict'])
#            optimizers['rgb'].load_state_dict(content['rgb']['opt_dict'])
#            if schedulers is not None:
#                schedulers['rgb'].load_state_dict(content['rgb']['sch_dict'])
#                
#            models['flow'].load_state_dict(content['flow']['state_dict'])
#            optimizers['flow'].load_state_dict(content['flow']['opt_dict'])
#            if schedulers is not None:
#                schedulers['flow'].load_state_dict(content['flow']['sch_dict'])
#            
#            models['fusion'].load_state_dict(content['fusion']['state_dict'])
#            optimizers['fusion'].load_state_dict(content['fusion']['opt_dict'])
#            if schedulers is not None:
#                schedulers['fusion'].load_state_dict(content['fusion']['sch_dict'])
#                
#        if content is not None:
#     
#            epoch = content['epoch']
#            losses = content['losses']
#            accs = content['accuracy']
#            start = content['train_elapsed']
#            
#        else:
#            
#            losses = {'train' : {'fusion':[]}, 'val': {'fusion':[]}} if args.train_option == 'pref' else {'train' : {'rgb' : [], 'flow' : [], 'fusion': []}, 'val' : {'rgb' : [], 'flow' : [], 'fusion': []}}
#            accs = {'train' : {'fusion':[]}, 'val': {'fusion':[]}} if args.train_option == 'pref' else {'train' : {'rgb' : [], 'flow' : [], 'fusion': []}, 'val' : {'rgb' : [], 'flow' : [], 'fusion': []}}
#            epoch = 0
#            start = time.time()
#        
#        del content
#        
#    else:
#        losses = {'train' : {'fusion':[]}, 'val': {'fusion':[]}} if args.train_option == 'pref' else {'train' : {'rgb' : [], 'flow' : [], 'fusion': []}, 'val' : {'rgb' : [], 'flow' : [], 'fusion': []}}
#        accs = {'train' : {'fusion':[]}, 'val': {'fusion':[]}} if args.train_option == 'pref' else {'train' : {'rgb' : [], 'flow' : [], 'fusion': []}, 'val' : {'rgb' : [], 'flow' : [], 'fusion': []}}
#        epoch = 0
#        start = time.time()
#        
#    # freeze the stream network all the way if it is pretrained fusion
#    if args.train_option == 'pref':
#       models['rgb'].freeze()
#       models['flow'].freeze()
#       models['rgb'].train()
#       models['flow'].train()
#        
#    for epoch in range(epoch, args.epoch):
#        
#        for phase in ['train', 'val']:
#            
#            batch = 1
#            total_batch = int(ceil(len(dataloaders[phase].dataset) / args.batch_size))
#            
#            # reset the loss and accuracy
#            current_loss = {'fusion' : 0} if args.train_option == 'pref' else {'rgb':0, 'flow':0, 'fusion':0}
#            current_correct = {'fusion' : 0} if args.train_option == 'pref' else {'rgb':0, 'flow':0, 'fusion':0}
#            
#            # put the model into appropriate mode
#            if phase == 'train':
#                models['fusion'].train()
#                if args.train_option == 'tranf':
#                    models['rgb'].train()
#                    models['flow'].train()
#            else:
#                models['fusion'].eval()
#                if args.train_option == 'tranf':
#                    models['rgb'].eval()
#                    models['flow'].eval()
#                    
#            # for each mini batch of dataset
#            for rgbX, flowX, labels in dataloaders[phase]:
#                
#                print('Phase', phase, '| Current batch', str(batch), '/', str(total_batch), end = '\r')
#                batch += 1
#                
#                # place the input and label into memory of gatherer unit
#                rgbX = rgbX.to(device)
#                flowX = flowX.to(device)
#                labels = labels.long().to(device)
#                
#                # format validation input volume
#                if phase == 'val':
#                    rgbX = rgbX.view(-1, rgbX.shape[2], rgbX.shape[3], 
#                             rgbX.shape[4], rgbX.shape[5])
#                    flowX = flowX.view(-1, flowX.shape[2], flowX.shape[3], 
#                             flowX.shape[4], flowX.shape[5])
#                    
#                # partioning each batch into subbatches to fit into memory
#                sub_rgbX, sub_flowX, sub_labels = generate_subbatches(subbatch_sizes[phase], rgbX, flowX, labels)
#                assert(len(sub_rgbX) == len(sub_flowX))
#                
#                # clear out the gradient in parameters
#                optimizers['fusion'].zero_grad()
#                if args.train_option == 'tranf':
#                    optimizers['rgb'].zero_grad()
#                    optimizers['flow'].zero_grad()
#                
#                # with enabling gradient descent on parameters during training phase
#                with torch.set_grad_enabled(phase == 'train'):
#                    
#                    if args.train_option == 'pref':
#                        outputs = {'fusion' : torch.tensor([], dtype = torch.float).to(device)}
#                    else:
#                        outputs = {'rgb' : torch.tensor([], dtype = torch.float).to(device), 
#                                   'flow' : torch.tensor([], dtype = torch.float).to(device),
#                                   'fusion' : torch.tensor([], dtype = torch.float).to(device)}
#                    sb = 0
#                        
#                        if phase == 'train':
#                            
#                            # transforming outcome from a series of scores to a single scalar index
#                            # indicating the index where the highest score held
#                            _, preds = torch.max(fusion_out['FC'], 1)
#                            
#                            # compute loss
#                            if args.train_option == 'tranf':
#                                
#                                _, rgb_preds = torch.max(rgb_out['FC'], 1)
#                                _, flow_preds = torch.max(flow_out['FC'], 1)
#                                
#                                current_correct['rgb'] += torch.sum(rgb_preds == sub_labels[sb].data)
#                                current_correct['flow'] += torch.sum(flow_preds == sub_labels[sb].data)
#                                
#                                rgb_loss = criterion(rgb_out['FC'], sub_labels[sb])
#                                flow_loss = criterion(flow_out['FC'], sub_labels[sb])
#                                
#                                current_loss['rgb'] += rgb_loss.item() * sub_rgbX[sb].size(0)
#                                current_loss['flow'] += flow_loss.item() * sub_rgbX[sb].size(0)
#                                
#                            fusion_loss = criterion(fusion_out['FC'], sub_labels[sb])
#                            
#                            # accumulate loss and true positive for the current subbatch
#                            current_loss['fusion'] += fusion_loss.item() * sub_rgbX[sb].size(0)
#                            current_correct['fusion'] += torch.sum(preds == sub_labels[sb].data)
#                            
#                            # backward
#                            if args.train_option == 'tranf':
#                                
#                                # backprop on stream network
#                                rgb_loss.backward(retain_graph = True)
#                                flow_loss.backward(retain_graph = True)
#                                
#                                # freeze the stream before backprop on fusion network
#                                models['rgb'].freeze()
#                                models['flow'].freeze()
#                                
#                                fusion_loss.backward()
#                                
#                                models['rgb'].freeze(unfreeze = True)
#                                models['flow'].freeze(unfreeze = True)
#                                
#                            else:
#                                # direct backprop in fusion network as stream is always froze
#                                fusion_loss.backward()
#                            
#                        else:
#                            
#                            # append the validation result until all subbatches are tested on
#                            outputs['fusion'] = torch.cat((outputs['fusion'], fusion_out['FC']))
#                            if args.train_option == 'tranf':
#                                outputs['rgb'] = torch.cat((outputs['rgb'], rgb_out['FC']))
#                                outputs['flow'] = torch.cat((outputs['flow'], flow_out['FC']))
#                                
#                    # avearging over validation results and compute val loss
#                    if phase == 'val':
#                        
#                        for key in list(outputs.keys()):
#                            outputs[key] = torch.mean(torch.reshape(outputs[key], (labels.shape[0], 10, -1)), dim = 1)
#                            current_loss[key] += criterion(outputs[key], labels) * labels.shape[0]
#                            
#                            _, preds = torch.max(outputs[key], 1)
#                            current_correct[key] += torch.sum(preds == labels.data)
#                            
#                    # update parameters
#                    if phase == 'train': 
#                        for key in list(optimizers.keys()):
#                            optimizers[key].step()
#                            
#            # compute the loss and accuracy for the current batch
#            lr = {}
#            for key in list(current_loss.keys()):
#                epoch_loss = current_loss[key] / len(dataloaders['train'].dataset)
#                epoch_acc = float(current_correct[key]) / len(dataloaders['train'].dataset)
#                
#                losses[phase][key].append(epoch_loss)
#                accs[phase][key].append(epoch_acc)
#                
#                # get current learning rate
#                for param_group in optimizers[key].param_groups:
#                    lr[key] = param_group['lr']
#                    
#                # step on reduceLRonPlateau with val acc
#                if schedulers[key] is not None and phase == 'val':
#                    schedulers[key].step(epoch_acc)
#                    
#        if args.verbose2:
#            for key in list(losses['train'].keys()):
#                print('Epoch %d | Network %s | lr %.1E | TrainLoss %.4f | ValLoss %.4f | TrainAcc %.4f | ValAcc %.4f' % 
#                      (epoch + 1, key, lr[key], losses['train'][key][epoch], losses['val'][key][epoch], 
#                       accs['train'][key][epoch], accs['val'][key][epoch]))
#                
#    # display the time elapsed
#    time_elapsed = time.time() - start
#    if args.verbose2:
#        print('\n\n+++++++++ TRAINING RESULT +++++++++++',
#              '\nElapsed Time = %d h %d m %d s' % (int(time_elapsed//3600), int((time_elapsed%3600)//60), int(time_elapsed %60)), 
#              '\n+++++++++++++++++++++++++++++++++++++')
#        
#    return losses, accs, time_elapsed