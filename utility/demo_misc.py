# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:11:52 2019

@author: Juen
"""
import numpy as np
import matplotlib.pyplot as plt

def get_class_label(mapfile_path):
    # read in the UCF class labels for visualization purpose
    class_f = open(mapfile_path, 'r')
    class_raw_str = class_f.read().split('\n')[:-1]
    class_label = [raw_str.split(' ')[1] for raw_str in class_raw_str]
    
    return class_label

def get_prediction(scores, top_k, class_label):
    
    scores = scores.cpu().detach().numpy()
    indices = np.argsort(scores, axis = 1)[0, (101 - top_k) :][::-1]
    return {class_label[index] : scores[0, index] for index in indices}

def plotlossgraph(data, x_label, y_label, legends, cap = 45):
    x = list(range(cap))
    plotline = ['r--', 'b--', 'g--', 'k--']
    handles = []
    
    assert(len(data) < 5)
    
    for i in range(len(data)):
        y = data[i][:cap]
        h, = plt.plot(x, y, plotline[i], label = legends[i])
        handles.append(h)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles = handles)
    plt.show()
    
def plotaccgraph(data, x_label, y_label, legends, cap = 45):
    x = list(range(cap))
    trainline = ['r--', 'b--', 'g--']
    testline = ['r', 'b', 'g']
    handles = []
    
    assert(len(data) < 6)
    
    for i in range(len(data) // 2):
        train_acc = data[i * 2][:cap]
        test_acc = [data[i * 2 + 1] for j in range(cap)]
        h, = plt.plot(x, train_acc, trainline[i], label = legends[i * 2])
        handles.append(h)
        h, = plt.plot(x, test_acc, testline[i], label = legends[i * 2 + 1])
        handles.append(h)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(handles = handles)
    plt.show()