# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:27:42 2019

@author: Juen
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from gbp_video_module import denormalize_buffer

def get_prediction(scores, top_k, class_label):
    
    scores = scores.cpu().detach().numpy()
    indices = np.argsort(scores, axis = 1)[0, (101 - top_k) :][::-1]
    return {class_label[index] : scores[0, index] for index in indices}

def plt_maps(args, test_frame, x_grads, pos_sal, neg_sal, label):
    
    contents = [test_frame.cpu().detach().numpy(), x_grads, pos_sal, neg_sal]
    titles = ['Original Frame', 'Gradient Map', 'Positive Saliency', 'Negative Saliency']

    col = 4
    row = args.frame_num
    
    fig = plt.figure(figsize = (col * 3, row * 3))
    plt.title(label + ' (' + str(args.test_label + 1) + ')\n'
              + str(row) + ' frames (' + args.test_video + ')\n',
              fontdict = {'fontsize' : 16}, loc = 'left')
    plt.axis('off')
    
    for i in range(0, test_frame.shape[2] * col):
        
        img = contents[i % col][0, :, i // col].transpose((1, 2, 0))
        img = denormalize_buffer(img)
        
        ax = fig.add_subplot(row, col, i + 1)
        if i < 4:
            ax.set_title(titles[i % col])
        
        plt.axis('off')
        plt.imshow(img)
        
    plt.subplots_adjust(hspace=0, wspace=0)
    
    return plt

def cv2_maps(args, test_frame, x_grads, pos_sal, neg_sal, label):
    
    contents = [test_frame.cpu().detach().numpy(), x_grads, pos_sal, neg_sal]
    titles = ['Original Frame', 'Gradient Map', 'Positive Saliency', 'Negative Saliency']
    
    # assgin output image size
    img_h = test_frame.shape[3] * 2
    img_w = test_frame.shape[4] * 2
    
    text_h = 80
    btm_space = 10
    space = 10
    
    out_h = img_h + text_h + btm_space
    out_w = img_w * 4 + 5 * space
    
    output_frames = np.ones((out_h, out_w, 3), np.uint8)
    output_frames *= 255
    
    # draw title
    cv2.putText(output_frames, label + ' (' + str(args.test_label + 1) + ')', 
                (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), thickness = 1)
    cv2.putText(output_frames, str(args.frame_num) + ' frames (' + args.test_video + ')', 
                (10, 35), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0))
    
    # draw subtitle
    for i in range(4):
        cv2.putText(output_frames, titles[i], (i * img_w + space * (i + 1), 65), 
                cv2.FONT_HERSHEY_PLAIN, 1, (100, 100, 100), thickness = 1)
    
    # plot contents for each testing frames
    for i in range(args.frame_num):
        
        # draw current frame number text
        # rectangle to cover previous text
        cv2.rectangle(output_frames, (out_w - 120, 0), (out_w - 1, 40), (255, 255, 255), cv2.FILLED)
        cv2.putText(output_frames, 'FRAME ' + str(i + 1), 
            (out_w - 120, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 200), thickness = 2)
        
        for j in range(4):
            # transform contents to be visualizable
            img = contents[j][0, :, i].transpose((1, 2, 0))
            img = cv2.resize(denormalize_buffer(img), (img_h, img_w))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            np.copyto(output_frames[text_h : out_h - btm_space, 
                                    j * img_w + space * (j + 1) : 
                                        (j + 1) * img_w + space * (j + 1), :], img)
        # show results
        cv2.imshow('RESULT', output_frames)
        cv2.waitKey(800 if i != args.frame_num - 1 else 0)
    
    cv2.destroyAllWindows()