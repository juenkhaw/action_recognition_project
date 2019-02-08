# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:13:01 2019

@author: Juen
"""

import os
import numpy as np
import cv2

def temporal_crop(buffer_len, clip_len):
    """
    This function accepts original video framecount (depth) and expected output length
    Returns start and end indices of frame index to be included in the training process
    """
    
    # randomly select time index for temporal jittering
    start_index = np.random.randint(buffer_len - clip_len)
    end_index = start_index + clip_len
    
    return start_index, end_index

def spatial_crop(buffer_size, clip_size):
    """
    This function accepts original video spatial size (h and w) and expected output spatial size
    Returns starting point (x, y) on the frames to be cropped from
    """
    
    # expected parameters to be a tuple of height and width
    assert(len(buffer_size) == 2 and len(clip_size) == 2)
    
    # randomly select start indices in spatial dimension to crop the video
    start_h = np.random.randint(buffer_size[0] - clip_size[0])
    end_h = start_h + clip_size[0]
    start_w = np.random.randint(buffer_size[1] - clip_size[1])
    end_w = start_w + clip_size[1]
    
    return (start_h, end_h), (start_w, end_w)

def normalize_video(buffer):
    """
    This function accepts np array of clip pixels
    Returns normalized np array of clip pixel values
    """
    
    #normalize the pixel values to be in between -1 and 1
    buffer = (buffer - 128) / 128
    return buffer

def load_video(frames_path, scale_h, scale_w, output_h, output_w, output_len):
    """
    This function accepts:
        frame_path : Directory where the frame images located
        scale_h, scale_w : Target spatial size to which frames should resize
        output_h, output_w : Target spatial size of training clip
        output_len : Target depth of training clip
        
    Returns preprocessed and normalized np array of a single training clip
    """
    
    # read path content and sample frame
    path_content = os.listdir(frames_path)
    #sample_frame = cv2.imread(frames_path + '/' + path_content[0], cv2.IMREAD_COLOR)
    
    # retrieve frame properties
    frame_count = int(len(path_content))
    
    # retrieve indices for random cropping    
    t_index = temporal_crop(frame_count, output_len)
    s_index = spatial_crop((scale_h, scale_w), (output_h, output_w))
    
    # create a buffer with size of 
    buffer = np.empty((output_len, output_h, output_w, 3), np.float32)
    
    # loading cropped video frames into the numpy array
    count = t_index[0]
    while count < t_index[1]:
        buffer_frame = cv2.imread(frames_path + '/' + path_content[count], cv2.IMREAD_COLOR)
        buffer_frame = cv2.cvtColor(buffer_frame, cv2.COLOR_BGR2RGB)
        
        # resize the frame
        buffer_frame = cv2.resize(buffer_frame, (scale_w, scale_h))
        
        # apply the data augmentation
        buffer_frame = buffer_frame[s_index[0][0] : s_index[0][1], 
                                    s_index[1][0] : s_index[1][1], :]
        
        #cv2.imshow('buffer', buffer_frame)
        #cv2.waitKey(0)
        
        # copy to the video buffer
        np.copyto(buffer[count - t_index[0], :, :, :], buffer_frame)
        
        count += 1
    
    # normalize the video buffer
    buffer = normalize_video(buffer)
        
    return buffer
    
if __name__ == '__main__':
    video_path = r'..\dataset\UCF-101\ucf101_jpegs_256\jpegs_256\v_ApplyEyeMakeup_g01_c01'
    buffer = load_video(video_path, 128, 171, 112, 112, 16)
    cv2.imshow('buffer', buffer[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
