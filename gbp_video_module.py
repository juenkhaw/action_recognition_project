# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:21:08 2019

@author: Juen
"""

import numpy as np
import cv2
import glob

def spatial_center_crop(buffer_size, clip_size):
    """
    Crops a center patch from frames
    
    Inputs:
        buffer_size : size of the original scaled frames
        clip_size : size of the output clip frames
        
    Returns:
        start_h, start_w : (x, y) point of the top-left corner of the patch
        end_h, end_w : (x, y) point of the bottom-right corner of the patch
    """
    
    # expected parameters to be a tuple of height and width
    assert(len(buffer_size) == 2 and len(clip_size) == 2)
    
    # obtain top-left and bottom right coordinate of the center patch
    start_h = (buffer_size[0] - clip_size[0]) // 2
    end_h = start_h + clip_size[0]
    start_w = (buffer_size[1] - clip_size[1]) // 2
    end_w = start_w + clip_size[1]
    
    return (start_h, end_h), (start_w, end_w)

def temporal_center_crop(buffer_len, clip_len):
    """
    Crops the center part of a video over its temporal dimension
    
    Inputs:
        buffer_len : original video framecount (depth)
        clip_len : expected output clip framecount
        
    Returns:
        start_index, end_index : starting and ending indices of the clip
    """
    
    # select the center portion of a video
    start_index = (buffer_len - clip_len) // 2 - 1
    end_index = start_index + clip_len
    
    return start_index, end_index

def normalize_buffer(buffer):
    """
    Normalizes values of input frames buffer
    
    Inputs:
        buffer : np array of original frames
        
    Returns:
        buffer : np array of normalized frames
    """
    
    # normalize the pixel intensity with range [-1, 1]
    buffer = (buffer - 128) / 128
    return buffer

def denormalize_buffer(buffer, option = 1):
    """
    Restores the buffer to its original intensity values
    
    Inputs:
        buffer : np array of normalized frames
        
    Returns:
        buffer : np array of denormalized frames
    """
    
    # denormalize the intensity with range [0, 255]
    if option == 0:
        buffer = buffer * 128 + 128
    else:
        buffer = (buffer - buffer.min()) / (buffer.max() - buffer.min()) * 255
    
    return buffer.astype(np.uint8)

def load_clips(frames_path, scale_h, scale_w, output_h, output_w, output_len, frame_chn):
    """
    Reads original video RGB frames/flows into memory and preprocesses to form training/testing input volume
    
    Inputs:
        frame_path : list of directory where the original frames/flows located
        scale_h, scale_w : spatial size to be scaled into before cropping
        output_h, output_w : spatail size to be cropped from the scaled frames/flows
        output_len : temporal depth of the output clips
        frame_chn : indicating the modality of data to be loaded
        
    Returns:
        buffer : np array of preprocessed input volume
    """
    
    # read path content and sample frame and sort the frames based on its temporal sequence
    path_contents = []
    for i in range(len(frames_path)):
        path_content = glob.glob(frames_path[i] + '/*.jpg')
        path_content.sort()
        path_contents.append(path_content)
    
    # retrieve frame properties
    if frame_chn == 3:
        assert(len(path_contents) == 1)
    if (frame_chn == 1):
        assert(len(path_contents[0]) == len(path_contents[1]))
        assert(len(path_contents) == 2)
        
    frame_count = int(len(path_contents[0]))
    #frame_chn = 3
    
    # retrieves spatiotemporal indices for cropping
    t_index = temporal_center_crop(frame_count, output_len)
    s_index = spatial_center_crop((scale_h, scale_w), (output_h, output_w))
    
    # create a buffer with size of 
    # video [clip_count, clip_len, height, width, channel]
    #buffer = np.empty((clips_per_video, output_len, output_h, output_w, frame_chn), np.float32)
    buffer = np.empty((1, output_len * (1 if frame_chn == 3 else 2), 
                       output_h, output_w, frame_chn), np.float32)
    
    # loading cropped video frames into the numpy array
        
    count = t_index[0]
    
    while count < t_index[1]:
        
        for i in range(len(path_contents)):
        
            # read the original RGB frames
            buffer_frame = cv2.imread(path_contents[i][count], 
                                      cv2.IMREAD_COLOR if frame_chn == 3 else cv2.IMREAD_GRAYSCALE)
                
            if buffer_frame is not None:

                # resize the frame
                buffer_frame = cv2.resize(buffer_frame, (scale_w, scale_h))
                
                # revert arangement of colour channels
                if frame_chn == 3:
                    buffer_frame = cv2.cvtColor(buffer_frame, cv2.COLOR_BGR2RGB)
                else:
                    buffer_frame = np.expand_dims(buffer_frame, axis = 2)
                
                #print(buffer_frame.shape)
                #print(s_index)
                    
                # apply the random cropping
                buffer_frame = buffer_frame[s_index[0][0] : s_index[0][1], 
                             s_index[1][0] : s_index[1][1], :]
            
                # copy to the video buffer
                np.copyto(buffer[0, (count - t_index[0]) * (1 if frame_chn == 3 else 2) + i,
                                 :, :, :], buffer_frame)
                    
            else:
                print("Error loading", path_content[count])
        
        count += 1
    
    # normalize the video buffer
    buffer = normalize_buffer(buffer)
    
    # convert array format to cope with Pytorch
    # [chnl, depth, h, w] for clips
    # [clip, chnl, depth, h, w] for video
    buffer = buffer.transpose((0, 4, 1, 2, 3))
        
    return buffer