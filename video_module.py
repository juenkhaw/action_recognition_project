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

def temporal_uniform_crop(buffer_len, clip_len, clips_per_video):
    """
    This function accepts original video framecount (depth), expected clip length, and clip count per video
    Returns list of start and end indices of frame index uniformly sampled across the video with length of clips_per_video
    """
    
    # compute the average spacing between each consecutive clips
    # could be negative if buffer_len < clip_len * clips_per_video
    spacing = int((buffer_len - clip_len * clips_per_video) / (clips_per_video - 1))
    
    indices = [(0, clip_len - 1)]
    for i in range(1, clips_per_video - 1):
        start = indices[i - 1][1] + spacing + 1
        end = start + clip_len - 1
        indices.append((start, end))
    indices.append((buffer_len - clip_len, buffer_len - 1))
    
    return indices

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

def spatial_center_crop(buffer_size, clip_size):
    """
    This function accepts original video spatial size (h and w) and expected output spatial size
    Returns starting point (x, y) of the center patch on the frames to be cropped from
    """
    
    # expected parameters to be a tuple of height and width
    assert(len(buffer_size) == 2 and len(clip_size) == 2)
    
    # obtain top-left and bottom right coordinate of the center patch
    start_h = (buffer_size[0] - clip_size[0]) // 2
    end_h = start_h + clip_size[0]
    start_w = (buffer_size[1] - clip_size[1]) // 2
    end_w = start_w + clip_size[1]
    
    return (start_h, end_h), (start_w, end_w)

def normalize_buffer(buffer):
    """
    This function accepts np array of clip pixels
    Returns normalized np array of clip pixel values
    """
    
    #normalize the pixel values to be in between -1 and 1
    buffer = (buffer - 128) / 128
    return buffer

def load_clips(frames_path, modality, scale_h, scale_w, output_h, output_w, output_len, 
               mode = 'clip', clips_per_video = 1):
    """
    This function accepts:
        frame_path : (List of) directory where the frame images located
        modality: [rgb / flow], to be passed from VideoDataset class argument
        scale_h, scale_w : Target spatial size to which frames should resize
        output_h, output_w : Target spatial size of training clip
        output_len : Target depth of training clip
        
    Returns preprocessed and normalized np array of a single training clip
    """
    
    # mode can only be as clip or video
    assert(mode in ['clip', 'video'])
    
    # read path content and sample frame
    path_content = []
    for i in range(len(frames_path)):
        path_content.append(os.listdir(frames_path[i]))
    #sample_frame = cv2.imread(frames_path + '/' + path_content[1], cv2.IMREAD_GRAYSCALE)
    
    # retrieve frame properties
    frame_count = int(len(path_content[0]))
    if modality == 'rgb':
        frame_chn = 3
    else:
        frame_chn = 2
    
    if mode == 'clip':
        # retrieve indices for random cropping
        t_index = temporal_crop(frame_count, output_len)
        s_index = spatial_crop((scale_h, scale_w), (output_h, output_w))
        # create a buffer with size of 
        # clips [clip_len, height, width, channel] or 
        buffer = np.empty((output_len, output_h, output_w, frame_chn), np.float32)
    else:
        # retrieve indices for center cropping and temporal index for each clips
        t_index = temporal_uniform_crop(frame_count, output_len, clips_per_video)
        s_index = spatial_center_crop((scale_h, scale_w), (output_h, output_w))
        # create a buffer with size of 
        # video [clip_count, clip_len, height, width, channel]
        buffer = np.empty((clips_per_video, output_len, output_h, output_w, frame_chn), np.float32)
    
    # loading cropped video frames into the numpy array
    for c in range(clips_per_video):
        
        count = t_index[0] if mode == 'clips' else t_index[c][0]
        
        while count < t_index[1] if mode == 'clips' else t_index[c][1]:
            buffer_frame = []
            
            if frame_chn == 3:
                buffer_frame.append(cv2.imread(frames_path[0] + '/' + path_content[0][count], cv2.IMREAD_COLOR))
                buffer_frame[0] = cv2.cvtColor(buffer_frame[0], cv2.COLOR_BGR2RGB)
                
            else:
                buffer_frame.append(cv2.imread(frames_path[0] + '/' + path_content[0][count], cv2.IMREAD_GRAYSCALE))
                buffer_frame.append(cv2.imread(frames_path[1] + '/' + path_content[1][count], cv2.IMREAD_GRAYSCALE))
                
            for i in range(len(buffer_frame)):
                
                # resize the frame
                buffer_frame[i] = cv2.resize(buffer_frame[i], (scale_w, scale_h))
                
                # add channel dimension if frame is flow
                if modality == 'flow':
                    buffer_frame[i] = buffer_frame[i][:, :, np.newaxis]
                    
                # apply the random cropping
                buffer_frame[i] = buffer_frame[i][s_index[0][0] : s_index[0][1], 
                             s_index[1][0] : s_index[1][1], :]
            
                # copy to the video buffer
                if modality == 'rgb':
                    np.copyto(buffer[count - t_index[0], :, :, :] if mode == 'clips' else 
                              buffer[c, count - t_index[c][0], :, :, :], buffer_frame[i])
                else:
                    np.copyto(buffer[count - t_index[0], :, :, i] if mode == 'clips' else
                              buffer[c, count - t_index[c][0], :, :, i], buffer_frame[i][:, :, 0])
            
            count += 1
    
    # normalize the video buffer
    buffer = normalize_buffer(buffer)
    
    # convert array format to [chnl, depth, h, w] to cope with Pytorch
    buffer = buffer.transpose((3, 0, 1 ,2))
        
    return buffer
            
    
if __name__ == '__main__':
    #video_path = r'..\dataset\UCF-101\ucf101_jpegs_256\jpegs_256\v_ApplyEyeMakeup_g01_c01'
    video_path = r'..\dataset\UCF-101\ucf101_tvl1_flow\tvl1_flow\u\v_ApplyEyeMakeup_g01_c01'
    video_path2 = r'..\dataset\UCF-101\ucf101_tvl1_flow\tvl1_flow\v\v_ApplyEyeMakeup_g01_c01'
    buffer = load_video([video_path, video_path2], 'flow', 128, 171, 112, 112, 16)
    cv2.destroyAllWindows()
