# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:13:01 2019

@author: Juen
"""
import numpy as np
import cv2
import glob

def temporal_crop(buffer_len, clip_len):
    """
    Randomly crops the video over temporal dimension
    
    Inputs:
        buffer_len : original video framecount (depth)
        clip_len : expected output clip framecount
        
    Returns:
        start_index, end_index : starting and ending indices of the clip
    """
    
    # randomly select time index for temporal jittering
    start_index = np.random.randint(buffer_len - clip_len)
    end_index = start_index + clip_len
    
    return start_index, end_index

def temporal_uniform_crop(buffer_len, clip_len, clips_per_video):
    """
    Uniformly crops the video into N clips over temporal dimension
    
    Inputs:
        buffer_len : original video framecount (depth)
        clip_len : expected output clip framecount
        clips_per_video : number of clips for each video sample
        
    Returns:
        indices : list of starting and ending indices of each clips
    """
    
    # compute the average spacing between each consecutive clips
    # could be negative if buffer_len < clip_len * clips_per_video
    spacing = (buffer_len - clip_len * clips_per_video) // (clips_per_video - 1)
    
    indices = [(0, clip_len)]
    for i in range(1, clips_per_video):
        start = indices[i - 1][1] + spacing
        end = start + clip_len
        indices.append((start, end))
    
    return indices

def spatial_crop(buffer_size, clip_size):
    """
    Randomly crops original video frames over spatial dimension
    
    Inputs:
        buffer_size : size of the original scaled frames
        clip_size : size of the output clip frames
        
    Returns:
        start_h, start_w : (x, y) point of the top-left corner of the patch
        end_h, end_w : (x, y) point of the bottom-right corner of the patch
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

def normalize_buffer(buffer):
    """
    Normalizes values of input frames buffer
    
    Inputs:
        buffer : np array of unnormalized frames
        
    Returns:
        buffer : np array of normalized frames
    """
    
    #normalize the pixel values to be in between -1 and 1
    buffer = (buffer - 128) / 128
    return buffer

def denormalize_buffer(buffer):
    """
    Denormalizes intensity values of input buffer frames
    
    Inputs:
        buffer : np array of normalized frames
        
    Returns:
        buffer : np array of unnormalized frames
    """
    
    buffer = ((buffer - buffer.min()) / (buffer.max() - buffer.min()) * 255).astype(np.uint8)
    
    return buffer

def flow_mean_sub(buffer):
    
    # buffer sum of shape (clip_num, flow_dim, h, w, chnl)
    sh = buffer.shape
    buffer_sum = np.zeros((sh[0], 2, sh[2], sh[3], 1))
    
    # summation of flow, 0 -> u, 1 -> v
    for l in range(sh[1]):
        buffer_sum[:, l % 2] += buffer[:, l]
    
    # average
    buffer_sum /= (sh[1] // 2)
    
    # substract the mean
    for l in range(sh[1]):
        buffer[:, l] -= buffer_sum[:, l % 2]
        
    return buffer

def load_clips(frames_path, modality, scale_h, scale_w, output_h, output_w, output_len, 
               mode = 'clip', mean_sub = True):
    """
    Reads original video frames/flows into memory and preprocesses them into training/testing input volume
    
    Inputs:
        frame_path : list of directories where the original frames/flows located
        modality : [rgb/flow] modality of video to be processed on
        scale_h, scale_w : spatial size to be scaled into before cropping
        output_h, output_w : spatail size to be cropped from the scaled frames/flows
        output_len : temporal depth of the output clips
        mode (optional) (testing only): [clip/video] 
        to export a single randomly cropped clip or series of uniformly cropped clips as per video
        clips_per_video (optional) (testing only) : clips count if exporting as series of clips
        
    Returns:
        buffer : np array of preprocessed input volume
    """
    
    # mode can only be as clip or video
    if modality == 'rgb':
        assert(len(frames_path) == 1)
    else:
        assert(len(frames_path) == 2)
    assert(mode in ['clip', 'video'])
    if mode == 'clip':
        clips_per_video = 1
    else:
        clips_per_video = 10
    
    # read path content and sample frame
    path_content = []
    for i in range(len(frames_path)):
        #path_content.append(os.listdir(frames_path[i]))
        path_content.append(glob.glob(frames_path[i] + '/*.jpg'))
        path_content[i].sort()
    #sample_frame = cv2.imread(frames_path + '/' + path_content[1], cv2.IMREAD_GRAYSCALE)
    
    # retrieve frame properties
    frame_count = int(len(path_content[0]))
    if modality == 'rgb':
        frame_chn = 3
    else:
        frame_chn = 1 #2
    
    if mode == 'clip':
        t_index = []
        # retrieve indices for random cropping
        t_index.append(temporal_crop(frame_count, output_len))
        s_index = spatial_crop((scale_h, scale_w), (output_h, output_w))
        #buffer = np.empty((output_len, output_h, output_w, frame_chn), np.float32)
    else:
        # retrieve indices for center cropping and temporal index for each clips
        t_index = temporal_uniform_crop(frame_count, output_len, clips_per_video)
        s_index = spatial_center_crop((scale_h, scale_w), (output_h, output_w))
    
    # create a buffer with size of 
    # video [clip_count, clip_len, height, width, channel]
    #buffer = np.empty((clips_per_video, output_len, output_h, output_w, frame_chn), np.float32)
    buffer = np.empty((clips_per_video, output_len if modality == 'rgb' else output_len * 2,
                       output_h, output_w, frame_chn), np.float32)
    
    # loading cropped video frames into the numpy array
    for c in range(clips_per_video):
        
        count = t_index[c][0]
        
        while count < t_index[c][1]:
            buffer_frame = []
            
            if frame_chn == 3:
                #buffer_frame.append(cv2.imread(frames_path[0] + '/' + path_content[0][count], cv2.IMREAD_COLOR))
                buffer_frame.append(cv2.imread(path_content[0][count]))
                buffer_frame[0] = cv2.cvtColor(buffer_frame[0], cv2.COLOR_BGR2RGB)
                
            else:
                #print(count)
                #print(path_content[0][count])
                #print(path_content[1][count])
                #buffer_frame.append(cv2.imread(frames_path[0] + '/' + path_content[0][count], cv2.IMREAD_GRAYSCALE))
                #buffer_frame.append(cv2.imread(frames_path[1] + '/' + path_content[1][count], cv2.IMREAD_GRAYSCALE))
                buffer_frame.append(cv2.imread(path_content[0][count], cv2.IMREAD_GRAYSCALE))
                buffer_frame.append(cv2.imread(path_content[1][count], cv2.IMREAD_GRAYSCALE))
            
            for i in range(len(buffer_frame)):
                
                if buffer_frame[i] is not None:
                
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
                        np.copyto(buffer[c, count - t_index[c][0], :, :, :], buffer_frame[i])
                    else:
                        #np.copyto(buffer[c, (count - t_index[c][0]), :, :, i], buffer_frame[i][:, :, 0])
                        np.copyto(buffer[c, (count - t_index[c][0]) * 2 + i, :, :, 0], buffer_frame[i][:, :, 0])
                        
                else:
                    print(path_content[i][count])
            
            count += 1
    
    # mean substraction
    if mean_sub and modality == 'flow':
        buffer = flow_mean_sub(buffer)
    
    # normalize the video buffer
    buffer = normalize_buffer(buffer)
    
    # convert array format to cope with Pytorch
    # [chnl, depth, h, w] for clips
    # [clip, chnl, depth, h, w] for video
    if mode == 'clip':
        buffer = buffer[0, :, :, :, :]
        buffer = buffer.transpose((3, 0, 1 ,2))
    else:
        buffer = buffer.transpose((0, 4, 1, 2, 3))
        
    return buffer
            
    
if __name__ == '__main__':
    #video_path = '../dataset/UCF-101/ucf101_jpegs_256/jpegs_256/v_BasketballDunk_g20_c06'
    video_path = '../dataset/UCF-101/ucf101_tvl1_flow/tvl1_flow/u/v_BasketballDunk_g20_c06'
    video_path2 = '../dataset/UCF-101/ucf101_tvl1_flow/tvl1_flow/v/v_BasketballDunk_g20_c06'
    buffer = load_clips([video_path, video_path2], 'flow', 128, 171, 112, 112, 8, mode = 'video', mean_sub=True)
    buffer = buffer.transpose((0, 2, 3, 4, 1))
#    buffer_chnl = buffer[:, :, :, :, 2]
#    buffer[:, :, :, :, 2] = buffer[:, :, :, :, 0]
#    buffer[:, :, :, :, 0] = buffer_chnl
#    cv2.imshow('buffer0', buffer[6, 0, :, :, :])
#    cv2.imshow('buffer1', buffer[6, 1, :, :, :])
#    cv2.waitKey(0)
    cv2.destroyAllWindows()
