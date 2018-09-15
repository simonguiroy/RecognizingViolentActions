import numpy as np
import cv2
import time
import argparse
import os
import math



def preprocess_rgb(file_path, resize_frames=1.0):        
    cap = cv2.VideoCapture(file_path)
    time.sleep(2)
    
    # videos from UCF101 are all at 25 FPS, so no need to resample. For Kinetics, use ffmpeg to resample before.
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print ("RGB - ORIGINAL FRAME COUNT= " + str(frame_count))
    print ("RGB - ORIGINAL FRAME RATE= " + str(frame_rate))

    '''
    #change frame count if video to be resampled
    cap.set(cv2.CAP_PROP_FPS, frame_rate//2)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_count//2)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    time.sleep(2)

  
    print ("RGB - FRAME COUNT= " + str(frame_count))
    print ("RGB - FRAME RATE= " + str(frame_rate))
    '''


    
    #Resized frame width and height so that the smallest dimension is 256 pixels
    if frame_width < frame_height:
        frame_width_resize = 256
        frame_height_resize = np.round(frame_height * np.true_divide(256.0, frame_width))
    elif frame_height < frame_width:
        frame_height_resize = 256
        frame_width_resize = np.round(frame_width * np.true_divide(256.0, frame_height))

    frame_width_resize = math.floor(frame_width_resize * resize_frames)
    frame_height_resize = math.floor(frame_height_resize * resize_frames)
        

    #Numpy array for the processed frame sequence
    buf = np.empty((1, int(frame_count), int(frame_height_resize), int(frame_width_resize), 3), np.dtype('float32'))
    # iterator throught the buffer
    fc = 0

    ret = True
    #using 0-based index of the frame to be decoded/captured next
    while(cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count and ret == True):
        ret, frame = cap.read()            
        if ret == True:
            #Resizing frame with bilinear interpolation
            frame_resized = cv2.resize(frame, (int(frame_width_resize), int(frame_height_resize)), interpolation = cv2.INTER_LINEAR)
    
            #Pixel values are then rescaled between -1 and 1
            frame_rescaled = np.true_divide(frame_resized, 255.0)*2.0 - 1.0
            buf[0, fc] = np.float32(frame_rescaled)
            fc += 1                    
    cap.release()

    #central 224x224 pixel crop
    crop_side = 224
    crop_side = math.floor(crop_side * resize_frames)
    top = np.int(np.round((frame_height_resize-crop_side)/2))
    left = np.int(np.round((frame_width_resize-crop_side)/2))
    return buf[:,0:fc, top:top+crop_side, left:left+crop_side, :]
    
    
    

def preprocess_flow(file_path, resize_frames=1.0):
    cap = cv2.VideoCapture(file_path)
    time.sleep(2)
    
    # videos from UCF101 are all at 25 FPS, so no need to resample. For Kinetics, use ffmpeg to resample before.
    
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print ("FLOW - ORIGINAL FRAME COUNT= " + str(frame_count))
    print ("FLOW - ORIGINAL FRAME RATE= " + str(frame_rate))

    '''
    #change frame count if video to be resampled
    cap.set(cv2.CAP_PROP_FPS, frame_rate//2)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_count//2)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    time.sleep(2)

  
    print ("FLOW - FRAME COUNT= " + str(frame_count))
    print ("FLOW - FRAME RATE= " + str(frame_rate))
    '''
    
    
    #Resized frame width and height so that the smallest dimension is 256 pixels
    if frame_width < frame_height:
        frame_width_resize = 256
        frame_height_resize = np.round(frame_height * np.true_divide(256.0, frame_width))
    elif frame_height < frame_width:
        frame_height_resize = 256
        frame_width_resize = np.round(frame_width * np.true_divide(256.0, frame_height))
    
    frame_width_resize = math.floor(frame_width_resize * resize_frames)
    frame_height_resize = math.floor(frame_height_resize * resize_frames)
    
    
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prvs_resized = cv2.resize(prvs, (int(frame_width_resize), int(frame_height_resize)), interpolation = cv2.INTER_LINEAR) 
    
    frame1_resized = cv2.resize(frame1, (int(frame_width_resize), int(frame_height_resize)), interpolation = cv2.INTER_LINEAR) 
    hsv = np.zeros_like(frame1_resized)
    hsv[...,1] = 255
    
    #Numpy array for the processed frame sequence
    buf = np.empty((1, int(frame_count), int(frame_height_resize), int(frame_width_resize), 2), np.dtype('float32'))
    # iterator throught the buffer
    fc = 0
    
    ret = True
    #using 0-based index of the frame to be decoded/captured next
    while(cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count and ret == True):
        ret, frame2 = cap.read()
        
        if ret == True:
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            next_resized = cv2.resize(next, (int(frame_width_resize), int(frame_height_resize)), interpolation = cv2.INTER_LINEAR) 
    
            #Computing dense optical flow from the previous to the current frame. The flow has 2 channels. Cliping the values
            #to the range [-20, 20]. Rescaling to the range [-1, 1].
            flow = np.divide(np.clip(cv2.calcOpticalFlowFarneback(prvs_resized,next_resized, None, 0.5, 3, 15, 3, 5, 1.2, 0), -20.0, 20.0), 20.0)
            #for representation of the optical flow, but actually need to work with 2-channel (vertical and horizontal) flow fields.
            buf[0, fc] = np.float32(flow)
            fc += 1
            prvs_resized = next_resized
        
    cap.release()
    
    
    #central 224x224 pixel crop
    crop_side = 224
    crop_side = math.floor(crop_side * resize_frames)
    top = np.int(np.round((frame_height_resize-crop_side)/2))
    left = np.int(np.round((frame_width_resize-crop_side)/2))
    return buf[:,0:fc, top:top+crop_side, left:left+crop_side, :]
