import numpy as np
import cv2
import time
import argparse
import os
import math
import sys
import torch
import torchvision
import PIL


def preprocess_rgb(file_path, max_frames_per_clip=-1):

    cap = cv2.VideoCapture(file_path)
    time.sleep(0.01)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if max_frames_per_clip == -1 or max_frames_per_clip > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        frame_count = max_frames_per_clip

    # Resized frame width and height so that the smallest dimension is 256 pixels
    if frame_width < frame_height:
        frame_width_resize = 256
        frame_height_resize = np.round(frame_height * np.true_divide(256.0, frame_width))
    elif frame_height < frame_width:
        frame_height_resize = 256
        frame_width_resize = np.round(frame_width * np.true_divide(256.0, frame_height))

    # Numpy array for the processed frame sequence
    buf = np.empty((1, int(frame_count), int(frame_height_resize), int(frame_width_resize), 3), np.dtype('float32'))
    #buf = torch.empty((1, int(frame_count), int(frame_height_resize), int(frame_width_resize), 3), dtype=torch.float, device=torch.device('cuda:0'))

    # Iterator through the buffer
    fc = 0
    ret = True

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count and ret is True:
        ret, frame = cap.read()
        if ret:
            # Resizing frame with bilinear interpolation
            frame_resized = cv2.resize(frame, (int(frame_width_resize), int(frame_height_resize)), interpolation = cv2.INTER_LINEAR)

            # Pixel values are then rescaled between -1 and 1
            frame_rescaled = np.true_divide(frame_resized, 255.0) * 2.0 - 1.0

            buf[0, fc] = np.float32(frame_rescaled)

            fc += 1

    cap.release()

    # Central crop of 224x224 pixels
    crop_side = 224
    top = np.int(np.round((frame_height_resize-crop_side)/2))
    left = np.int(np.round((frame_width_resize-crop_side)/2))

    return buf[:, 0:fc, top:top+crop_side, left:left+crop_side, :]


def preprocess_flow(file_path, max_frames_per_clip=-1):
    cap = cv2.VideoCapture(file_path)
    time.sleep(0.01)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if max_frames_per_clip == -1 or max_frames_per_clip > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        frame_count = max_frames_per_clip

    # IMPORTANT: Processing takes too long, therefore videos are only captured for 50 frames!
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 75:
        frame_count = 75
    else:
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Setting video to full length
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Resized frame width and height so that the smallest dimension is 256 pixels
    if frame_width < frame_height:
        frame_width_resize = 256
        frame_height_resize = np.round(frame_height * np.true_divide(256.0, frame_width))
    elif frame_height < frame_width:
        frame_height_resize = 256
        frame_width_resize = np.round(frame_width * np.true_divide(256.0, frame_height))

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prvs_resized = cv2.resize(prvs, (int(frame_width_resize), int(frame_height_resize)), interpolation=cv2.INTER_LINEAR)
    
    frame1_resized = cv2.resize(frame1, (int(frame_width_resize), int(frame_height_resize)), interpolation=cv2.INTER_LINEAR)
    hsv = np.zeros_like(frame1_resized)
    hsv[..., 1] = 255
    
    # Numpy array for the processed frame sequence
    buf = np.empty((1, int(frame_count), int(frame_height_resize), int(frame_width_resize), 2), np.dtype('float32'))
    # Iterator through the buffer
    fc = 0
    
    ret = True
    # DEBUGGING: here we force the sequence to have a low number of frames
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count and ret is True:
        ret, frame2 = cap.read()
        if ret:
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            next_resized = cv2.resize(next_frame, (int(frame_width_resize), int(frame_height_resize)), interpolation=cv2.INTER_LINEAR)
    
            # Computing dense optical flow using TV-L1 algorithm (Zach et al., 2007)
            # based on implementation of (Sanchez et al., 2011)
            dtvl1 = cv2.createOptFlow_DualTVL1()
            #print("computing flow: " + str(fc)) #debug

            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write(
                "[%-100s] %d%%" % ('=' * int(100 * (fc + 1) / (frame_count - 1)), int(100 * (fc + 1) / (frame_count - 1))))
            sys.stdout.flush()

            flow_frame = dtvl1.calc(prvs_resized, next_resized, None)

            # Clipping between -20 and 20, then rescaling between -1 and 1.
            flow = np.divide(np.clip(flow_frame, -20.0, 20.0), 20.0)

            # Adding frame of optical flow field to the buffer.
            buf[0, fc] = np.float32(flow)
            fc += 1
            prvs_resized = next_resized

    # Releasing video capture
    cap.release()
    # To reset output at beginning of line
    sys.stdout.write('\n')
    sys.stdout.flush()

    # central 224x224 pixel crop
    crop_side = 224
    top = np.int(np.round((frame_height_resize-crop_side)/2))
    left = np.int(np.round((frame_width_resize-crop_side)/2))

    #torch.save(torch.from_numpy(buf[:, 0:fc, top:top+crop_side, left:left+crop_side, :]), 'flow_clip.pkl') #debug
    #sys.exit() #debug
    return buf[:, 0:fc, top:top+crop_side, left:left+crop_side, :]
