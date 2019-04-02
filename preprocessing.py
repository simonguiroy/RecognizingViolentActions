import numpy as np
import cv2
import time
import sys


def preprocess_video(file_path, stream, max_frames_per_clip=-1):
    """
    :param file_path: Path to video file
    :param stream: Wether to perform RGB or Optical Flow preprocessing
    :param max_frames_per_clip: Maximujm number of frames to process per clip. If -1, process entire clip.
    :return: A sequence of processed frames, as a Numpy array.
    """
    cap = cv2.VideoCapture(file_path)
    time.sleep(0.01)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Resized frame width and height so that the smallest dimension is 256 pixels
    if frame_width < frame_height:
        frame_width_resize = 256
        frame_height_resize = np.round(frame_height * np.true_divide(256.0, frame_width))
    else:
        frame_height_resize = 256
        frame_width_resize = np.round(frame_width * np.true_divide(256.0, frame_height))
    # Pixel positions for 224x224 central crop
    crop_side = 224
    top = np.int(np.round((frame_height_resize-crop_side)/2))
    left = np.int(np.round((frame_width_resize-crop_side)/2))

    # Setting number of frames to process from video clip
    if max_frames_per_clip == -1 or max_frames_per_clip > cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        frame_count = max_frames_per_clip

    num_channels = 3 if stream == 'rgb' else 2
    # Numpy array for the sequence of processed frames
    buf = np.empty((1, int(frame_count), crop_side, crop_side, num_channels),
                   np.dtype('float32'))

    if stream == 'flow':
        _, previous_frame = cap.read()
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        previous_frame = cv2.resize(previous_frame, (int(frame_width_resize), int(frame_height_resize)),
                                  interpolation=cv2.INTER_LINEAR)
        dtvl1 = cv2.createOptFlow_DualTVL1()

    # Iterator through the buffer
    fc = 0
    ret = True
    while cap.get(cv2.CAP_PROP_POS_FRAMES) < frame_count and ret is True:
        ret, frame = cap.read()
        if ret:
            if stream == 'flow':
                # Converting to grayscale image
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resizing frame with bilinear interpolation
            frame_resized = cv2.resize(frame, (int(frame_width_resize), int(frame_height_resize)),
                                       interpolation=cv2.INTER_LINEAR)

            # Printing progress bar
            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-100s] %d%%" % ('=' * int(100 * (fc + 1) / (frame_count - 1)), int(100 * (fc + 1) / (frame_count - 1))))
            sys.stdout.flush()

            if stream == 'flow':
                # Computing optical flow frame
                frame = dtvl1.calc(previous_frame, frame_resized, None)
                # Clipping between -20 and 20, then rescaling between -1 and 1.
                frame = np.divide(np.clip(frame, -20.0, 20.0), 20.0)
                previous_frame = frame_resized
            else:
                # Pixel values are rescaled between -1 and 1
                frame = np.true_divide(frame_resized, 255.0) * 2.0 - 1.0

            # Adding processed frame to the buffer.
            buf[0, fc] = np.float32(frame[top:top+crop_side, left:left+crop_side, :])
            fc += 1

    # Releasing video capture
    cap.release()
    # Resetting stdout at beginning of line
    sys.stdout.write('\n')
    sys.stdout.flush()

    return buf[:, 0:fc, :, :, :]
