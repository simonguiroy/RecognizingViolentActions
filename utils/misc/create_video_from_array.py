from PIL import Image
from array2gif import write_gif
import gym
import numpy as np
import cv2

# This script runs a gym environment and creates a video from the numpy array.
# Also a good example of how to create a video from a numpy array, using OpenCV

env = gym.make('CartPole-v0')
env.reset()
frame = env.render(mode='rgb_array')
print ("Environment render sucessfully")


N_steps = 50
#frames = np.zeros(frame.shape[0], frame.shape[1], frame.shape[2], N_steps)
#frames[:,:,:,0] = frame
frames = []

# frame dimensions
# 400 x 600 x 3

#frames = np.ndarray(shape=(frame.shape[0], frame.shape[1], frame.shape[2], N_steps), dtype=int16)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('simple_output.avi',fourcc, 20.0, (frame.shape[1], frame.shape[0]))
#out = cv2.VideoWriter('simple_output.avi', -1, 20.0, (frame.shape[1], frame.shape[0]))

#writer = cvCreateVideoWriter('simple_output.avi', -1, 20.0, (frame.shape[0], frame.shape[1]), is_color=1)

i = 1
for _ in range(N_steps - 1):
    frame = env.render(mode='rgb_array')

    #frame2 = np.reshape( frame, (frame.shape[2], frame.shape[0], frame.shape[1]) ) 
    #frames[:,:,:, i] = frame2
    #frames.append(frame2)
    #frames.append(frame[:,:,0])
    out.write(frame)
    #cvWriteFrame(out, frame)
    env.step(env.action_space.sample()) # take a random action
    i+=1

env.close()
print ("Environment closed")
out.release()
#im = Image.fromarray(frame)
#im.save("frame.jpeg")
#write_gif(frames, 'rgb.gif', fps=24)
