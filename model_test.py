import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import W, A, S, D, PressKey, ReleaseKey
from getkeys import key_check
import os
import torch
import random

from inception_v3 import InceptionV3

# device - gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
LR = 0.001
WIDTH = 160
HEIGHT = 120
n_classes = 5
t_time = 0.05


# time threshold
time_t = 0.09

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def fleft():
    PressKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(A)


def fright():
    PressKey(W)
    PressKey(D)
    ReleaseKey(S)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(D)

def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(W)


# loading our model weights
model = InceptionV3(n_classes)
model = model.to(device)
model.load_state_dict(torch.load('./ModelSaves/inceptv3_model.pth'))



def main():
    # countdown timer
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    while True:
        paused = False
        #last_time = time.time()
        while(True):
            if not paused:
                # 800x600 windowed mode
                screen = grab_screen(region=(0,40,800,640))
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
                screen = cv2.resize(screen, (160,120))
                # print('Frame took {} seconds'.format(time.time()-last_time))
                # last_time = time.time()
                
                screen = screen.astype(float)
                image = torch.tensor(screen).unsqueeze(0)
                image = image.to(device)
                pred = model(image)
                pred = pred.argmax().item()
                #torch.nn.functional.softmax(pred, dim=1)


                if pred == 0:
                    straight()
                    #print("Forward")
                elif pred == 1:
                    fleft()
                    #print("Left")
                elif pred == 2:
                    fright() 
                    #print("Right")
                elif pred == 3:
                    reverse()
                      # barely used as of now
                elif pred == 4:
                    # do nothing
                    time.sleep(1)
                    
                
                    
            

            keys = key_check()
            if 'T' in keys:
                if paused:
                    paused = False
                    print('Unpaused model testing. Press T again to pause.')
                    time.sleep(1)
                else:
                    print('Pausing model testing. Press T to start again.')
                    paused = True
                    ReleaseKey(A)
                    ReleaseKey(W)
                    ReleaseKey(D)
                    ReleaseKey(S)
                    time.sleep(1)

main()