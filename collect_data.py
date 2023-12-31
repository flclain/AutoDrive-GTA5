# create_training_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os
import sys
import random


## arguments reference
# disable do nothing
# do nothing is randomly replaced by some error input

# this is a form of error introduction
disable_no_input = True if sys.argv[1] == '1' else False

if disable_no_input:
    print("WARNING: `No Input` keybind is disabled for data collection.\n")

def keys_to_output(keys):
   """
    Convert our key inputs to one-hot array
    
    WA -         [0 1 0 0 0]
    WD -         [0 0 1 0 0]
    S -          [0 0 0 1 0]
    W -          [1 0 0 0 0]
    do nothing - [0 0 0 0 1]
   """
   output = [0,0,0,0,0]

   if 'A' in keys and 'W' in keys:
       output[1] = 1
   elif 'D' in keys and 'W' in keys:
       output[2] = 1
   elif 'S' in keys:
       output[3] = 1
   elif 'W' in keys:
       output[0] = 1
   else:
       if disable_no_input: # if no input option is disabled
           choice = random.random()
           if choice >= 0 and choice < 0.3:
                output[1] = 1
           elif choice >= 0.3 and choice < 0.6:
                output[2] = 1 
           elif choice >= 0.6 and choice <0.98:
                output[3] = 1
           else:
                output[0] = 1
       else: # if not then no input will be considered
        output[4] = 1

   return output

dir_path = './Data/'
img_file_name = 'train_images.npy'
labels_file_name = 'train_labels.npy'


# trying to load the data file
if os.path.isfile(dir_path+img_file_name):
    print('Image File exists, loading previous data!')
    imgs = list(np.load(dir_path+img_file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    imgs = []

if os.path.isfile(dir_path+labels_file_name):
    print('Image File exists, loading previous data!')
    labels = list(np.load(dir_path+labels_file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh!')
    labels = []

def main():

    # countdown timer
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)


    paused = False
    last_time = time.time()
    while(True):
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            keys = key_check()
            output = keys_to_output(keys)
            imgs.append(screen)
            labels.append(output)
            
            print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            if len(imgs) % 500 == 0 and len(imgs) == len(labels):
                print(len(imgs))
                np.save(dir_path + img_file_name,imgs)
                np.save(dir_path + labels_file_name,labels)
        
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused data collection. Press T again to pause.')
                time.sleep(1)
            else:
                print('Pausing data collection. Press T to start again.')
                paused = True
                time.sleep(1)

main()