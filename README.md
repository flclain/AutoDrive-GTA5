# AutoDrive GTA5
 Autonomous Driving simulation in GTA 5 game


### Requirements

- Your own instance of GTA 5 game
- Your python environment (anaconda preferably) and required libraries for a conda env are listed in `Requirements.txt` file. Link is [here](./Requirements/Requirements.txt)
- a GPU is preferable for training 
- You require installation of Native Trainer Mod in your GTA5. [Here](https://www.gta5-mods.com/tools/script-hook-v) is the link for it. The instructions are present on the website and also included in the download zip file.


### Training scenarios
- Currently, only considered training scenario where there is no traffic.
- Different routes were selected to generate data on these routes repeatedly.
- Below map shows highlighted routes on which data was generated:-
 ![Map](./Snapshots/Map_route_scenarios.png)

- Due to time constraints, we were able to generate just 1.5 GB worth of data. The model requires a huge amount of data to have enough values for `rare` inputs such as `reverse()` or take into consideration `rare cases` such as turns or sharp turns.

### Evaluation


### Comparative Analysis

- This is more qualitative.
- I will be comparing it with SentDex's implementation (update upto video #14).
- Our model is Google's Inception v3 whereas the related work makes use of AlexNet
- There are some qualitative points to be considered when comparing both models:-
    - Inception model is much more complex and is deeper. This complexity helps it capture more complex representations
    - Inception model was performing better than AlexNet and it has the capability to generalise better on the data
- However, the quality of training also depends heavily on fine tuning process for the game as well as the quality of the data that we collect.


### Areas of Improvement
- In continuation to last section
- Based on our evaluation, it seems the more data we have the more better it is. However, the larger data will require taking into consideration modifying the preprocessing part to accomodate for large scale image data.
- Increasing training set size will require more training time. For instance, training nearly 2GB of collected training data on just few route scenarios takes the model 4-5 minutes at minimum per epoch. The model is being trained for 25 epochs in average. So the total (minm) training time is at an estimate of 2 hours. 
