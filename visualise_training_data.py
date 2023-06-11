### show training data files with output in terminal for reference
import numpy as np
import cv2



dir_path = './Data/'
imgs, labels = np.load(dir_path + 'train_images.npy'), np.load(dir_path+ 'train_labels.npy')

"""
LABEL REFERENCE

W  [1 0 0 0 0]
WA [0 1 0 0 0]
WD [0 0 1 0 0]
S  [0 0 0 1 0]
N/A[0 0 0 0 1] (no input scenario)

no input scenario is sometimes replaced
with incorrect label randomly assigned within the set: 
labels \ {[0 0 0 0 1]}

"""


for image, label in zip(imgs, labels):
    cv2.imshow('window', image)
    print(">>> ",label)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break