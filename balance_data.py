import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

# Balancing of data is not working well
# avoiding the use of this

dir_path = './Data/'
imgs, labels = np.load(dir_path + 'train_images.npy'), np.load(dir_path+ 'train_labels.npy')

BALANCE = False

### helper functions
def normalize(x):
    """
        INPUT
            x : an array of images
        OUTPUT  
            normalize by max value
    """
    return x / 255


# just to see how much imbalance is there in the data
print("\nData Imbalance stats:-")
df = pd.DataFrame(zip(imgs,labels))
print(Counter(df[1].apply(str)))


if not BALANCE:
    print("Balancing is disabled. To enable set BALANCE to True.")
    exit()

print("Balancing Data...")
train_data = df.values
lefts = []
rights = []
forwards = []


for data in train_data:
    img = data[0]
    label = list(data[1])

    if label == [1, 0, 0]:
        lefts.append([img, label ])
    elif label == [0, 1, 0]:
        forwards.append([img, label ])
    elif label == [0, 0, 1]:
        rights.append([img, label])
    else:
        print('Errorrr')

forwards = forwards[:len(lefts)][:len(rights)]
lefts = lefts[:len(forwards)]
rights = rights[:len(rights)]

final_data = forwards + lefts + rights

print("Shuffling...")
shuffle(final_data) # shuffle it
print("\nBalanced data stats:-")


print("Length of train data: ", len(final_data))

shuffled_data = pd.DataFrame(final_data)

#shuffled_data = df.sample(frac=1).reset_index()
train_imgs, train_labels = shuffled_data[0].values, shuffled_data[1].values
train_imgs = normalize(train_imgs)


# saving this
print("Saving the balanced data:-")
np.save('train_images_balanced.npy', train_imgs)
np.save('train_labels_balanced.npy', train_labels)
print("Saved with name `train_images_balanced.npy`.")
print("Saved with name `train_labels_balanced.npy`.")


# for img, label in zip(imgs,labels):

#     cv2.imshow('test', img)
#     print(label)
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
