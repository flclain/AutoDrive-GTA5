
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from inception_v3 import InceptionV3
import time
import tqdm
import random

# hyper parameters
batch_size = 32
LR = 0.001
num_epochs = 25
WIDTH = 160
HEIGHT = 120
n_classes = 5


# path and file names
dir_path = './Data/'
img_file_name = 'train_images.npy'
labels_file_name = 'train_labels.npy'

print("Loading our dataset...")
# looading our model data
train_imgs = np.load(dir_path + img_file_name, allow_pickle=True)
train_labels = np.load(dir_path + labels_file_name, allow_pickle=True)

print("Preprocessing the data...")
# a little changes
X = np.stack(train_imgs, axis=0)
y = np.stack(train_labels, axis=0)

X = X.astype(float)/255 # normalization for training

# custom dataset definition
class DatasetGTA5(Dataset):
    def __init__(self, numpy_data, labels):
        self.data = numpy_data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return data, label



# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# custom dataset transformations
# transform = transforms.Compose([
#     transforms.Resize((WIDTH, HEIGHT)),
#     transforms.ToTensor(),
# ])


# initialize pytorch custom dataset
dataset = DatasetGTA5(X, y)

# splitting dataset into train and test dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# dataloaders for training and testing
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print("Initializing our Inception3 model...")
# our inception 3 model
model = InceptionV3(num_classes=n_classes)
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# training loop
print("\nTraining the model:-")
total_steps = len(train_loader)

start = time.time()
for epoch in range(num_epochs):
    running_loss = 0.
    correct = 0
    total = 0

    print("\nEpoch #{}:-\n".format(epoch+1))
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader)):
        # move tensors to devices
        images = images.to(device)
        labels = labels.to(device).to(torch.float32)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # calculate running loss
        running_loss += loss.item()

        # calculating running accuracy
        _, pred_classes = torch.max(outputs, dim=1)
        _, true_classes = torch.max(labels, dim=1)
        
        correct += (pred_classes == true_classes).sum().item()
        total += labels.size(0)

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    # Print epoch statistics
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss:\
           {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")
last = time.time()

print("\nTraining time elapsed: %.2fs"%(last-start))

# evaluate the model
# Set the model to evaluation mode
model.eval()

# Initialize variables for accuracy calculation
correct = 0
total = 0

# Disable gradient calculation
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device).to(torch.float32)

        # Forward pass
        outputs = model(images)

        # Calculate accuracy
        _, predicted_classes = torch.max(outputs, dim=1)
        _, true_classes = torch.max(labels, dim=1)

        correct += (predicted_classes == true_classes).sum().item()
        total += labels.size(0)

# Calculate accuracy
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


# Saving model state
print("\nSaving the model weights...")
torch.save(model.state_dict(), './ModelSaves/inceptv3_model.pth')
print("Model saved in ModelSaves folder.")