# very common imports
from math import floor
import numpy as np
from numpy.dtypes import StringDType
import pandas as pd
import matplotlib.pyplot as plt

# common torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision.transforms import v2

# common sklearn imports 
from sklearn.preprocessing import LabelEncoder

from zootopia3 import MyCNN

import argparse

parser = argparse.ArgumentParser(description='Fit a convolutional neural network')
parser.add_argument(
    '--folder',
    type=str,
    required=True,
)
parser.add_argument(
    '--target_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--predictor_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--output_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--num_cnn_channels',
    type=int,
    default=2**5,
)
parser.add_argument(
    '--num_cnn_layers',
    type=int,
    default=2,
)
parser.add_argument(
    '--num_layers',
    type=int,
    default=1,
)
parser.add_argument(
    '--hidden_dim',
    type=int,
    default=2**5,
)
parser.add_argument(
    '--kernel_size',
    type=int,
    default=3,
)
parser.add_argument(
    '--stride',
    type=int,
    default=1,
)
parser.add_argument(
    '--padding',
    type=int,
    default=1,
)
parser.add_argument(
    '--pooling',
    type=int,
    default=2,
)
parser.add_argument(
    '--dropout',
    type=float,
    default=0.2,
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=2**7,
)
parser.add_argument(
    '--num_epochs',
    type=int,
    default=50,
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-4,
)
parser.add_argument(
    '--test_size',
    type=float,
    default=0.2,
)

args = parser.parse_args()

# specify files
predictor_file = args.predictor_file
target_file = args.target_file
folder = args.folder
output_file = args.output_file

# choose model architecture
num_cnn_channels = args.num_cnn_channels
num_cnn_layers = args.num_cnn_layers
num_layers = args.num_layers
hidden_dim = args.hidden_dim
kernel_size = args.kernel_size
stride = args.stride
padding = args.padding
pooling = args.pooling
dropout = args.dropout
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
test_size = args.test_size

# activation function is hard coded
linear_activation = nn.ReLU

# we will not have a gpu accessible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("CUDA is NOT available to Pytorch")
print(device)


# these parameters come from an imagenet study
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
transform_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=means, std=stds)
])


# Create some data with the `simulate-exercise.ipynb` file
# Start small, and `ls -lh` to check how big the data is
# Modify the file names below accordingly 

# load the data
target = np.loadtxt(f'{folder}/{target_file}', dtype=StringDType)
X = torch.from_numpy(
    np.load(f'{folder}/{predictor_file}')
    ).permute(0,3,1,2)

# label encode the targets
# so that the target is numeric
encoder = LabelEncoder()
encoder.fit(target)
y = encoder.transform(target)
y = torch.tensor(y)

# Define the dataset loader as train_loader and test_loader
# https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

train_size = 1 - test_size
full_data = TensorDataset(X, y)
train_data, test_data = random_split(full_data, [train_size, test_size])

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

# initialize the model
channels, width, height = X[0].shape
num_classes = len(y.unique())
model = MyCNN(
    num_classes, # number of classes
    height,
    width,
    channels,
    num_cnn_channels,
    num_cnn_layers,
    hidden_dim,
    num_layers,
    linear_activation,
    kernel_size,
    stride,
    padding,
    pooling,
    dropout,
)

model.to(device) # relevant if using gpus

# define the loss function
# what is a good loss function for multiclass classification
criterion = nn.CrossEntropyLoss()

# choose an optimizer
# https://docs.pytorch.org/docs/stable/optim.html

optimizer = optim.Adam(
    model.parameters(), # would not be available if not for super()
    lr=lr, # learning rate
)


# write the training loop

train_losses = []
test_losses = []

for epoch in range(num_epochs):

    model.train() # very important
    running_loss = 0.
    for inputs, labels in train_loader:

        optimizer.zero_grad()
        
        # put on the gpu or cpu device
        inputs = transform_pipeline(inputs)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward pass
        pred = model(inputs.float())

        # compute the loss
        loss = criterion(pred, labels)

        # backward pass
        # to change parameters
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # to track progress relative to training
    model.eval() # very important
    running_loss = 0.
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = transform_pipeline(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            pred = model(inputs.float())
            loss = criterion(pred, labels)
            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    test_losses.append(test_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss {test_loss:.4f}")

# save the model
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': {
        'num_classes': num_classes,
        'height': height,
        'width': width,
        'num_input_channels': channels,
        'num_cnn_channels': num_cnn_channels,
        'num_cnn_layers': num_cnn_layers,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'kernel_size': kernel_size,
        'stride': stride,
        'padding': padding,
        'pooling': pooling,
        'dropout': dropout
    },
    # custom items
    'train_losses': train_losses,
    'test_losses': test_losses,
}
torch.save(checkpoint, f'{folder}/{output_file}')