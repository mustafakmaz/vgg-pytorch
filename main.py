import torch 
import torchvision
from utils import BasicUtils, TrainTestUtils
import numpy as np
import matplotlib.pyplot as plt
from vgg import VGG11, VGG11LRN, VGG13, VGG16, VGG19

# Choosing device (NVIDIA CUDA GPU, Apple Silicon GPU, CPU)
device = BasicUtils().device_chooser()

# Loss lists
train_losses = []
test_losses = []

# Hyper parameters
num_epochs = 1
num_classes = 10
batch_size = 100
learning_rate = 0.001

# Dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Choosing model
model = VGG11(num_classes).to(device)

# Defining loss and optimizer functions
loss_fn = BasicUtils().loss_chooser("crossentropy")
optimizer = BasicUtils().optim_chooser("adamw", model, learning_rate)

# Train and test stage
for i in range(num_epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    train_losses.append(TrainTestUtils().train(train_loader, model, loss_fn, optimizer, i, num_epochs, batch_size))
    test_losses.append(TrainTestUtils().test(test_loader, model, loss_fn))

# Showing results (train and test losses)
plt.plot(train_losses,"g",label="train loss")
plt.plot(test_losses,"r",label="test loss")
plt.legend(loc="upper left")
plt.show()

# Saving model
model_name = input("Enter model name:")
BasicUtils().model_saver(model,model_name)