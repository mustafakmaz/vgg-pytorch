import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Batch size
batch_size = 100

# Dataset
train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.CIFAR10(root="\data", train=False, transform=transforms.ToTensor(), download=True)

# Data loader
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load model and enable evaluation mode
model_name = input("Please enter the model name you wanted to load: ")
model = torch.load(model_name)
model.eval()

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim = 1)
    return train_dataset.classes[preds[0].item()]

correct = 0
wrong = 0

for i in range(0, 10):
    random_integer = np.random.randint(0,len(test_dataset))
    img, label = test_dataset[random_integer]
    plt.imshow(img.permute(1, 2, 0))
    result = predict_image(img, model)
    if train_dataset.classes[label] == result:
        correct = correct + 1
    else:
        wrong = wrong + 1

    print("Index:", random_integer, "Label:", train_dataset.classes[label], ',Predicted:', predict_image(img, model))
    plt.show()

print("Correct predictions:", correct, "Wrong predictions:", wrong)