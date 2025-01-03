# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:41:30 2025

@author: Yunus
"""
# pip install torch 
# pip install torchvision
# pip install unet 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import random
import matplotlib.pyplot as plt

# Define a simple UNet model (as an example)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layer
        # After 3 pooling layers (each halving the spatial size), we get a 3x3 feature map.
        self.fc = nn.Linear(256 * 3 * 3, 28 * 28)  # Adjusted to match flattened size
        
        # Sigmoid activation for output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t):
        # Pass through conv layers
        x = self.pool(torch.relu(self.conv1(x)))  # (batch_size, 64, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))  # (batch_size, 128, 7, 7)
        x = self.pool(torch.relu(self.conv3(x)))  # (batch_size, 256, 3, 3)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256*3*3)
        
        # Fully connected layer to get output
        x = self.fc(x)  # (batch_size, 28*28)
        
        # Reshape back to image shape (batch_size, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)
        
        return self.sigmoid(x)


# Add Gaussian Noise to the images
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


# Random Erasing for image augmentation
class RandomErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.random() < self.p:
            area = img.size()[1] * img.size()[2]  # Image height * width
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h = int((erase_area * aspect_ratio) ** 0.5)
            w = int((erase_area / aspect_ratio) ** 0.5)

            # Ensure that the erase area is valid (not too large for small images like MNIST)
            h = min(h, img.size(1))
            w = min(w, img.size(2))

            top = random.randint(0, img.size(1) - h)
            left = random.randint(0, img.size(2) - w)

            img[:, top:top+h, left:left+w] = torch.randn_like(img[:, top:top+h, left:left+w]) * 0.5
        return img


# Augmentation Pipelines
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

augmentation_advanced = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),  # This already normalizes size to 28x28
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.1),
    RandomErasing(p=0.3, scale=(0.02, 0.2)),  # Reduced scale for MNIST images
])

# Load MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=augmentation, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train_advanced_dataset = datasets.MNIST(root='./data', train=True, transform=augmentation_advanced, download=True)
train_advanced_dataloader = DataLoader(train_advanced_dataset, batch_size=64, shuffle=True)

# Model Training Function
def train_model(dataloader, model, optimizer, loss_fn, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images, t=torch.randint(0, 100, (images.size(0),), device=device))
            loss = loss_fn(outputs, images)
            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Model Evaluation Function
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images, t=torch.randint(0, 100, (images.size(0),), device=device))
            loss = loss_fn(outputs, images)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss / len(dataloader):.4f}")

# Visualization of Augmented Images
def visualize_augmentations(dataloader, num_images=8):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Create a grid of images
    images_grid = torchvision.utils.make_grid(images[:num_images], nrow=4, normalize=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(images_grid.permute(1, 2, 0))  # Convert CHW to HWC for display
    plt.title("Augmented Images")
    plt.axis("off")
    plt.show()

# Initialize Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Visualize Augmented Images
visualize_augmentations(train_dataloader)

# Train Model on Unaugmented Data
print("Training on Unaugmented Data:")
train_model(train_dataloader, model, optimizer, loss_fn)

# Train Model on Advanced Augmented Data
print("\nTraining on Advanced Augmented Data:")
train_model(train_advanced_dataloader, model, optimizer, loss_fn)

# Evaluate Models on Test Set
test_dataset = datasets.MNIST(root='./data', train=False, transform=augmentation, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("\nEvaluating on Unaugmented Data:")
evaluate_model(test_dataloader, model, loss_fn)

# Evaluate on Advanced Augmented Data
test_advanced_dataset = datasets.MNIST(root='./data', train=False, transform=augmentation_advanced, download=True)
test_advanced_dataloader = DataLoader(test_advanced_dataset, batch_size=64, shuffle=False)

print("\nEvaluating on Advanced Augmented Data:")
evaluate_model(test_advanced_dataloader, model, loss_fn)

