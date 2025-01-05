# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 01:14:40 2025

@author: Yunus
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import torchvision

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
        self.ratio = ratio  # Correct attribute name

    def __call__(self, img):
        if random.random() < self.p:
            area = img.size()[1] * img.size()[2]  # Image height * width
            erase_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])  # Use the corrected attribute
            
            h = int((erase_area * aspect_ratio) ** 0.5)
            w = int((erase_area / aspect_ratio) ** 0.5)

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
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0)),
    transforms.ToTensor(),
    AddGaussianNoise(mean=0., std=0.1),
    RandomErasing(p=0.3, scale=(0.02, 0.2)),
])

# Define Beta schedule for noise
def get_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# Function for the forward process (Adding noise over T steps)
def forward_diffusion_process(x_0, timesteps, beta_schedule):
    device = x_0.device
    noise = torch.randn_like(x_0)
    x_t = x_0

    for t in range(timesteps):
        beta_t = beta_schedule[t]
        x_t = torch.sqrt(1 - beta_t) * x_t + torch.sqrt(beta_t) * noise

    return x_t, noise

# Define the reverse process (denoising) model
class DiffusionUNet(nn.Module):
    def __init__(self):
        super(DiffusionUNet, self).__init__()
        self.unet = UNet()

    def forward(self, x_t, t):
        return self.unet(x_t, t)

# Loss function to train the diffusion model
def loss_fn(predicted_image, true_image):
    return nn.MSELoss()(predicted_image, true_image)

# Model Training Function
def train_diffusion_model(dataloader, model, optimizer, timesteps, beta_schedule, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in dataloader:
            images = images.to(device)

            # Forward pass: Add noise to the images over T timesteps
            noisy_images, noise = forward_diffusion_process(images, timesteps, beta_schedule)

            # Forward pass through the reverse model
            predicted_noise = model(noisy_images, torch.randint(0, timesteps, (images.size(0),), device=device))

            # Compute the loss between predicted noise and actual noise
            loss = loss_fn(predicted_noise, noise)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Sampling Function to Generate Images
def sample(model, timesteps, beta_schedule, batch_size=64):
    device = next(model.parameters()).device
    x_t = torch.randn(batch_size, 1, 28, 28).to(device)

    for t in reversed(range(timesteps)):
        beta_t = beta_schedule[t]
        predicted_noise = model(x_t, t)
        
        # Reverse the noise process (denoise)
        x_t = (x_t - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(1 - beta_t)
        
    return x_t

# Evaluate the model on test data
def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # Forward pass: Add noise to the images over T timesteps
            noisy_images, noise = forward_diffusion_process(images, timesteps, beta_schedule)

            # Forward pass through the reverse model
            predicted_noise = model(noisy_images, torch.randint(0, timesteps, (images.size(0),), device=device))

            # Compute the loss between predicted noise and actual noise
            loss = loss_fn(predicted_noise, noise)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(dataloader):.4f}")

# Initialize Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiffusionUNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Beta schedule
timesteps = 1000
beta_schedule = get_beta_schedule(timesteps)

# Load MNIST Dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=augmentation, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train_advanced_dataset = datasets.MNIST(root='./data', train=True, transform=augmentation_advanced, download=True)
train_advanced_dataloader = DataLoader(train_advanced_dataset, batch_size=64, shuffle=True)

# Train the model on unaugmented data
print("Training on Unaugmented Data:")
train_diffusion_model(train_dataloader, model, optimizer, timesteps, beta_schedule)

# Train the model on advanced augmented data
print("\nTraining on Advanced Augmented Data:")
train_diffusion_model(train_advanced_dataloader, model, optimizer, timesteps, beta_schedule)

# Visualization of Augmented Images
def visualize_augmentations(dataloader, num_images=8):
    data_iter = iter(dataloader)
    images, _ = next(data_iter)

    images_grid = torchvision.utils.make_grid(images[:num_images], nrow=4, normalize=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(images_grid.permute(1, 2, 0))  # Convert CHW to HWC for display
    plt.title("Augmented Images")
    plt.axis("off")
    plt.show()

# Visualize Augmented Images
visualize_augmentations(train_dataloader)

# Evaluate Models on Test Set
test_dataset = datasets.MNIST(root='./data', train=False, transform=augmentation, download=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate on Unaugmented Data
print("\nEvaluating on Unaugmented Data:")
evaluate_model(test_dataloader, model, loss_fn)

# Evaluate on Advanced Augmented Data
test_advanced_dataset = datasets.MNIST(root='./data', train=False, transform=augmentation_advanced, download=True)
test_advanced_dataloader = DataLoader(test_advanced_dataset, batch_size=64, shuffle=False)

print("\nEvaluating on Advanced Augmented Data:")
evaluate_model(test_advanced_dataloader, model, loss_fn)
