# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:14:39 2025

@author: Yunus
"""
# Generative Adversarial Network (GAN)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 5
lr = 0.0002

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()  # Output scaled to [-1, 1]
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# Prepare Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize Models, Optimizers, and Loss Function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
loss_fn = nn.BCELoss()

# Training Loop
for epoch in range(epochs):
    for real_images, _ in dataloader:
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z).detach()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_loss = loss_fn(discriminator(real_images), real_labels)
        fake_loss = loss_fn(discriminator(fake_images), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = generator(z)
        g_loss = loss_fn(discriminator(fake_images), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# Generate Samples
import matplotlib.pyplot as plt
z = torch.randn(16, latent_dim).to(device)
samples = generator(z).detach().cpu()
grid = torch.cat([img for img in samples], dim=2).squeeze().numpy()
plt.imshow(grid, cmap='gray')
plt.show()
