# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 13:15:50 2025

@author: Yunus
"""
# VAE (Variational Autoencoder)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Hyperparameters
latent_dim = 2
batch_size = 64
epochs = 5
lr = 1e-3

# VAE Model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# Loss Function (Reconstruction + KL Divergence)
def loss_fn(recon_x, x, mu, logvar):
    bce = nn.BCELoss(reduction='sum')(recon_x, x.view(x.size(0), -1))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

# Prepare Data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize Model, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        recon_images, mu, logvar = model(images)
        loss = loss_fn(recon_images, images, mu, logvar)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss / len(dataloader.dataset):.4f}")

# Generate Samples
model.eval()
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    samples = model.decoder(z).view(-1, 1, 28, 28).cpu()
    grid = torch.cat([img for img in samples], dim=2).squeeze().numpy()
    plt.imshow(grid, cmap='gray')
    plt.show()
