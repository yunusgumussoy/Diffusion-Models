# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 21:00:08 2025

@author: Yunus
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
image_size = 28  # For example, a 28x28 image
timesteps = 100  # Number of diffusion steps
beta_start = 1e-4
beta_end = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"

# Linear beta schedule
beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
alpha = 1.0 - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)  # Cumulative product of alpha

# Noise schedule visualization
plt.plot(alpha_cumprod.cpu().numpy())
plt.title("Alpha Cumulative Product")
plt.show()

# A simple U-Net-style model for denoising
class DenoisingModel(nn.Module):
    def __init__(self):
        super(DenoisingModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        # Create a sinusoidal time embedding
        t_emb = torch.sin(
            t.float().unsqueeze(1).unsqueeze(2).unsqueeze(3) * torch.arange(1, x.size(1) + 1).to(x.device)
        )
        t_emb = t_emb.expand_as(x)  # Ensure it matches the input shape
        return self.net(x + t_emb)

model = DenoisingModel().to(device)

# Forward diffusion process: Add noise
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0).to(device)
    alpha_t = alpha_cumprod[t].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    noisy_image = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
    return noisy_image, noise

# Training step
def train_step(x0, t, model, optimizer, loss_fn):
    noisy_image, noise = forward_diffusion(x0, t)
    predicted_noise = model(noisy_image, t)
    loss = loss_fn(predicted_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# Sampling: Reverse diffusion process
@torch.no_grad()
def sample(model, shape, timesteps):
    x = torch.randn(shape).to(device)  # Start with random noise
    for t in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)  # Batch of time steps
        alpha_t = alpha_cumprod[t].view(1, 1, 1, 1).to(device)  # Reshape alpha_t for broadcasting
        beta_t = beta[t].view(1, 1, 1, 1).to(device)  # Reshape beta_t for broadcasting
        predicted_noise = model(x, t_tensor)
        x = (x - beta_t * predicted_noise) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(1 - alpha_t) * noise
    return x

# Dummy dataset and training loop
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

dataset = MNIST(root="./data", train=True, transform=ToTensor(), download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training the model
for epoch in range(2):  # For brevity, only 2 epochs
    for batch, _ in dataloader:
        batch = batch.to(device)
        t = torch.randint(0, timesteps, (batch.size(0),), device=device)
        loss = train_step(batch, t, model, optimizer, loss_fn)
    print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

# Sampling from the model
samples = sample(model, (16, 1, image_size, image_size), timesteps).cpu()

# Visualize samples
fig, axs = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(samples[i, 0], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()
