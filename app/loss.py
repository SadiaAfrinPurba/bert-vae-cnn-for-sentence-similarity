import torch
import torch.nn as nn


def calculate_vae_loss(x, x_hat, mu, logvar, kl_coefficient):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = reconstruction_loss + kl_loss * kl_coefficient
    return total_loss