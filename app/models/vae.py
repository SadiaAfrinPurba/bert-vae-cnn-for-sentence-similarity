import torch
import torch.nn as nn

from app.models.base_encoder_decoder import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, latent_dim: int):
        super(VAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        # x shape-> (64, 768, 128) -> (batch_size, embed_size, max_length)
        mu_logvar = self.encoder(x) # (8192, 49152)
        # mu = mu_logvar[:, :self.latent_dim] # (8192, 32)
        # logvar = mu_logvar[:, self.latent_dim:] # (8192, 49120)
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1) # (8192, 24576), (8192, 24576)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar