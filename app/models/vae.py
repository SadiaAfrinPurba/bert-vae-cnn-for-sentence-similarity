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

        mu, logvar = self.encoder(x) # -> output shape of mu and logvar (64,32)
        z = self.reparameterize(mu, logvar) # output shape of z -> (64, 32)
        x_hat = self.decoder(z) # output shape of x_hat -> (64, 768, 128)

        return x_hat, mu, logvar