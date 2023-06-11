import torch.nn as nn


from app.models.base_encoder_decoder import Encoder, Decoder


class CNNEncoder(Encoder):
    def __init__(self, input_dim, latent_dim):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1) # {(128 - 3 +2)+1}
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64*128, latent_dim) # 8192, 32 (latent)
        self.fc_logvar = nn.Linear(64 * 128, latent_dim)  # 8192, 32 (latent)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape-> (64, 768, 128) -> (batch_size, embed_size, max_length)

        x = self.relu(self.conv1(x)) # (64, 32, 128)
        x = self.relu(self.conv2(x)) # (64, 64, 128)
        x = self.flatten(x) # (64,8192) 64*128=8192
        # x = x.t() # (8192, 64)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)

        return mu, log_var


class CNNDecoder(Decoder):
    def __init__(self, latent_dim, output_dim):
        super(CNNDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * 128)
        self.unflatten = nn.Unflatten(1, (64, 128))
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x -> (8192, 24576)
        # x = x.t()
        x = self.relu(self.fc(x))
        x = self.unflatten(x)
        x = self.relu(self.deconv1(x))
        x = self.sigmoid(self.deconv2(x))
        return x
