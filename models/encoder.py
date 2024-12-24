import torch
from torch import nn

class Encoder(nn.Module):
    """Encoder for Variational Autoencoder

    Args:
        latent_dim (int): Dimension of the latent space
        channel_num (int): Number of input channels
        w_multi (int): Width multiplier for the convolutional layers
    """

    def __init__(self, latent_dim: int, channel_num: int, w_multi: int):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        self.current_size = self.out_channels = w_multi
        self.in_channels = channel_num

        while self.current_size > 4:
            self.layers.append(self._make_layer(self.in_channels, self.out_channels))
            self.in_channels = self.out_channels
            self.out_channels *= 2
            self.current_size //= 2

        # Finalize with normalization and projection layers
        self.first_norm = nn.BatchNorm2d(self.in_channels)
        self.flatten = nn.Flatten()
        self.dense_mu = nn.Linear(self.in_channels * 4 * 4, latent_dim)
        self.log_sigma = nn.Linear(self.in_channels * 4 * 4, latent_dim)

        self._initialize_weights()

    def _make_layer(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a convolutional layer with batch normalization and LeakyReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def _initialize_weights(self):
        """Initializes weights for the layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x)
        x = self.first_norm(x)
        x = self.flatten(x)
        mu = self.dense_mu(x)
        log_sigma = self.log_sigma(x)
        return mu, log_sigma
