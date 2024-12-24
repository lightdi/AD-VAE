import torch
from torch import nn

class Critic(nn.Module):
    """Discriminator

    Args:
        w_multi (int): Width multiplier for the convolutional layers
        channel_num (int): Number of input channels
        Nd (int): Dimension of the output
    """

    def __init__(self, w_multi: int, channel_num: int, Nd: int):
        super(Critic, self).__init__()

        self.layers = nn.ModuleList()
        self.current_size = self.out_channels = w_multi
        self.in_channels = channel_num

        while self.current_size < w_multi * 8:
            self.layers.append(self._make_layer(self.in_channels, self.out_channels))
            self.in_channels = self.out_channels
            self.out_channels *= 2
            self.current_size *= 2

        # Finalize with normalization and projection layers
        self.first_norm = nn.BatchNorm2d(self.in_channels)
        self.flatten = nn.Flatten()
        self.project = nn.Linear(self.in_channels * 4 * 4, Nd + 3)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.first_norm(x)
        x = self.flatten(x)
        x = self.project(x)
        return x