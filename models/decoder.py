
from torch import nn

class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, latent_dim, channel_num, w_mult):
        super(Decoder, self).__init__()
        self.w_mult = w_mult

        self.layers = nn.ModuleList()
        self.current_size = 4
        self.in_channels = w_mult
        self.out_channels = self.in_channels // 2

        # Adiciona camadas até que a dimensão atual seja igual ao w_mult
        while self.current_size < w_mult:
            self.layers.append(self._make_layer(self.in_channels, self.out_channels))
            self.in_channels = self.out_channels
            self.out_channels //= 2
            self.current_size *= 2

        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, channel_num, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        self.project = nn.Linear(latent_dim, self.in_channels * 4 * 4, bias=False)
        self.project_norm = nn.BatchNorm2d(self.in_channels)
        self.activation = nn.ReLU()

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.project(input)
        x = x.view(-1, self.in_channels, 4, 4)
        x = self.project_norm(x)
        x = self.activation(x)

        for layer in reversed(self.layers):
            x = layer(x)

        x = self.to_rgb(x)
        return x