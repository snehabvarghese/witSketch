import torch
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, photo, sketch):
        x = torch.cat([photo, sketch], dim=1)
        return self.model(x)
