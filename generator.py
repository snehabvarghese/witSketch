import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, attr_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100 + attr_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256*256),
            nn.Tanh()
        )

    def forward(self, noise, attrs):
        x = torch.cat((noise, attrs), dim=1)
        img = self.fc(x)
        return img.view(-1, 1, 256, 256)
