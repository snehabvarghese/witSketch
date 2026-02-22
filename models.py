import torch
import torch.nn as nn

class AttributeSketchGenerator(nn.Module):
    """
    Generator that produces a sketch from:
    - Random Noise (z)
    - Attribute Vector (c)
    
    Structure:
    (z + c) -> FC -> Reshape -> ResNet/UNet Decoder -> Sketch
    """
    def __init__(self, attr_dim, noise_dim=100, img_size=256):
        super().__init__()
        self.img_size = img_size
        
        # Input dimension info: Noise + Attributes (No ID)
        self.in_dim = noise_dim + attr_dim
        
        # Initial Dense Layer to expand to spatial dimensions
        # Starting at 8x8 spatial resolution
        self.init_channels = 512
        self.init_size = 8
        
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.init_channels * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        # Upsampling blocks (8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.blocks = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 16 -> 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 32 -> 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 -> 128
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 128 -> 256
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Final output layer
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, attrs):
        # noise: (B, 100)
        # attrs: (B, attr_dim)
        x = torch.cat([noise, attrs], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        img = self.blocks(x)
        return img

class PersonSketchGenerator(nn.Module):
    """
    Generator that produces a sketch from:
    - Random Noise (z)
    - Attribute Vector (c)
    - Face Identity Embedding (id)
    
    Structure:
    (z + c + id) -> FC -> Reshape -> ResNet/UNet Decoder -> Sketch
    """
    def __init__(self, attr_dim, id_dim=512, noise_dim=100, img_size=256):
        super().__init__()
        self.img_size = img_size
        
        # Input dimension info
        self.in_dim = noise_dim + attr_dim + id_dim
        
        # Initial Dense Layer to expand to spatial dimensions
        # Starting at 8x8 spatial resolution
        self.init_channels = 512
        self.init_size = 8
        
        self.fc = nn.Sequential(
            nn.Linear(self.in_dim, self.init_channels * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        # Upsampling blocks (8 -> 16 -> 32 -> 64 -> 128 -> 256)
        self.blocks = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 16 -> 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 32 -> 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 -> 128
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # 128 -> 256
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Final output layer
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, attrs, id_emb):
        # Concatenate inputs
        # noise: (B, 100)
        # attrs: (B, attr_dim)
        # id_emb: (B, 512)
        
        x = torch.cat([noise, attrs, id_emb], dim=1)
        x = self.fc(x)
        x = x.view(-1, self.init_channels, self.init_size, self.init_size)
        img = self.blocks(x)
        return img

class Discriminator(nn.Module):
    """
    Standard DCGAN discriminator adapted for conditional generation.
    For 256x256 images, conditioned on `attr_dim`.
    """
    def __init__(self, in_channels=3, use_conditional=False, attr_dim=10):
        super().__init__()
        self.use_conditional = use_conditional
        
        # If conditional, we expand the attribute vector to a full spatial channel map
        # and concatenate it with the input image.
        if self.use_conditional:
            in_channels = in_channels + attr_dim
            
        self.main = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 -> 64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 -> 32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 -> 16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 -> 8
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() 
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, img, attrs=None):
        if self.use_conditional and attrs is not None:
            # attrs: (B, attr_dim) -> reshape to (B, attr_dim, 1, 1) -> expand to (B, attr_dim, 256, 256)
            b, c, h, w = img.size()
            attrs_spatial = attrs.view(b, -1, 1, 1).expand(b, -1, h, w)
            x = torch.cat([img, attrs_spatial], dim=1)
        else:
            x = img
            
        out = self.main(x)
        out = self.avg_pool(out)
        return out.view(-1, 1)
