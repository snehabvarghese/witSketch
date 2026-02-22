import torch
import torch.nn as nn
from torchvision import transforms
import ssl

# Bypass SSL verification for model downloads (common issue on Mac)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

class FaceEncoder(nn.Module):
    """
    Encoder to extract identity embeddings from face images.
    Prioritizes facenet-pytorch (InceptionResnetV1) if available.
    Falls back to a standard ResNet18 if not.
    """
    def __init__(self, device='cpu', pretrained=True):
        super().__init__()
        self.device = device
        self.use_facenet = False
        
        try:
            from facenet_pytorch import InceptionResnetV1
            print("Loading FaceNet InceptionResnetV1...")
            # pretrained='vggface2' is standard for recognition
            self.model = InceptionResnetV1(pretrained='vggface2' if pretrained else None).eval()
            self.use_facenet = True
            self.output_dim = 512
        except ImportError:
            print("facenet-pytorch not found. Falling back to ResNet18.")
            from torchvision.models import resnet18, ResNet18_Weights
            # Use default pretrained weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
            # Remove classification layer to get embeddings
            # ResNet18 fc in: 512
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.output_dim = 512
            self.use_facenet = False

        self.model.to(self.device)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        
        # Facenet expects strictly cropped faces ~160x160 usually, but can handle others.
        # Training data is 256x256. 
        # Normalize to what the model expects.
        # Facenet-pytorch expects whitened images or specific normalization, 
        # but the library handles it if we pass raw tensors? 
        # Actually InceptionResnetV1 expects standardized inputs.
        # For simplicity, we assume input is [0,1] or [-1,1].
        
    def forward(self, x):
        """
        Input: Batch of images (B, 3, H, W).
               Run assumes images are roughly aligned faces.
        Output: Embedding vector (B, 512)
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # If input is [-1, 1], convert to what model roughly expects.
        # ResNet usually expects [0, 1] normalized by ImageNet stats.
        # FaceNet (vggface2) usually expects standard normalization.
        # Here we just pass through, assuming the caller handles basic scaling or we add normalization here if strictly needed.
        # For now, simplistic pass.
        
        with torch.no_grad():
            emb = self.model(x)
            
        if self.use_facenet:
            # Facenet output is already (B, 512)
            pass
        else:
            # ResNet output is (B, 512, 1, 1), flatten it
            emb = emb.view(emb.size(0), -1)
            
        return emb

    def get_embedding(self, x):
        return self(x)
