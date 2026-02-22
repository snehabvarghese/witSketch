import torch
from torchvision.utils import save_image
from models import AttributeSketchGenerator
from attribute_sketch_dataset import attrs_to_vector
import os

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints_cufs/generator_cufs_epoch_25.pth"

gen = AttributeSketchGenerator(attr_dim=10, noise_dim=100).to(DEVICE)
gen.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
gen.eval()

# Generate 5 examples for a single description
attrs = {"gender": "female", "hair_length": "long"}
attr_vec = attrs_to_vector(attrs).unsqueeze(0).to(DEVICE)

images = []
with torch.no_grad():
    for i in range(5):
        z = torch.randn(1, 100).to(DEVICE)
        sketch = gen(z, attr_vec)
        sketch = (sketch + 1) / 2
        images.append(sketch)

all_images = torch.cat(images, dim=0) # (5, 3, 256, 256)
save_image(all_images, "test_diversity.png", nrow=5)
print("Saved 5 distinct generated sketches to test_diversity.png")
