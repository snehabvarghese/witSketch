import os
import argparse
import torch
from torchvision import transforms
from PIL import Image

from models import AttributeSketchGenerator
from attribute_sketch_dataset import attrs_to_vector

# Try to import LLM attribute extractor, else fallback
try:
    from llm_text_to_attributes import extract_attributes_llm
except ImportError:
    print("Warning: llm_text_to_attributes not found. Using simple fallback.")
    def extract_attributes_llm(text):
        return {}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Generate a sketch from description.")
    parser.add_argument("--desc", type=str, required=True, help="Description needed for attributes (e.g. 'female with short hair')")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--output", type=str, default="output_attribute_sketch.png", help="Output filename")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Scale of random noise (higher = more diversity)")
    
    args = parser.parse_args()
    
    print(f"Loading models on {DEVICE}...")
    
    # Load Generator
    generator = AttributeSketchGenerator(attr_dim=10, noise_dim=100).to(DEVICE)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        generator.load_state_dict(checkpoint)
        print("Generator checkpoint loaded.")
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    generator.eval()
    
    # Process Attributes
    print(f"Extracting attributes from: '{args.desc}'...")
    attrs = extract_attributes_llm(args.desc)
    
    # Add defaults if missing
    if 'gender' not in attrs: attrs['gender'] = 'male' 
    if 'hair_length' not in attrs: attrs['hair_length'] = 'short'
    
    print(f"Attributes: {attrs}")
    
    attr_vec = attrs_to_vector(attrs).unsqueeze(0).to(DEVICE)
    
    # Generate
    print("Generating sketch...")
    with torch.no_grad():
        z = torch.randn(1, 100).to(DEVICE) * args.noise_scale
        sketch = generator(z, attr_vec)
        
    # Post-process: [-1, 1] -> [0, 1]
    sketch = (sketch + 1) / 2
    sketch = sketch.clamp(0, 1)
    
    # Contrast Stretch: Normalize [min, max] -> [0, 1] to make lines visible
    s_min, s_max = sketch.min(), sketch.max()
    if s_max - s_min > 1e-5:
        sketch = (sketch - s_min) / (s_max - s_min)
    
    save_img = transforms.ToPILImage()(sketch.squeeze(0).cpu())
    save_img.save(args.output)
    print(f"Saved sketch to {args.output}")

if __name__ == "__main__":
    main()
