import os
import argparse
import torch
from torchvision import transforms
from PIL import Image

from utils.face_encoder import FaceEncoder
from models import PersonSketchGenerator
from attribute_sketch_dataset import attrs_to_vector

# Try to import LLM attribute extractor, else fallback
try:
    from llm_text_to_attributes import extract_attributes_llm
except ImportError:
    print("Warning: llm_text_to_attributes not found. Using simple fallback.")
    def extract_attributes_llm(text):
        # Stub
        return {}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser(description="Generate a sketch of a specific person.")
    parser.add_argument("--photo", type=str, required=True, help="Path to the person's photo")
    parser.add_argument("--desc", type=str, default="", help="Description needed for attributes (e.g. 'female with short hair')")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--output", type=str, default="output_person_sketch.png", help="Output filename")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Scale of random noise (higher = more diversity)")
    
    args = parser.parse_args()
    
    print(f"Loading models on {DEVICE}...")
    
    # Load Encoder
    encoder = FaceEncoder(device=DEVICE).eval()
    
    # Load Generator
    # Note: Ensure dimensions match training
    generator = PersonSketchGenerator(attr_dim=10, id_dim=512, noise_dim=100).to(DEVICE)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        generator.load_state_dict(checkpoint)
        print("Generator checkpoint loaded.")
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return

    generator.eval()
    
    # Process Photo
    if not os.path.exists(args.photo):
        print(f"Error: Photo {args.photo} not found.")
        return
        
    print(f"Processing photo: {args.photo}...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    raw_img = Image.open(args.photo).convert('RGB')
    photo_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        id_emb = encoder.get_embedding(photo_tensor)
        
    # Process Attributes
    print(f"Extracting attributes from: '{args.desc}'...")
    if args.desc:
        attrs = extract_attributes_llm(args.desc)
    else:
        attrs = {}
        print("No description provided. Using default attributes.")
        
    # Add defaults if missing, to avoid zeros
    if 'gender' not in attrs: attrs['gender'] = 'male' # Default
    if 'hair_length' not in attrs: attrs['hair_length'] = 'short'
    
    print(f"Attributes: {attrs}")
    
    attr_vec = attrs_to_vector(attrs).unsqueeze(0).to(DEVICE)
    
    # Generate
    print("Generating sketch...")
    with torch.no_grad():
        z = torch.randn(1, 100).to(DEVICE) * args.noise_scale
        # attrs, id
        sketch = generator(z, attr_vec, id_emb)
        
    # Save
    sketch = (sketch + 1) / 2 # [-1, 1] -> [0, 1]
    sketch = sketch.clamp(0, 1)
    
    save_img = transforms.ToPILImage()(sketch.squeeze(0).cpu())
    save_img.save(args.output)
    print(f"Saved sketch to {args.output}")

if __name__ == "__main__":
    main()
