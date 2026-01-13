import os
import torch
from transformers import pipeline
from PIL import Image
from torchvision import transforms
from pix2pix_generator import Generator  # your Pix2Pix generator class

# ----------------------------
# Step 1: LLM Attribute Extraction
# ----------------------------
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def extract_attributes_llm(description):
    prompt = f"""
    Extract facial attributes from the description.
    Attributes:
    gender, face_shape, hair_length, hair_color, beard, glasses

    Retunern valid JSON only.

    Description:
    {description}
    """
    output = llm(prompt, max_new_tokens=256)[0]["generated_text"]
    import json
    try:
        return json.loads(output)
    except:
        return None


def fallback_extract(description):
    # simple heuristic extraction for common keywords
    text = description.lower()
    attrs = {}
    attrs['gender'] = 'male' if 'male' in text or 'man' in text else ('female' if 'female' in text or 'woman' in text else 'male')
    attrs['hair_length'] = 'long' if 'long hair' in text or 'long-haired' in text else 'short'
    if 'blonde' in text:
        attrs['hair_color'] = 'blonde'
    elif 'brown' in text:
        attrs['hair_color'] = 'brown'
    else:
        attrs['hair_color'] = 'black'
    # handle negations for beard and glasses
    if 'no beard' in text or 'without beard' in text or 'clean shaven' in text:
        attrs['beard'] = 'no'
    elif 'beard' in text or 'mustache' in text:
        attrs['beard'] = 'yes'
    else:
        attrs['beard'] = 'no'

    if 'no glasses' in text or 'without glasses' in text:
        attrs['glasses'] = 'no'
    elif 'glasses' in text or 'spectacles' in text:
        attrs['glasses'] = 'yes'
    else:
        attrs['glasses'] = 'no'
    # face shape default
    attrs['face_shape'] = 'oval'
    return attrs

# ----------------------------
# Step 2: Convert Attributes → GAN Input
# ----------------------------
def attributes_to_tensor(attrs):
    # Map attributes to a 3-channel image tensor compatible with the Pix2Pix generator
    gender = 0.0 if attrs.get('gender', 'male') == 'male' else 1.0
    hair_length = 0.0 if attrs.get('hair_length', 'short') == 'short' else 1.0
    hair_color = {'black': 0.0, 'brown': 1.0, 'blonde': 2.0}.get(attrs.get('hair_color', 'black'), 0.0)

    # normalize to 0..1 then scale to -1..1 (same as training Normalize)
    ch0 = gender
    ch1 = hair_length
    ch2 = hair_color / 2.0

    vec3 = torch.tensor([ch0, ch1, ch2], dtype=torch.float)
    vec3 = vec3 * 2.0 - 1.0

    # expand to (C,H,W) = (3,256,256)
    return vec3.unsqueeze(-1).unsqueeze(-1).repeat(1, 256, 256)

# ----------------------------
# Step 3: Load GAN Model
# ----------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# find a checkpoint to load (prefer explicit final file, else latest epoch ckpt)
ckpt_dir = "checkpoints"
ckpt_path = None
if os.path.exists(os.path.join(ckpt_dir, "pix2pix_final.pth")):
    ckpt_path = os.path.join(ckpt_dir, "pix2pix_final.pth")
else:
    # pick newest .pth file
    if os.path.isdir(ckpt_dir):
        pths = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        if pths:
            ckpt_path = sorted(pths, key=os.path.getmtime)[-1]

generator = Generator().to(device)
if ckpt_path is None:
    raise FileNotFoundError("No checkpoint found in checkpoints/; run training first or supply a checkpoint named pix2pix_final.pth")

ckpt = torch.load(ckpt_path, map_location=device)
if isinstance(ckpt, dict) and 'G_state' in ckpt:
    generator.load_state_dict(ckpt['G_state'])
elif isinstance(ckpt, dict) and 'G_state_dict' in ckpt:
    generator.load_state_dict(ckpt['G_state_dict'])
elif isinstance(ckpt, dict) and any(k.startswith('G') or k.startswith('generator') for k in ckpt.keys()):
    # attempt to find a nested state dict
    if 'G_state' in ckpt:
        generator.load_state_dict(ckpt['G_state'])
    else:
        # assume this is a state_dict itself
        generator.load_state_dict(ckpt)
else:
    # assume state_dict
    generator.load_state_dict(ckpt)

generator.eval()

# ----------------------------
# Step 4: Generate Sketch
# ----------------------------
def generate_sketch(attrs):
    attr_tensor = attributes_to_tensor(attrs).unsqueeze(0).to(device)  # batch dim
    with torch.no_grad():
        sketch_tensor = generator(attr_tensor)
    # generator uses Tanh -> outputs in [-1, 1]; convert to [0, 1]
    sketch_tensor = (sketch_tensor + 1.0) / 2.0
    sketch_tensor = sketch_tensor.clamp(0.0, 1.0)
    # Convert to PIL image
    sketch_img = transforms.ToPILImage()(sketch_tensor.squeeze(0).cpu())
    return sketch_img

# ----------------------------
# Step 5: Generate Multiple Sketches (Optional)
# ----------------------------
def generate_multiple_sketches(attrs, n=5):
    sketches = []
    attr_tensor = attributes_to_tensor(attrs).unsqueeze(0).to(device)
    for _ in range(n):
        noise = torch.randn_like(attr_tensor)
        with torch.no_grad():
            sketch_tensor = generator(attr_tensor + noise)
        sketch_tensor = (sketch_tensor + 1.0) / 2.0
        sketch_tensor = sketch_tensor.clamp(0.0, 1.0)
        sketches.append(transforms.ToPILImage()(sketch_tensor.squeeze(0).cpu()))
    return sketches

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    description = input("Enter witness description: ")
    attrs = extract_attributes_llm(description)
    if attrs is None:
        print("LLM failed to extract attributes. Falling back to simple parser.")
        attrs = fallback_extract(description)

    print("Extracted attributes:", attrs)
    sketch = generate_sketch(attrs)
    # save and show
    out_path = "generated_sketch.png"
    sketch.save(out_path)
    print(f"Saved generated sketch to {out_path}")
    try:
        sketch.show()
    except Exception:
        pass
        # Optional: generate multiple sketches
        # sketches = generate_multiple_sketches(attrs)
