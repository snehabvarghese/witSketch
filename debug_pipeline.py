"""
Debug script to inspect intermediate outputs in the pipeline.
"""
import os
import torch
from PIL import Image
from torchvision import transforms
from mapper import SimpleMapper
from pix2pix_generator import Generator

# Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}\n")

def attributes_to_tensor(attrs):
    """Current 4-channel attribute tensor generation"""
    gender = 0.0 if attrs.get('gender', 'male') == 'male' else 1.0
    hair_length = 0.0 if attrs.get('hair_length', 'short') == 'short' else 1.0
    hair_color = {'black': 0.0, 'brown': 1.0, 'blonde': 2.0}.get(attrs.get('hair_color', 'black'), 0.0)
    
    beard = 1.0 if attrs.get('beard', 'no') == 'yes' else 0.0
    glasses = 1.0 if attrs.get('glasses', 'no') == 'yes' else 0.0
    
    face_shape_map = {'oval': 0.0, 'round': 0.5, 'square': 1.0}
    face_shape = face_shape_map.get(attrs.get('face_shape', 'oval'), 0.0)

    ch0 = (gender * 0.6 + face_shape * 0.4)
    ch1 = (hair_length * 0.6 + beard * 0.4)
    ch2 = (hair_color / 2.0 * 0.6 + glasses * 0.4)

    vec4 = torch.tensor([ch0, ch1, ch2, 0.0], dtype=torch.float)
    vec4 = vec4 * 2.0 - 1.0
    vec4[3] = torch.randn(1).item()

    expanded = vec4.unsqueeze(-1).unsqueeze(-1).repeat(1, 256, 256)
    
    grid = torch.linspace(-1, 1, 256)
    yy, xx = torch.meshgrid(grid, grid, indexing='ij')
    dist_from_center = torch.sqrt(xx**2 + yy**2)
    dist_normalized = (dist_from_center / dist_from_center.max()) * 2.0 - 1.0
    
    expanded[0] = expanded[0] * 0.7 + dist_normalized * 0.3
    
    spatial_noise = torch.randn_like(expanded[3:4])
    expanded[3] = expanded[3] * 0.5 + spatial_noise * 0.5
    
    return expanded


# Test attributes
test_attrs = {
    'gender': 'male',
    'face_shape': 'oval',
    'hair_length': 'long',
    'hair_color': 'black',
    'beard': 'yes',
    'glasses': 'no'
}

print(f"Test attributes: {test_attrs}\n")

# Generate attribute tensor
attr_tensor = attributes_to_tensor(test_attrs).unsqueeze(0).to(device)
print(f"Attribute tensor shape: {attr_tensor.shape}")
print(f"Attribute tensor range: [{attr_tensor.min():.3f}, {attr_tensor.max():.3f}]")
print(f"Attribute tensor mean: {attr_tensor.mean():.3f}, std: {attr_tensor.std():.3f}\n")

# Load mapper
mapper = SimpleMapper(in_ch=4).to(device)
mapper_path = 'checkpoints/mapper_with_encoder.pth'

print(f"Loading mapper from: {mapper_path}")
raw = torch.load(mapper_path, map_location=device)
if isinstance(raw, dict):
    if 'state_dict' in raw:
        sd = raw['state_dict']
    elif 'model_state_dict' in raw:
        sd = raw['model_state_dict']
    else:
        sd = raw
else:
    sd = raw

mapper.load_state_dict(sd)
mapper.eval()
print(f"✓ Mapper loaded\n")

# Test mapper forward pass
print("Testing mapper output...")
with torch.no_grad():
    photo = mapper(attr_tensor)

print(f"Mapper output shape: {photo.shape}")
print(f"Mapper output range: [{photo.min():.3f}, {photo.max():.3f}]")
print(f"Mapper output mean: {photo.mean():.3f}, std: {photo.std():.3f}\n")

# Save mapper output for inspection
photo_normalized = (photo + 1.0) / 2.0
photo_normalized = photo_normalized.clamp(0.0, 1.0)
pil_photo = transforms.ToPILImage()(photo_normalized.squeeze(0).cpu())
pil_photo.save("debug_mapper_output.png")
print(f"✓ Saved mapper output to debug_mapper_output.png\n")

# Load generator
print("Loading pix2pix generator...")
generator = Generator().to(device)

# Find best checkpoint
ckpt_dir = "checkpoints"
all_pths = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
pix_files = [p for p in all_pths if 'ckpt_epoch' in p or 'pix2pix' in os.path.basename(p).lower()]
if pix_files:
    pix_ckpt = sorted(pix_files, key=os.path.getmtime)[-1]
else:
    non_mapper = [p for p in all_pths if 'mapper' not in os.path.basename(p).lower()]
    pix_ckpt = sorted(non_mapper if non_mapper else all_pths, key=os.path.getmtime)[-1]

print(f"Loading checkpoint: {pix_ckpt}")
ckpt = torch.load(pix_ckpt, map_location=device)
if isinstance(ckpt, dict) and 'G_state' in ckpt:
    generator.load_state_dict(ckpt['G_state'])
elif isinstance(ckpt, dict) and 'G_state_dict' in ckpt:
    generator.load_state_dict(ckpt['G_state_dict'])
else:
    try:
        generator.load_state_dict(ckpt)
    except:
        for v in ckpt.values() if isinstance(ckpt, dict) else []:
            if isinstance(v, dict):
                try:
                    generator.load_state_dict(v)
                    break
                except:
                    continue

generator.eval()
print(f"✓ Generator loaded\n")

# Test generator forward pass
print("Testing generator output...")
with torch.no_grad():
    sketch_tensor = generator(photo)

print(f"Generator output shape: {sketch_tensor.shape}")
print(f"Generator output range (raw): [{sketch_tensor.min():.3f}, {sketch_tensor.max():.3f}]")
print(f"Generator output mean: {sketch_tensor.mean():.3f}, std: {sketch_tensor.std():.3f}\n")

# Convert to [0, 1]
sketch_01 = (sketch_tensor + 1.0) / 2.0
sketch_01 = sketch_01.clamp(0.0, 1.0)

print(f"Generator output range (after normalization): [{sketch_01.min():.3f}, {sketch_01.max():.3f}]")
print(f"Generator output mean (normalized): {sketch_01.mean():.3f}, std: {sketch_01.std():.3f}\n")

# Save raw normalized output
pil_raw = transforms.ToPILImage()(sketch_01.squeeze(0).cpu())
pil_raw.save("debug_generator_raw.png")
print(f"✓ Saved raw generator output to debug_generator_raw.png\n")

# Apply current post-processing
print("Testing post-processing...")
mean = sketch_01.mean().item()
std = sketch_01.std().item()
print(f"Sketch stats: mean={mean:.3f}, std={std:.3f}")

if std < 1e-6:
    std = 1.0
    print(f"  -> std too small, using 1.0")

norm = (sketch_01 - mean) / (std + 1e-6) * 0.6 + 0.5
norm = norm.clamp(0.0, 1.0)

print(f"After contrast normalization:")
print(f"  Range: [{norm.min():.3f}, {norm.max():.3f}]")
print(f"  Mean: {norm.mean():.3f}, std: {norm.std():.3f}\n")

pil_norm = transforms.ToPILImage()(norm.squeeze(0).cpu())
pil_norm.save("debug_generator_normalized.png")
print(f"✓ Saved normalized output to debug_generator_normalized.png\n")

# Also invert
inverted = (1.0 - norm).clamp(0.0, 1.0)
pil_inv = transforms.ToPILImage()(inverted.squeeze(0).cpu())
pil_inv.save("debug_generator_inverted.png")
print(f"✓ Saved inverted output to debug_generator_inverted.png\n")

print("All debug outputs saved!")
print("Check these files to understand what's happening:")
print("  - debug_mapper_output.png: mapper's pseudo-photo")
print("  - debug_generator_raw.png: raw generator output")
print("  - debug_generator_normalized.png: after contrast normalization")
print("  - debug_generator_inverted.png: inverted version")
