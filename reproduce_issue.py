import torch
import torch.nn as nn
import os
from models import AttributeSketchGenerator

def check_variation():
    device = "cpu"
    
    # Try to find a valid checkpoint
    checkpoint_dir = "checkpoints_attribute"
    # sort by epoch
    candidates = [f for f in os.listdir(checkpoint_dir) if f.startswith("generator_epoch_")]
    candidates.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
    target_ckpt = candidates[-1] # Last one
    ckpt_path = os.path.join(checkpoint_dir, target_ckpt)
    
    gen = AttributeSketchGenerator(attr_dim=10, noise_dim=100).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        gen.load_state_dict(ckpt, strict=False)
    except:
        return
    gen.eval()
    
    attr_vec = torch.randn(1, 10).to(device)
    
    print("\n--- Test: Scaling Z ---")
    z1 = torch.randn(1, 100).to(device)
    z2 = torch.randn(1, 100).to(device)
    
    # Normal Z
    with torch.no_grad():
        out1 = gen(z1, attr_vec)
        out2 = gen(z2, attr_vec)
    diff_normal = (out1 - out2).abs().mean().item()
    print(f"Diff (Scale 1.0): {diff_normal:.6f}")
    
    # Scaled Z
    scale = 3.0
    with torch.no_grad():
        out1s = gen(z1 * scale, attr_vec)
        out2s = gen(z2 * scale, attr_vec)
    diff_scaled = (out1s - out2s).abs().mean().item()
    print(f"Diff (Scale {scale}): {diff_scaled:.6f}")
    
    if diff_scaled > diff_normal * 1.5:
        print(">> SUCCESS: Scaling Z increases diversity significantly.")
    else:
        print(">> FAIL: Scaling Z has little effect.")

if __name__ == "__main__":
    check_variation()
