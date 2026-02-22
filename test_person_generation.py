import torch
import os
from utils.face_encoder import FaceEncoder
from models import PersonSketchGenerator
from attribute_sketch_dataset import AttributeSketchDataset

def test_components():
    print("=== Testing Components ===")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Test FaceEncoder
    print("\n[1] Testing FaceEncoder...")
    try:
        encoder = FaceEncoder(device=device)
        dummy_img = torch.randn(2, 3, 256, 256).to(device)
        emb = encoder.get_embedding(dummy_img)
        print(f"  Img shape: {dummy_img.shape}")
        print(f"  Emb shape: {emb.shape}")
        assert emb.shape == (2, 512)
        print("  ✓ FaceEncoder working")
    except Exception as e:
        print(f"  X FaceEncoder failed: {e}")

    # 2. Test Generator
    print("\n[2] Testing Generator...")
    try:
        gen = PersonSketchGenerator(attr_dim=10, id_dim=512, noise_dim=100).to(device)
        z = torch.randn(2, 100).to(device)
        attrs = torch.randn(2, 10).to(device)
        id_emb = torch.randn(2, 512).to(device)
        
        out = gen(z, attrs, id_emb)
        print(f"  Output shape: {out.shape}")
        assert out.shape == (2, 1, 256, 256)
        print("  ✓ Generator working")
    except Exception as e:
        print(f"  X Generator failed: {e}")

    # 3. Test Dataset
    print("\n[3] Testing AttributeSketchDataset...")
    try:
        # Create dummy locations if not exist for test
        sketches_dir = "dataset/CUFS/train/sketches"
        photos_dir = "dataset/CUFS/train/photos"
        attrs_path = "annotations.jsonl"
        
        if os.path.exists(sketches_dir) and os.path.exists(photos_dir) and os.path.exists(attrs_path):
            ds = AttributeSketchDataset(sketches_dir, attrs_path, photos_dir=photos_dir)
            if len(ds) > 0:
                item = ds[0]
                # attr_tensor, img, photo_img, filename, attr_vector
                print(f"  Item 0 filename: {item[3]}")
                print(f"  Sketch shape: {item[1].shape}")
                print(f"  Photo shape: {item[2].shape}")
                
                # Check if photo is placeholder or real
                if item[2].sum() == 0:
                    print("  ! Photo is all zeros (placeholder). Check file matching.")
                else:
                    print("  ✓ Photo loaded correctly")
            else:
                print("  ! Dataset empty (filtered by existence?)")
        else:
            print("  ! Dataset paths do not exist, skipping dataset test")
    except Exception as e:
        print(f"  X Dataset failed: {e}")

if __name__ == "__main__":
    test_components()
