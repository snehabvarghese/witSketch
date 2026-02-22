import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from attribute_sketch_dataset import AttributeSketchDataset
from utils.face_encoder import FaceEncoder
from models import PersonSketchGenerator, Discriminator

import argparse

# Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8 # Small batch for safety
LR = 0.0002
NOISE_DIM = 100
ATTR_DIM = 10 
ID_DIM = 512

# Paths
SKETCHES_DIR = "dataset/CUFS/train/sketches"
PHOTOS_DIR = "dataset/CUFS/train/photos"
ATTRS_PATH = "annotations.jsonl"
CHECKPOINT_DIR = "checkpoints_person"
SAMPLE_DIR = "samples_person"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    EPOCHS = args.epochs

    print(f"Initializing training on {DEVICE}...")
    
    # Dataset
    dataset = AttributeSketchDataset(
        sketches_dir=SKETCHES_DIR,
        attrs_path=ATTRS_PATH,
        photos_dir=PHOTOS_DIR
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset loaded: {len(dataset)} samples.")

    # Models
    face_encoder = FaceEncoder(device=DEVICE).eval() # Frozen
    generator = PersonSketchGenerator(attr_dim=ATTR_DIM, id_dim=ID_DIM, noise_dim=NOISE_DIM).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    # Losses
    criterion_gan = nn.MSELoss() # LSGAN is more stable usually, or BCE
    criterion_pixel = nn.L1Loss()
    
    print("Starting training loop...")
    
    for epoch in range(EPOCHS):
        for i, (attr_map, real_sketch, real_photo, filename, attr_vec) in enumerate(dataloader):
            
            real_sketch = real_sketch.to(DEVICE)
            real_photo = real_photo.to(DEVICE)
            attr_vec = attr_vec.to(DEVICE) # (B, 10)
            
            # 1. Extract Identity Embedding
            with torch.no_grad():
                id_emb = face_encoder.get_embedding(real_photo) # (B, 512)
            
            batch_size = real_sketch.size(0)
            
            # labels
            valid = torch.ones(batch_size, 1).to(DEVICE)
            fake = torch.zeros(batch_size, 1).to(DEVICE)
            
            # -----------------
            # Train Generator
            # -----------------
            opt_g.zero_grad()
            
            z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            gen_sketch = generator(z, attr_vec, id_emb)
            
            # Adv Loss
            pred_fake = discriminator(gen_sketch)
            loss_gan = criterion_gan(pred_fake, valid)
            
            # Pixel Loss
            loss_pixel = criterion_pixel(gen_sketch, real_sketch)
            
            # Total G Loss
            loss_g = loss_gan + 100 * loss_pixel # 100 is lambda for L1
            
            loss_g.backward()
            opt_g.step()
            
            # -----------------
            # Train Discriminator
            # -----------------
            opt_d.zero_grad()
            
            # Real loss
            pred_real = discriminator(real_sketch)
            loss_real = criterion_gan(pred_real, valid)
            
            # Fake loss
            pred_fake = discriminator(gen_sketch.detach())
            loss_fake = criterion_gan(pred_fake, fake)
            
            loss_d = 0.5 * (loss_real + loss_fake)
            
            loss_d.backward()
            opt_d.step()
            
            # Logging
            if i % 10 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")
                
        # Save sample
        save_image(gen_sketch[:16], f"{SAMPLE_DIR}/epoch_{epoch}.png", nrow=4, normalize=True)
        print(f"Saved sample for epoch {epoch}")
        
        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    train()
