import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import argparse
import glob

from attribute_sketch_dataset import AttributeSketchDataset
from models import AttributeSketchGenerator, Discriminator

# Configuration
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 4 # Reduced batch size for fine-tuning if data is small
LR = 0.0001    # Lower LR for fine-tuning
NOISE_DIM = 100
ATTR_DIM = 10 

# Paths
SKETCHES_DIR = "dataset/CUFS/train/sketches"
# Use the annotation file we generated
ATTRS_PATH = "dataset/CUFS/train.jsonl"
PHOTOS_DIR = "dataset/CUFS/train/photos"

# Checkpoint to fine-tune FROM
PRETRAINED_G = "checkpoints_attribute/generator_epoch_2000.pth"
PRETRAINED_D = "checkpoints_attribute/discriminator_epoch_2000.pth"

# Where to save NEW checkpoints
CHECKPOINT_DIR = "checkpoints_cufs"
SAMPLE_DIR = "samples_cufs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs")
    args = parser.parse_args()
    EPOCHS = args.epochs

    print(f"Initializing CUFS fine-tuning on {DEVICE}...")
    
    # Dataset
    # We include photos_dir just in case, though generator logic uses attributes mostly
    if not os.path.exists(ATTRS_PATH):
        print(f"Error: {ATTRS_PATH} not found. Run cufs_preprocess.py first.")
        return

    dataset = AttributeSketchDataset(
        sketches_dir=SKETCHES_DIR,
        attrs_path=ATTRS_PATH,
        photos_dir=PHOTOS_DIR 
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check paths.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Dataset loaded: {len(dataset)} samples.")

    # Models
    generator = AttributeSketchGenerator(attr_dim=ATTR_DIM, noise_dim=NOISE_DIM).to(DEVICE)
    discriminator = Discriminator(in_channels=3, use_conditional=True, attr_dim=ATTR_DIM).to(DEVICE)
    
    # Load Pretrained
    start_epoch = 0
    if os.path.exists(PRETRAINED_G):
        print(f"Loading pretrained generator from {PRETRAINED_G}")
        generator.load_state_dict(torch.load(PRETRAINED_G, map_location=DEVICE))
    else:
        print("Warning: Pretrained generator not found. Training from scratch.")
        
        if os.path.exists(PRETRAINED_D):
            print(f"Loading pretrained discriminator from {PRETRAINED_D}")
            try:
                discriminator.load_state_dict(torch.load(PRETRAINED_D, map_location=DEVICE), strict=False)
            except Exception as e:
                print(f"Failed to load discriminator weights perfectly (expected due to architecture changes): {e}")
    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    # Losses
    criterion_gan = nn.MSELoss() 
    # Removed L1 loss to prevent mode collapse. 
    # The conditional discriminator will enforce attribute matching now.
    
    print("Starting training loop...")
    
    for epoch in range(EPOCHS):
        for i, (attr_map, real_sketch, _, filename, attr_vec) in enumerate(dataloader):
            
            real_sketch = real_sketch.to(DEVICE)
            attr_vec = attr_vec.to(DEVICE) # (B, 10)
            
            batch_size = real_sketch.size(0)
            
            # labels
            valid = torch.ones(batch_size, 1).to(DEVICE)
            fake = torch.zeros(batch_size, 1).to(DEVICE)
            
            # -----------------
            # Train Generator
            # -----------------
            opt_g.zero_grad()
            
            z = torch.randn(batch_size, NOISE_DIM).to(DEVICE)
            # Generate sketch from attributes (which we inferred from filename)
            gen_sketch = generator(z, attr_vec)
            
            # Adv Loss (Conditional)
            pred_fake = discriminator(gen_sketch, attr_vec)
            loss_gan = criterion_gan(pred_fake, valid)
            
            # Total G Loss
            loss_g = loss_gan
            
            loss_g.backward()
            opt_g.step()
            
            # -----------------
            # Train Discriminator
            # -----------------
            opt_d.zero_grad()
            
            # Real loss (Conditional)
            pred_real = discriminator(real_sketch, attr_vec)
            loss_real = criterion_gan(pred_real, valid)
            
            # Fake loss (Conditional)
            pred_fake = discriminator(gen_sketch.detach(), attr_vec)
            loss_fake = criterion_gan(pred_fake, fake)
            
            loss_d = 0.5 * (loss_real + loss_fake)
            
            loss_d.backward()
            opt_d.step()
            
            # Logging
            if i % 5 == 0:
                print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {loss_d.item():.4f}] [G loss: {loss_g.item():.4f}]")
                
        # Save sample
        save_image(gen_sketch, f"{SAMPLE_DIR}/epoch_{epoch}.png", nrow=4, normalize=True)
        print(f"Saved sample for epoch {epoch}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/generator_cufs_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{CHECKPOINT_DIR}/discriminator_cufs_epoch_{epoch+1}.pth")
            print(f"Saved checkpoint for epoch {epoch+1}")

if __name__ == "__main__":
    train()
