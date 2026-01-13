import argparse
import os
import torch
from torch.utils.data import DataLoader
from datasets import SketchDataset
from pix2pix_generator import Generator
from pix2pix_discriminator import Discriminator
import torch.nn as nn


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    dataset = SketchDataset(args.photo_dir, args.sketch_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        for photo, sketch in loader:
            photo = photo.to(device)
            sketch = sketch.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            fake = G(photo)
            real_pred = D(photo, sketch)
            fake_pred = D(photo, fake.detach())

            loss_D = criterion_GAN(real_pred, torch.ones_like(real_pred)) + \
                     criterion_GAN(fake_pred, torch.zeros_like(fake_pred))

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            fake_pred = D(photo, fake)
            loss_G = criterion_GAN(fake_pred, torch.ones_like(fake_pred)) + \
                     args.l1_lambda * criterion_L1(fake, sketch)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch}: G={loss_G.item():.4f}, D={loss_D.item():.4f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
        }
        path = os.path.join(args.checkpoint_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(ckpt, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train simple Pix2Pix')
    parser.add_argument('--photo-dir', default='dataset/CUFS/train/photos')
    parser.add_argument('--sketch-dir', default='dataset/CUFS/train/sketches')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--l1-lambda', dest='l1_lambda', type=float, default=100.0)
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    main(args)
