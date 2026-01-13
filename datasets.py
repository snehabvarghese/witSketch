import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class SketchDataset(Dataset):
    def __init__(self, photo_dir, sketch_dir):
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        photo_files = sorted(os.listdir(photo_dir))
        sketch_files = set(os.listdir(sketch_dir))
        # keep only files that exist in both directories
        self.images = [f for f in photo_files if f in sketch_files]
        if len(self.images) == 0:
            raise RuntimeError(f"No matching files found in {photo_dir} and {sketch_dir}")

        # separate transforms for RGB photos and single-channel sketches
        self.photo_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.sketch_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        photo = Image.open(os.path.join(self.photo_dir, img_name)).convert("RGB")
        sketch = Image.open(os.path.join(self.sketch_dir, img_name)).convert("L")

        return self.photo_transform(photo), self.sketch_transform(sketch)
