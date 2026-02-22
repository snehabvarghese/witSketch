import os
import json
from PIL import Image
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def encode_attributes(attrs: dict) -> torch.Tensor:
    """Encode attribute dict to 3-channel tensor used by mapper.
    Reuses the same mapping as `generate_sketch_from_description.attributes_to_tensor`.
    """
    gender = 0.0 if attrs.get('gender', 'male') == 'male' else 1.0
    hair_length = 0.0 if attrs.get('hair_length', 'short') == 'short' else 1.0
    hair_color = {'black': 0.0, 'brown': 1.0, 'blonde': 2.0}.get(attrs.get('hair_color', 'black'), 0.0)

    ch0 = gender
    ch1 = hair_length
    ch2 = hair_color / 2.0

    vec3 = torch.tensor([ch0, ch1, ch2], dtype=torch.float)
    vec3 = vec3 * 2.0 - 1.0
    return vec3.unsqueeze(-1).unsqueeze(-1).repeat(1, 256, 256)


def attrs_to_vector(attrs: dict) -> torch.Tensor:
    """Expose a compact vector encoding for learned encoders.
    Delegates to `attr_encoder.attrs_to_vector` if available, else uses same discrete mapping.
    """
    try:
        from attr_encoder import attrs_to_vector as a2v
        return a2v(attrs)
    except Exception:
        # fallback: create a small vector similar to LearnedAttrEncoder expectations
        gender = 1.0 if attrs.get('gender', 'male') == 'female' else 0.0
        hair_length = 1.0 if attrs.get('hair_length', 'short') == 'long' else 0.0
        hc = attrs.get('hair_color', 'black')
        hair_black = 1.0 if hc == 'black' else 0.0
        hair_brown = 1.0 if hc == 'brown' else 0.0
        hair_blonde = 1.0 if hc == 'blonde' else 0.0
        beard = 1.0 if attrs.get('beard', 'no') == 'yes' else 0.0
        glasses = 1.0 if attrs.get('glasses', 'no') == 'yes' else 0.0
        fs = attrs.get('face_shape', 'oval')
        fs_oval = 1.0 if fs == 'oval' else 0.0
        fs_round = 1.0 if fs == 'round' else 0.0
        fs_square = 1.0 if fs == 'square' else 0.0
        return torch.tensor([gender, hair_length, hair_black, hair_brown, hair_blonde, beard, glasses, fs_oval, fs_round, fs_square], dtype=torch.float)


class AttributeSketchDataset(Dataset):
    """Dataset for attribute -> sketch pairs.

    attrs_csv can be either:
    - a CSV with header containing `filename` and attribute columns
    - a JSON Lines file where each line is {"filename": ..., "gender":..., ...}
    """

    def __init__(
        self,
        sketches_dir: str,
        attrs_path: str,
        photos_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.sketches_dir = sketches_dir
        self.attrs_path = attrs_path
        self.photos_dir = photos_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # load attributes mapping
        self.records = []  # list of (filename, attrs_dict)
        if attrs_path.lower().endswith('.csv'):
            import csv
            with open(attrs_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    filename = r.get('filename')
                    if not filename:
                        continue
                    # keep rest as attributes
                    attrs = {k: v for k, v in r.items() if k != 'filename' and v is not None}
                    # coerce simple types
                    for k, v in list(attrs.items()):
                        if v.lower() in ('yes', 'y', 'true', '1'):
                            attrs[k] = 'yes'
                        elif v.lower() in ('no', 'n', 'false', '0'):
                            attrs[k] = 'no'
                    self.records.append((filename, attrs))
        else:
            # assume JSON lines or JSON list
            with open(attrs_path, 'r', encoding='utf-8') as f:
                first = f.read(4096)
                f.seek(0)
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            filename = item.get('filename')
                            if filename:
                                attrs = {k: v for k, v in item.items() if k != 'filename'}
                                self.records.append((filename, attrs))
                    elif isinstance(data, dict):
                        # maybe a map
                        for filename, attrs in data.items():
                            self.records.append((filename, attrs))
                except Exception:
                    # fallback to JSON lines
                    f.seek(0)
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        filename = item.get('filename')
                        if filename:
                            attrs = {k: v for k, v in item.items() if k != 'filename'}
                            self.records.append((filename, attrs))

        # filter records to existing sketch files
        self.records = [r for r in self.records if os.path.exists(os.path.join(sketches_dir, r[0]))]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        filename, attrs = self.records[idx]
        path = os.path.join(self.sketches_dir, filename)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        # encode attrs into 3-channel tensor
        attr_tensor = encode_attributes(attrs)
        # also return compact vector for learned encoder training
        attr_vector = attrs_to_vector(attrs)
        
        # Load photo if available
        photo_img = torch.zeros_like(img) # Placeholder
        if self.photos_dir:
            # Try to find corresponding photo
            # Photos might have different extensions or exact match
            photo_path = os.path.join(self.photos_dir, filename)
            if not os.path.exists(photo_path):
                # Try common extensions
                base = os.path.splitext(filename)[0]
                for ext in ['.jpg', '.png', '.jpeg']:
                    p = os.path.join(self.photos_dir, base + ext)
                    if os.path.exists(p):
                        photo_path = p
                        break
            
            if os.path.exists(photo_path):
                try:
                    p_img = Image.open(photo_path).convert('RGB')
                    photo_img = self.transform(p_img)
                except Exception as e:
                    print(f"Error loading photo {photo_path}: {e}")
            
        return attr_tensor, img, photo_img, filename, attr_vector
