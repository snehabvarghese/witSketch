"""
create_mock_db.py  (attribute-enriched, sketch-to-sketch matching)

For each CUFS person:
 - Reads their sketch + attributes from train.jsonl / test.jsonl
 - Embeds the sketch with FaceNet (sketch embedding)
 - Stores: name, crime, attributes dict, attr_vector, sketch_path, photo_path, embedding
"""

import os
import json
import random
import torch
from PIL import Image
from torchvision import transforms
from utils.face_encoder import FaceEncoder
from attribute_sketch_dataset import attrs_to_vector

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_SKETCHES_DIR = "dataset/CelebA/train/sketches"
TRAIN_PHOTOS_DIR   = "dataset/CelebA/train/photos"
TRAIN_JSONL        = "dataset/CelebA/train.jsonl"
TEST_SKETCHES_DIR  = "dataset/CelebA/test/sketches"
TEST_PHOTOS_DIR    = "dataset/CelebA/test/photos"
TEST_JSONL         = "dataset/CelebA/test.jsonl"
OUTPUT_FILE        = "criminal_records.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Fake data ───────────────────────────────────────────────────────────────
CRIMES = ["Grand Theft Auto","Burglary","Cyber Fraud","Arson","Assault",
          "Money Laundering","Identity Theft","Public Intoxication",
          "Vandalism","Petty Theft","Embezzlement","Racketeering"]
LOCATIONS = [
    "Downtown", "Northside", "West End", "South Park", 
    "Eastside", "Midtown", "Harbor District", "Industrial Zone"
]
MALE_NAMES   = ["John","Michael","David","Robert","William","James","Joseph",
                "Charles","Liam","Noah","Aiden","Ravi","Wei","Carlos"]
FEMALE_NAMES = ["Jane","Emily","Sarah","Jessica","Ashley","Amanda","Jennifer",
                "Olivia","Emma","Priya","Anjali","Mei","Sofia","Laura"]
LAST_NAMES   = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
                "Davis","Rodriguez","Martinez","Kumar","Patel","Lee","Chen"]

def load_jsonl(path: str) -> dict:
    """Return {filename -> attrs_dict} for all records in a JSONL file."""
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fname = item.pop("filename")
            mapping[fname] = item
    return mapping

def find_photo(filename: str, photos_dir: str) -> str:
    base = os.path.splitext(filename)[0].replace("-sz1", "")
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(photos_dir, base + ext)
        if os.path.exists(p):
            return p
    return ""

def generate_record(filename: str, attrs: dict, sketch_path: str, photo_path: str) -> dict:
    gender = attrs.get("gender", "male")
    age_group = attrs.get("age_group", "young")
    first  = random.choice(FEMALE_NAMES if gender == "female" else MALE_NAMES)
    
    # Generate an age based on the age group
    if age_group == "young":
        age = random.randint(18, 35)
    else:
        age = random.randint(36, 65)
        
    return {
        "id":          os.path.splitext(filename)[0],
        "name":        f"{first} {random.choice(LAST_NAMES)}",
        "age":         age,
        "crime":       random.choice(CRIMES),
        "location":    random.choice(LOCATIONS),
        "sentence":    f"{random.randint(1, 20)} years",
        "risk_level":  random.choice(["Low", "Medium", "High", "Critical"]),
        "past_arrests": random.randint(0, 5),
        "sketch_path": sketch_path,
        "photo_path":  photo_path,
        "attributes":  attrs,          # store raw attrs dict for display
    }

def process_split(sketches_dir, photos_dir, jsonl_path,
                  encoder, transform, records):
    if not os.path.exists(sketches_dir):
        print(f"  Skipping {sketches_dir} (not found)")
        return

    attr_map = load_jsonl(jsonl_path)
    files = sorted(f for f in os.listdir(sketches_dir)
                   if f.lower().endswith((".jpg",".jpeg",".png")))
    print(f"  Processing {len(files)} sketches from {sketches_dir}...")

    for i, filename in enumerate(files):
        sketch_path = os.path.join(sketches_dir, filename)
        photo_path  = find_photo(filename, photos_dir)

        # Attributes — from JSONL or fallback from filename
        attrs = attr_map.get(filename, {})
        if not attrs:
            # Fallback if something went wrong
            attrs = {
                "gender":     "male",
                "hair_length":"short",
                "hair_color": "black",
                "beard":      "no",
                "glasses":    "no",
                "face_shape": "oval",
                "age_group":  "young"
            }

        try:
            # Embed the sketch image
            img = Image.open(sketch_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = encoder.get_embedding(img_tensor)
                emb_list = emb.cpu().numpy().tolist()[0]

            # Compute attribute vector and store as list
            attr_vec = attrs_to_vector(attrs).tolist()

            record = generate_record(filename, attrs, sketch_path, photo_path)
            record["embedding"]   = emb_list   # FaceNet sketch embedding (512D)
            record["attr_vector"] = attr_vec   # Attribute vector (10D)
            records.append(record)

        except Exception as e:
            print(f"    Failed {filename}: {e}")

        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(files)} done")

def create_db():
    print(f"Initializing FaceEncoder on {DEVICE}...")
    encoder = FaceEncoder(device=DEVICE).eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    records = []
    print("\n[1/2] Train split:")
    process_split(TRAIN_SKETCHES_DIR, TRAIN_PHOTOS_DIR, TRAIN_JSONL,
                  encoder, transform, records)
    print("\n[2/2] Test split:")
    process_split(TEST_SKETCHES_DIR, TEST_PHOTOS_DIR, TEST_JSONL,
                  encoder, transform, records)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(records, f, indent=4)

    print(f"\n✓ Saved {len(records)} attribute-enriched records to {OUTPUT_FILE}")
    print("  Each record now stores: attributes dict + attr_vector + sketch embedding.")

if __name__ == "__main__":
    create_db()
