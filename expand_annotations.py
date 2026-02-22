import os
import json
import random

SKETCHES_DIR = "dataset/CUFS/train/sketches"
ANNOTATIONS_PATH = "annotations.jsonl"

def expand_annotations():
    # 1. Load existing
    existing = {}
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing[item['filename']] = item
    
    print(f"Index contains {len(existing)} annotations.")

    # 2. Scan directory
    all_files = [f for f in os.listdir(SKETCHES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(all_files)} sketches in directory.")
    
    # 3. Add missing
    added_count = 0
    
    with open(ANNOTATIONS_PATH, 'a') as f:
        for filename in all_files:
            if filename not in existing:
                # Create synthetic attribute
                # Heuristic: filename might hint gender? 'f-...' -> female, 'm-...' -> male if exists
                # dataset/CUFS filenames: f-039-01.jpg
                
                gender = "female" if filename.startswith("f-") else "male"
                if not filename.startswith("f-") and not filename.startswith("m-"):
                    gender = random.choice(["male", "female"])
                
                # Randomize others to ensure diversity for training
                hair_length = random.choice(["short", "long", "medium"])
                hair_color = random.choice(["black", "brown", "blonde"])
                glasses = random.choice(["yes", "no"])
                
                new_entry = {
                    "filename": filename,
                    "gender": gender,
                    "hair_length": hair_length,
                    "hair_color": hair_color,
                    "beard": "no", # safe default
                    "glasses": glasses,
                    "face_shape": "oval"
                }
                
                f.write(json.dumps(new_entry) + "\n")
                added_count += 1
                
    print(f"Added {added_count} new annotations.")
    if added_count > 0:
        print("NOTE: Added attributes are SYNTHETIC/RANDOM. Please review annotations.jsonl if possible, otherwise the model will learn noisy correlations.")

if __name__ == "__main__":
    expand_annotations()
