import os
import cv2
import json
from tqdm import tqdm
from datasets import load_dataset
import warnings

# Ignore dataset warnings
warnings.filterwarnings('ignore')

# Config
DATA_ROOT = "dataset"
CELEBA_DIR = os.path.join(DATA_ROOT, "CelebA")
TRAIN_SUBSET_SIZE = 5000
TEST_SUBSET_SIZE = 500

def create_dirs():
    dirs = [
        os.path.join(CELEBA_DIR, 'train', 'photos'),
        os.path.join(CELEBA_DIR, 'train', 'sketches'),
        os.path.join(CELEBA_DIR, 'test', 'photos'),
        os.path.join(CELEBA_DIR, 'test', 'sketches'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def generate_sketch(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None: return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    cv2.imwrite(save_path, sketch)
    return True

def process():
    create_dirs()
    print("Loading streaming dataset from Hugging Face...")
    iterator = iter(load_dataset('huggan/CelebA-faces-with-attributes', split='train', streaming=True))
    
    def run_split(split, subset_size):
        print(f"Processing {split} split ({subset_size} images)...")
        jsonl_path = os.path.join(CELEBA_DIR, f"{split}.jsonl")
        out_photos_dir = os.path.join(CELEBA_DIR, split, 'photos')
        out_sketches_dir = os.path.join(CELEBA_DIR, split, 'sketches')
        
        count = 0
        with open(jsonl_path, 'w') as f:
            for _ in tqdm(range(subset_size)):
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                    
                fname = f"{split}_{count:06d}.jpg"
                dst_photo = os.path.join(out_photos_dir, fname)
                dst_sketch = os.path.join(out_sketches_dir, fname)
                
                if not os.path.exists(dst_photo):
                    img = item['image']
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(dst_photo)
                    
                if not os.path.exists(dst_sketch):
                    if not generate_sketch(dst_photo, dst_sketch):
                        continue
                        
                record = {'filename': fname}
                
                # Extract attributes
                # E.g. item['Male'] is 0 or 1.
                attrs = item
                
                record['gender'] = 'male' if attrs.get('Male', 0) == 1 else 'female'
                record['glasses'] = 'yes' if attrs.get('Eyeglasses', 0) == 1 else 'no'
                record['beard'] = 'no' if attrs.get('No_Beard', 1) == 1 else 'yes'
                record['hair_length'] = 'long' if record.get('gender') == 'female' else 'short'
                
                if attrs.get('Black_Hair', 0) == 1: record['hair_color'] = 'black'
                elif attrs.get('Blond_Hair', 0) == 1: record['hair_color'] = 'blonde'
                elif attrs.get('Brown_Hair', 0) == 1: record['hair_color'] = 'brown'
                else: record['hair_color'] = 'black'
                
                record['age_group'] = 'young' if attrs.get('Young', 1) == 1 else 'old'
                record['face_shape'] = 'oval'
                
                f.write(json.dumps(record) + '\n')
                count += 1
                
        print(f"Finished {split} split. Saved {count} annotations.")

    run_split('train', TRAIN_SUBSET_SIZE)
    run_split('test', TEST_SUBSET_SIZE)

if __name__ == '__main__':
    process()
    print("Dataset preparation complete!")
