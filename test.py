import os
import shutil
import random
import re

photo_folder = "dataset/CUFS/train/photos"
sketch_folder = "dataset/CUFS/train/sketches"

test_photo_folder = "dataset/CUFS/test/photos"
test_sketch_folder = "dataset/CUFS/test/sketches"

os.makedirs(test_photo_folder, exist_ok=True)
os.makedirs(test_sketch_folder, exist_ok=True)

# List of photo files
photos = os.listdir(photo_folder)
num_test = int(0.2 * len(photos))
test_files = random.sample(photos, num_test)

for photo_file in test_files:
    # Move photo
    shutil.move(os.path.join(photo_folder, photo_file), os.path.join(test_photo_folder, photo_file))
    
    # Extract number from photo filename (e.g., 005 from f-005-01.jpg)
    photo_num = re.search(r'\d+', photo_file).group()
    
    # Find matching sketch containing this number
    sketch_file = None
    for f in os.listdir(sketch_folder):
        if photo_num in f:
            sketch_file = f
            break
    
    if sketch_file:
        shutil.move(os.path.join(sketch_folder, sketch_file), os.path.join(test_sketch_folder, sketch_file))
    else:
        print(f"No matching sketch found for photo {photo_file}")
