import os
import re

sketch_folder = "dataset/CUFS/train/sketches"

for f in os.listdir(sketch_folder):
    # Extract number
    num = re.search(r'\d+', f).group()  # e.g., '005'
    # Create new name in the format of photos (e.g., f-005-01.jpg)
    new_name = f"f-{num}-01.jpg"
    old_path = os.path.join(sketch_folder, f)
    new_path = os.path.join(sketch_folder, new_name)
    os.rename(old_path, new_path)