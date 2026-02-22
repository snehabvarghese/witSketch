import os
import json

def get_attributes_from_filename(filename):
    """Infer attributes from CUFS filenames.
    e.g., f-039-01.jpg -> gender: female
          m-001-01.jpg -> gender: male
    """
    fname = os.path.basename(filename).lower()
    attrs = {
        # Defaults
        'hair_length': 'short', # Can't infer easily without vision
        'hair_color': 'black',  # Most CUFS are Asian/Indian faces often, so black is safe default but not always true
        'beard': 'no',
        'glasses': 'no',
        'face_shape': 'oval'
    }
    
    if fname.startswith('f'):
        attrs['gender'] = 'female'
        attrs['hair_length'] = 'long' # Bias for female
    elif fname.startswith('m'):
        attrs['gender'] = 'male'
        attrs['hair_length'] = 'short' # Bias for male
    else:
        # Fallback
        attrs['gender'] = 'male'
        
    return attrs

def process_directory(base_dir, split):
    photos_dir = os.path.join(base_dir, split, 'photos')
    sketches_dir = os.path.join(base_dir, split, 'sketches')
    out_path = os.path.join(base_dir, f'{split}.jsonl')
    
    if not os.path.exists(sketches_dir):
        print(f"Directory not found: {sketches_dir}")
        return

    print(f"Proocessing {split} split...")
    count = 0
    with open(out_path, 'w') as f:
        # Iterate sketches as they are the target
        for fname in sorted(os.listdir(sketches_dir)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            # Check if corresponding photo exists (optional but good for verification)
            # CUFS usually has same name for sketch and photo
            
            attrs = get_attributes_from_filename(fname)
            record = {'filename': fname}
            record.update(attrs)
            
            f.write(json.dumps(record) + '\n')
            count += 1
            
    print(f"Generated {count} annotations in {out_path}")

def main():
    base_dir = 'dataset/CUFS'
    process_directory(base_dir, 'train')
    process_directory(base_dir, 'test')

if __name__ == '__main__':
    main()
