"""CLI annotator for sketch attributes.

Usage:
    python annotate_sketches.py --sketch-dir path/to/sketches --out attrs.jsonl --start 0 --count 200

This will iterate images and prompt in terminal. It optionally opens the image using the default image viewer.
"""
import os
import json
import argparse
from PIL import Image


def prompt_attributes(filename):
    print('\nAnnotating:', filename)
    try:
        Image.open(filename).show()
    except Exception:
        pass
    gender = input('gender (male/female) [male]: ').strip() or 'male'
    hair_length = input('hair_length (short/long) [short]: ').strip() or 'short'
    hair_color = input('hair_color (black/brown/blonde) [black]: ').strip() or 'black'
    beard = input('beard (yes/no) [no]: ').strip() or 'no'
    glasses = input('glasses (yes/no) [no]: ').strip() or 'no'
    face_shape = input('face_shape (oval/round/square) [oval]: ').strip() or 'oval'
    return {
        'gender': gender,
        'hair_length': hair_length,
        'hair_color': hair_color,
        'beard': beard,
        'glasses': glasses,
        'face_shape': face_shape,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketch-dir', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--extension', default='.png')
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.sketch_dir) if f.lower().endswith(args.extension)])
    files = files[args.start:args.start + args.count]

    with open(args.out, 'a', encoding='utf-8') as out_f:
        for fname in files:
            path = os.path.join(args.sketch_dir, fname)
            attrs = prompt_attributes(path)
            record = {'filename': fname}
            record.update(attrs)
            out_f.write(json.dumps(record) + '\n')
            out_f.flush()
            print('Saved annotation for', fname)

if __name__ == '__main__':
    main()
