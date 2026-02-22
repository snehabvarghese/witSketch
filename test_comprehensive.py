"""
Comprehensive test of the sketch generation fix with various attribute combinations.
"""
import torch
from generate_sketch_from_description import (
    fallback_extract,
    generate_sketch
)

test_cases = [
    ("Short hair, no beard", {
        'gender': 'male',
        'hair_length': 'short',
        'hair_color': 'black',
        'beard': 'no',
        'glasses': 'no',
        'face_shape': 'oval'
    }),
    ("Long hair, with glasses", {
        'gender': 'female',
        'hair_length': 'long',
        'hair_color': 'blonde',
        'beard': 'no',
        'glasses': 'yes',
        'face_shape': 'round'
    }),
    ("Male with beard", {
        'gender': 'male',
        'hair_length': 'medium',
        'hair_color': 'brown',
        'beard': 'yes',
        'glasses': 'no',
        'face_shape': 'square'
    }),
]

for desc, attrs in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {desc}")
    print(f"Attributes: {attrs}")
    print(f"{'='*60}")
    
    try:
        sketch = generate_sketch(attrs)
        out_path = f"test_sketch_{desc.replace(' ', '_').replace(',', '')}.png"
        sketch.save(out_path)
        print(f"✓ Successfully generated: {out_path}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("All tests completed!")
print(f"{'='*60}")
