"""
Visual quality test - generate multiple sketches and check output characteristics.
"""
import torch
from generate_sketch_from_description import generate_sketch
import os

# Test with several attribute combinations
test_cases = [
    {
        'name': 'Male_Short_Black_Beard',
        'attrs': {
            'gender': 'male',
            'hair_length': 'short',
            'hair_color': 'black',
            'beard': 'yes',
            'glasses': 'no',
            'face_shape': 'oval'
        }
    },
    {
        'name': 'Female_Long_Blonde_Glasses',
        'attrs': {
            'gender': 'female',
            'hair_length': 'long',
            'hair_color': 'blonde',
            'beard': 'no',
            'glasses': 'yes',
            'face_shape': 'round'
        }
    },
]

print("Generating test sketches...\n")

for test in test_cases:
    name = test['name']
    attrs = test['attrs']
    
    print(f"Generating: {name}")
    print(f"  Attributes: {attrs}")
    
    try:
        sketch = generate_sketch(attrs)
        out_path = f"quality_test_{name}.png"
        sketch.save(out_path)
        
        # Check file size
        file_size = os.path.getsize(out_path)
        print(f"  ✓ Generated: {out_path} ({file_size:,} bytes)")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print()

print("Quality test complete!")
print("\nCheck the generated quality_test_*.png files")
