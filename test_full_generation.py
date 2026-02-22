"""
Test the full sketch generation pipeline with corrected mapper.
"""
import torch
from generate_sketch_from_description import (
    extract_attributes_llm,
    fallback_extract,
    generate_sketch
)

# Test description
description = "Male with long black hair and beard"
print(f"Testing with description: {description}\n")

# Extract attributes
print("Extracting attributes...")
attrs = extract_attributes_llm(description)
if attrs is None:
    print("LLM failed, using fallback parser...")
    attrs = fallback_extract(description)

print(f"Extracted attributes: {attrs}\n")

# Generate sketch
print("Generating sketch...")
sketch = generate_sketch(attrs)

# Save output
out_path = "test_generated_sketch.png"
sketch.save(out_path)
print(f"✓ Saved generated sketch to {out_path}")

# Also check the debug files
import os
if os.path.exists('generated_sketch_contrast.png'):
    print("✓ generated_sketch_contrast.png created")
if os.path.exists('generated_sketch_inverted.png'):
    print("✓ generated_sketch_inverted.png created")
