import torch
from text_to_attributes import extract_attributes
from attribute_encoder import encode_attributes
from generator import Generator

# Input
description = "Male with short black hair and beard"

# Text → attributes
attrs = extract_attributes(description)

# Attributes → vector
attr_vec = encode_attributes(attrs)
attr_tensor = torch.tensor(attr_vec).float().unsqueeze(0)

# Generator
gen = Generator(attr_dim=len(attr_vec))

# Noise
noise = torch.randn(1, 100)

# Generate sketch
sketch = gen(noise, attr_tensor)

print("Sketch generated:", sketch.shape)
