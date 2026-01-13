import numpy as np
from attributes import ATTRIBUTES

def encode_attributes(attr_dict):
    vector = []

    for key, values in ATTRIBUTES.items():
        for v in values:
            vector.append(1 if attr_dict[key] == v else 0)

    return np.array(vector)
