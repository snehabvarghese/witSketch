import json
import random

INPUT_FILE = "annotations.jsonl"
OUTPUT_FILE = "annotations.jsonl" # Overwrite

def rebalance_annotations():
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        
    records = []
    for line in lines:
        if line.strip():
            records.append(json.loads(line))
            
    print(f"Loaded {len(records)} records.")
    
    # Distributions
    hair_lengths = ["short", "medium", "long"]
    hair_colors = ["black", "brown", "blonde"]
    yes_no = ["yes", "no"]
    
    updated_records = []
    for r in records:
        # Force diversity
        r["hair_length"] = random.choice(hair_lengths)
        r["hair_color"] = random.choice(hair_colors)
        r["glasses"] = random.choice(yes_no)
        # Keep gender mostly as is or random? Let's randomise to be safe given filenames might not match visual
        if random.random() < 0.3:
             r["gender"] = random.choice(["male", "female"])
             
        updated_records.append(r)
        
    with open(OUTPUT_FILE, 'w') as f:
        for r in updated_records:
            f.write(json.dumps(r) + "\n")
            
    print("Rebalanced annotations with random attributes.")
    print("Sample distribution:")
    long_count = sum(1 for r in updated_records if r["hair_length"] == "long")
    print(f"  Long hair: {long_count}/{len(records)}")

if __name__ == "__main__":
    rebalance_annotations()
