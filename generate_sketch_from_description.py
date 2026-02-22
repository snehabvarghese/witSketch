import os
import torch
import json
import random
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from llm_text_to_attributes import extract_attributes_llm

# ----------------------------
# Configuration
# ----------------------------
GLOBAL_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ----------------------------
# Dataset & Attribute Helpers
# ----------------------------
def load_dataset_stats(annotations_path="annotations.jsonl"):
    """
    Loads the dataset and computes simple statistics/priors.
    Returns a dict mapping attribute names to lists of observed values.
    """
    stats = {
        "gender": [], "hair_length": [], "hair_color": [], 
        "beard": [], "glasses": [], "face_shape": []
    }
    
    if not os.path.exists(annotations_path):
        print(f"Warning: {annotations_path} not found. Attribute refinement will be skipped.")
        return None

    try:
        with open(annotations_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    for k in stats.keys():
                        if k in data:
                            stats[k].append(data[k])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading stats: {e}")
        return None

    return stats

def refine_attributes(extracted_attrs, dataset_stats):
    """
    Refines extracted attributes using dataset statistics.
    If an attribute is missing or ambiguous, sample from the dataset distribution.
    """
    refined = extracted_attrs.copy() if extracted_attrs else {}
    
    if not dataset_stats:
        return refined
        
    for k, v in dataset_stats.items():
        # If attribute missing or None, sample from prior
        if k not in refined or not refined[k]:
            if v: # Ensure we have data to sample from
                refined[k] = random.choice(v)
            
    return refined

def construct_guided_description(attrs):
    """
    Constructs a descriptive string from structured attributes.
    """
    if not attrs:
        return ""
        
    gender = attrs.get("gender", "person")
    hair_len = attrs.get("hair_length", "medium")
    hair_col = attrs.get("hair_color", "")
    face_shape = attrs.get("face_shape", "")
    beard = attrs.get("beard", "no")
    glasses = attrs.get("glasses", "no")
    
    parts = [f"a {gender}"]
    
    hair_part = f"{hair_len}"
    if hair_col:
        hair_part += f" {hair_col}"
    parts.append(f"with {hair_part} hair")
    
    if face_shape:
        parts.append(f"and {face_shape} face")
        
    if beard == "yes":
        parts.append("wearing a beard")
    if glasses == "yes":
        parts.append("wearing glasses")
        
    return ", ".join(parts)

# ----------------------------
# Diffusion Pipeline Class
# ----------------------------
class TextToSketchPipeline:
    """
    Text-conditioned sketch generation using Stable Diffusion.
    """
    def __init__(self, model_id=MODEL_ID, device=GLOBAL_DEVICE):
        self.device = torch.device(device)
        print(f"Loading diffusion model from {model_id} on {self.device}...")
        
        # Load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        
        # Use DPMSolver for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

    def generate(self, prompt, attributes=None, num_inference_steps=20, guidance_scale=7.5):
        """
        Generate a sketch from the description.
        Injects 'sketch', 'pencil drawing', etc. into the prompt to force the style.
        """
        # Style prompt engineering
        style_prefix = "professional pencil sketch of "
        style_suffix = ", detailed charcoal drawing, high contrast, white background, realistic face details, monochrome, rough lines"
        negative_prompt = "color, photo, photorealistic, 3d render, cartoon, anime, low quality, bad anatomy, deformed"

        # Construct prompt with attribute guidance
        core_subject = prompt
        
        # Ensure attributes is not None for safety
        if attributes is None:
            attributes = {}
            
        if attributes:
            guided_desc = construct_guided_description(attributes)
            if guided_desc:
                print(f"Guided description from attributes: {guided_desc}")
                # Combine guided description with original prompt for nuances
                core_subject = f"{guided_desc}. {prompt}"

        full_prompt = f"{style_prefix}{core_subject}{style_suffix}"
        
        print(f"Generating for prompt: {full_prompt}")
        
        with torch.no_grad():
            image = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
        
        # Post-process: Ensure grayscale
        return image.convert("L")

# ----------------------------
# Helper: Attribute Extraction (Legacy/Stub removed)
# ----------------------------
# (extract_attributes_llm is imported now)

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    print(f"Initializing WitSketch Diffusion Pipeline on {GLOBAL_DEVICE}...")
    
    # Load dataset stats
    print("Loading dataset statistics from annotations.jsonl...")
    stats = load_dataset_stats("annotations.jsonl")
    if stats:
        print(f"Loaded stats for {len(stats)} attributes.")
    else:
        print("No stats loaded.")

    pipeline = TextToSketchPipeline()
    
    while True:
        try:
            description = input("\nEnter witness description (or 'q' to quit): ")
            if description.lower() in ['q', 'quit', 'exit']:
                break
            
            if not description.strip():
                continue
                
            # 1. Extract attributes
            print("Extracting attributes...")
            raw_attrs = extract_attributes_llm(description)
            print(f"Raw extracted: {raw_attrs}")
            
            # 2. Refine with dataset stats
            refined_attrs = refine_attributes(raw_attrs, stats)
            print(f"Refined attributes: {refined_attrs}")
            
            # 3. Generate
            sketch = pipeline.generate(description, attributes=refined_attrs)
            
            # Save output
            filename = "generated_sketch.png"
            sketch.save(filename)
            print(f"Saved sketch to {os.path.abspath(filename)}")
            
            # Show image
            try:
                sketch.show()
            except:
                pass
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error generating sketch: {e}")
