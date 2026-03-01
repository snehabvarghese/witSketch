import os
import json
import base64
import io
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torchvision import transforms

from models import AttributeSketchGenerator
from utils.face_encoder import FaceEncoder
from attribute_sketch_dataset import attrs_to_vector
from diffusion_generator import DiffusionSketchGenerator

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints_attribute/generator_epoch_2000.pth" # Best stable checkpoint
DB_PATH = "criminal_records.json"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (Frontend)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount dataset for photos
if os.path.exists("dataset"):
    app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

# Mount Face Sketch Elements
if os.path.exists("Face Sketch Elements"):
    app.mount("/elements", StaticFiles(directory="Face Sketch Elements"), name="elements")

@app.get("/api/elements")
async def get_face_elements():
    """Returns a list of all available face sketch elements categorized."""
    base_dir = "Face Sketch Elements"
    if not os.path.exists(base_dir):
        return {"error": "Elements directory not found."}
        
    elements = {}
    
    # Valid categories we expect to find in the folder
    categories = ["head", "hair", "eyes", "eyebrows", "nose", "lips", "mustach", "more"]
    
    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if os.path.exists(cat_dir):
            files = []
            for f in sorted(os.listdir(cat_dir)):
                if f.lower().endswith('.png') and not f.lower().startswith('group'):
                    # Provide the URL path that the frontend will use to load the image
                    files.append(f"/elements/{category}/{f}")
            elements[category] = files
            
    return elements


# --- Global State ---
model_state = {
    "generator": None,       # Legacy GAN (fallback)
    "diffusion_gen": None,   # Primary: Stable Diffusion
    "encoder": None,
    "db": None,
    "db_embeddings": None,
    "db_attr_vectors": None  # 10D attribute vectors for each DB record
}

# --- Helpers ---
def load_models():
    print(f"Loading models on {DEVICE}...")

    # 1. Primary: Stable Diffusion generator
    try:
        diff_gen = DiffusionSketchGenerator(device=DEVICE)
        diff_gen.load()
        model_state["diffusion_gen"] = diff_gen
        print("Diffusion generator ready.")
    except Exception as e:
        print(f"Warning: Could not load diffusion model: {e}")

    # 2. Fallback: Legacy GAN generator
    gen = AttributeSketchGenerator(attr_dim=10, noise_dim=100).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        gen.load_state_dict(ckpt, strict=False)
        gen.eval()
        model_state["generator"] = gen
        print("Legacy GAN generator loaded (fallback).")
    else:
        print(f"Warning: GAN checkpoint {CHECKPOINT_PATH} not found.")

    # 3. Encoder
    enc = FaceEncoder(device=DEVICE).eval()
    model_state["encoder"] = enc
    print("Encoder loaded.")

    # 4. Database
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            data = json.load(f)
            model_state["db"] = data
            # Sketch embeddings (512D) for image-based matching
            embs = [d['embedding'] for d in data]
            model_state["db_embeddings"] = torch.tensor(embs).to(DEVICE)
            # Attribute vectors (10D) for description-based matching
            if data and 'attr_vector' in data[0]:
                attr_vecs = [d['attr_vector'] for d in data]
                model_state["db_attr_vectors"] = torch.tensor(attr_vecs).to(DEVICE)
                print(f"Database loaded ({len(data)} records, attribute vectors ready).")
            else:
                print(f"Database loaded ({len(data)} records). Note: no attr_vectors found — re-run create_mock_db.py")
    else:
        print("Warning: Database not found. Matching will fail.")

@app.on_event("startup")
async def startup_event():
    load_models()

def extract_attributes_simple(text: str):
    """Keyword matching to extract rich facial attributes from a description."""
    text = text.lower()
    attrs = {}
    
    # Gender
    if any(w in text for w in ["female", "woman", "girl", "lady", "she", "her"]):
        attrs["gender"] = "female"
    elif any(w in text for w in ["male", "man", "boy", "guy", "he", "his"]):
        attrs["gender"] = "male"
    else:
        attrs["gender"] = "male"
    
    # Hair length
    if any(w in text for w in ["long hair", "long, flowing", "flowing hair"]):
        attrs["hair_length"] = "long"
    elif any(w in text for w in ["short hair", "cropped", "buzz cut", "bald"]):
        attrs["hair_length"] = "short"
    elif any(w in text for w in ["medium hair", "mid-length", "shoulder-length"]):
        attrs["hair_length"] = "medium"
    else:
        attrs["hair_length"] = "short" if attrs["gender"] == "male" else "long"

    # Hair color
    if any(w in text for w in ["blonde", "blond", "golden hair", "fair hair"]):
        attrs["hair_color"] = "blonde"
    elif any(w in text for w in ["brown hair", "brunette", "dark brown"]):
        attrs["hair_color"] = "brown"
    elif any(w in text for w in ["gray hair", "grey hair", "silver hair", "white hair"]):
        attrs["hair_color"] = "brown"  # closest mapping
    else:
        attrs["hair_color"] = "black"
        
    # Beard / Facial hair
    if any(w in text for w in ["beard", "moustache", "mustache", "stubble", "goatee"]):
        attrs["beard"] = "yes"
    else:
        attrs["beard"] = "no"
        
    # Glasses
    if any(w in text for w in ["glasses", "spectacles", "eyeglasses"]):
        attrs["glasses"] = "yes"
    else:
        attrs["glasses"] = "no"
        
    # Face shape
    if any(w in text for w in ["round face", "chubby", "full face"]):
        attrs["face_shape"] = "round"
    elif any(w in text for w in ["square jaw", "angular", "square face"]):
        attrs["face_shape"] = "square"
    else:
        attrs["face_shape"] = "oval"
    
    return attrs

def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- API Models ---
class GenerateRequest(BaseModel):
    description: str
    noise_scale: float = 1.0
    manual_attributes: dict = None
    mode: str = "diffusion"  # "diffusion" (accurate) or "gan" (fast fallback)

class GenerateFromBuilderRequest(BaseModel):
    image_b64: str
    prompt: Optional[str] = ""
    style: Optional[str] = "sketch"

# --- Endpoints ---

# --- Auth Models ---
class LoginRequest(BaseModel):
    username: str
    password: str

# --- Global Stats ---
stats = {
    "generated_count": 0,
    "matched_count": 0,
    "active_users": 0
}

# --- Endpoints ---

@app.get("/")
async def read_root():
    from fastapi.responses import FileResponse
    return FileResponse('static/login.html')

@app.post("/login")
async def login(req: LoginRequest):
    # Hardcoded auth
    if req.username == "admin" and req.password == "admin123":
        return {"token": "ADMIN_TOKEN", "role": "admin"}
    elif req.username == "user" and req.password == "user123":
        stats["active_users"] += 1
        return {"token": "USER_TOKEN", "role": "user"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/generate")
async def generate_sketch(req: GenerateRequest):
    stats["generated_count"] += 1

    # ── Mode: Stable Diffusion (accurate, ~30s) ───────────────────────────
    diff_gen = model_state["diffusion_gen"]
    if req.mode == "diffusion" and diff_gen is not None:
        print(f"[Diffusion] Generating: '{req.description}'")
        try:
            pil_img = diff_gen.generate(
                description=req.description,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=None,  # random each call
            )
            b64_img = pil_to_base64(pil_img)
            return {
                "image": f"data:image/png;base64,{b64_img}",
                "attributes": {"mode": "diffusion", "prompt": req.description},
            }
        except Exception as e:
            print(f"[Diffusion] Error: {e} — falling back to GAN")

    # ── Mode: GAN fallback (fast, less accurate) ──────────────────────────
    gen = model_state["generator"]
    if gen is None:
        raise HTTPException(status_code=500, detail="No generator available")

    target_attrs = extract_attributes_simple(req.description)
    if req.manual_attributes:
        target_attrs.update(req.manual_attributes)

    print(f"[GAN] Generating for: {target_attrs}")
    attr_vec = attrs_to_vector(target_attrs).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z = torch.randn(1, 100).to(DEVICE) * req.noise_scale
        sketch = gen(z, attr_vec)

    # Denormalize
    sketch = (sketch + 1) / 2
    sketch = sketch.clamp(0, 1)
    pil_img = transforms.ToPILImage()(sketch.squeeze(0).cpu())

    # Pencil-sketch post-processing
    pil_gray = pil_img.convert("L")
    pil_gray = ImageOps.autocontrast(pil_gray, cutoff=2)
    pil_gray = pil_gray.filter(ImageFilter.SHARPEN)
    pil_gray = pil_gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
    pil_gray = ImageEnhance.Contrast(pil_gray).enhance(1.5)
    pil_img = pil_gray.convert("RGB")

    b64_img = pil_to_base64(pil_img)
    return {
        "image": f"data:image/png;base64,{b64_img}",
        "attributes": target_attrs,
    }

@app.post("/generate_from_builder")
async def generate_from_builder(req: GenerateFromBuilderRequest):
    stats["generated_count"] += 1
    
    diff_gen = model_state["diffusion_gen"]
    if diff_gen is None or not hasattr(diff_gen, 'img2img_pipe'):
        raise HTTPException(status_code=500, detail="Img2Img generator not available")
        
    try:
        # Decode base64 image
        header, encoded = req.image_b64.split(",", 1)
        image_data = base64.b64decode(encoded)
        init_image = Image.open(io.BytesIO(image_data))
        
        # Build prompt
        extra_prompt = req.prompt.strip()
        
        if req.style == "real":
            base_prompt = "photorealistic human face, highly detailed, real person, 8k photography style, symmetrical features"
        else:
            base_prompt = "realistic pencil face sketch, high quality"

        if extra_prompt:
            final_prompt = f"{extra_prompt}, {base_prompt}"
        else:
            final_prompt = base_prompt
            
        print(f"[Img2Img] Generating from composite. Style: '{req.style}', Prompt: '{final_prompt}'")
        
        # Generate refined sketch
        pil_img = diff_gen.generate_img2img(
            init_image=init_image,
            description=final_prompt,
            strength=0.7, # 0.7 gives a good balance of creativity vs source structure
            num_inference_steps=30,
            guidance_scale=7.5,
        )
        
        b64_img = pil_to_base64(pil_img)
        return {
            "image": f"data:image/png;base64,{b64_img}",
        }
    except Exception as e:
        print(f"[Img2Img Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match")
async def match_criminal(file: UploadFile = File(...)):
    stats["matched_count"] += 1
    enc = model_state["encoder"]
    db_embs = model_state["db_embeddings"]
    db = model_state["db"]
    
    if not enc or db_embs is None:
        raise HTTPException(status_code=500, detail="Matching system not initialized")
        
    # Read image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        query_emb = enc.get_embedding(img_tensor) # (1, 512)
        
    # Cosine Similarity
    # Normalize query
    query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
    # Normalize db
    db_norm = torch.nn.functional.normalize(db_embs, p=2, dim=1)
    
    # Dot product
    sims = torch.mm(query_norm, db_norm.t()) # (1, N)
    
    # Get Top 3
    k = min(3, len(db)) # Handle case where db has < 3 records
    top_scores, top_idxs = torch.topk(sims[0], k)
    
    results = []
    for i in range(k):
        idx = top_idxs[i].item()
        score = top_scores[i].item()
        record = db[idx].copy()
        record.pop('embedding', None)
        record.pop('attr_vector', None)
        
        # Convert sketch or photo image to base64 for reliable frontend display
        sketch_b64 = None
        target_path = record.get("sketch_path", "")
        if not target_path or not os.path.exists(target_path):
            target_path = record.get("photo_path", "")
            
        if target_path and os.path.exists(target_path):
            try:
                with Image.open(target_path).convert("RGB") as sk_img:
                    sketch_b64 = "data:image/png;base64," + pil_to_base64(sk_img)
            except Exception:
                pass
                
        results.append({
            "match": record, 
            "score": float(score),
            "sketch_image": sketch_b64
        })
    
    return {"results": results}


# --- Attribute-Based Match (primary matching method) ---
class AttributeMatchRequest(BaseModel):
    description: str

@app.post("/attribute_match")
async def attribute_match(req: AttributeMatchRequest):
    """
    Find the top-3 CUFS criminal records whose stored attributes
    best match the witness description.
    This is the correct approach: description → attributes → DB lookup.
    """
    db = model_state["db"]
    db_attr_vecs = model_state["db_attr_vectors"]

    if db is None or db_attr_vecs is None:
        raise HTTPException(status_code=500, detail="Database or attribute vectors not loaded. Re-run create_mock_db.py.")

    # Extract attributes from the description text
    query_attrs = extract_attributes_simple(req.description)
    query_vec = attrs_to_vector(query_attrs).unsqueeze(0).to(DEVICE)  # (1, 10)

    # Cosine similarity between query attribute vector and all DB attribute vectors
    query_norm = torch.nn.functional.normalize(query_vec, p=2, dim=1)
    db_norm    = torch.nn.functional.normalize(db_attr_vecs, p=2, dim=1)
    sims       = torch.mm(query_norm, db_norm.t())  # (1, N)

    k = min(3, len(db))
    top_scores, top_idxs = torch.topk(sims[0], k)

    results = []
    for i in range(k):
        idx   = top_idxs[i].item()
        score = top_scores[i].item()
        record = db[idx].copy()
        record.pop('embedding', None)
        record.pop('attr_vector', None)

        # Convert sketch image to base64 for frontend display
        sketch_b64 = None
        sketch_path = record.get("sketch_path", "")
        if sketch_path and os.path.exists(sketch_path):
            try:
                with Image.open(sketch_path).convert("RGB") as sk_img:
                    sketch_b64 = "data:image/png;base64," + pil_to_base64(sk_img)
            except Exception:
                pass

        results.append({
            "match": record,
            "score": float(score),
            "attribute_similarity_pct": round(float(score) * 100, 1),
            "sketch_image": sketch_b64,
            "matched_on": query_attrs,
        })

    return {
        "query_attributes": query_attrs,
        "results": results,
    }


@app.post("/admin/add_record")
async def add_record(
    name: str = Form(...),
    age: int = Form(...),
    crime: str = Form(...),
    sentence: str = Form(...),
    risk_level: str = Form(...),
    file: UploadFile = File(...)
):
    # Security check (mock)
    # In real app, check header Authorization: Bearer ADMIN_TOKEN
    
    # 1. Save File
    # Create custom dir if needed
    save_dir = "dataset/custom"
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{file.filename}"
    
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
        
    # 2. Compute Embedding
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model_state["encoder"].get_embedding(img_tensor)
        emb_list = emb.cpu().numpy().tolist()[0]
        
    # 3. Create Record
    new_id = f"custom_{len(model_state['db'])}"
    new_record = {
        "id": new_id,
        "name": name,
        "age": age,
        "crime": crime,
        "sentence": sentence,
        "photo_path": file_path,
        "risk_level": risk_level,
        "embedding": emb_list
    }
    
    # 4. Update In-Memory DB & File
    model_state["db"].append(new_record)
    # Re-stack embeddings
    # Concatenate new embedding to existing tensor 
    # (Checking if not None first)
    new_emb_tensor = torch.tensor([emb_list]).to(DEVICE)
    if model_state["db_embeddings"] is not None:
        model_state["db_embeddings"] = torch.cat([model_state["db_embeddings"], new_emb_tensor], dim=0)
    else:
        model_state["db_embeddings"] = new_emb_tensor
        
    # Save to JSON (Backup)
    with open(DB_PATH, 'w') as f:
        json.dump(model_state["db"], f, indent=4)
        
    return {"status": "success", "id": new_id, "message": "Record added to database."}

@app.get("/admin/stats")
async def get_stats():
    return {
        "total_records": len(model_state["db"]) if model_state["db"] else 0,
        "usage": stats
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Application running at: http://127.0.0.1:8000")
