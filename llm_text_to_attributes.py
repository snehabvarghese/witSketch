from transformers import pipeline
import json
import re

# -----------------------------
# Global Lazy Loader
# -----------------------------
_LLM_PIPELINE = None

def get_llm():
    global _LLM_PIPELINE
    if _LLM_PIPELINE is None:
        print("Loading FLAN-T5 model...")
        _LLM_PIPELINE = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device_map="auto",
            do_sample=False
        )
    return _LLM_PIPELINE

# -----------------------------
# Default attribute values (Reference only)
# -----------------------------
DEFAULT_ATTRS = {
    "gender": "male",
    "face_shape": "round",
    "hair_length": "short",
    "hair_color": "black",
    "beard": "no",
    "glasses": "no"
}

# -----------------------------
# Helper: extract JSON from LLM output
# -----------------------------
def clean_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group()
    return None

# -----------------------------
# Helper: normalize values
# -----------------------------
def normalize(attrs):
    mapping = {
        "man": "male",
        "woman": "female",
        "yes": "yes",
        "no": "no"
    }
    for k, v in attrs.items():
        v = str(v).lower().strip()
        attrs[k] = mapping.get(v, v)
    return attrs

# -----------------------------
# Keyword fallback for safety
# -----------------------------
def keyword_fallback(description):
    attrs = {}
    desc = description.lower()

    # Gender
    if re.search(r"\b(?:woman|female)\b", desc):
        attrs["gender"] = "female"
    elif re.search(r"\b(?:man|male)\b", desc):
        attrs["gender"] = "male"

    # Face shape
    if "oval" in desc:
        attrs["face_shape"] = "oval"
    if "square" in desc:
        attrs["face_shape"] = "square"

    # Hair length
    if re.search(r"\bshort\b.*\bhair\b", desc):
        attrs["hair_length"] = "short"
    if re.search(r"\bmedium\b.*\bhair\b", desc):
        attrs["hair_length"] = "medium"
    if re.search(r"\blong\b.*\bhair\b", desc):
        attrs["hair_length"] = "long"

    # Hair color
    for color in ["black", "brown", "blonde"]:
        if color in desc:
            attrs["hair_color"] = color

    # Beard
    if "beard" in desc and "no beard" not in desc:
        attrs["beard"] = "yes"
    if "no beard" in desc:
        attrs["beard"] = "no"

    # Glasses
    if "glasses" in desc:
        attrs["glasses"] = "yes"

    return attrs

# -----------------------------
# Main function
# -----------------------------
def extract_attributes_llm(description):
    prompt = f"""
Extract facial attributes from the description. Return ONLY valid JSON.

Attributes:
- gender: male or female
- face_shape: oval, round, square
- hair_length: short, medium, long
- hair_color: black, brown, blonde
- beard: yes or no
- glasses: yes or no

Return JSON only. Example:
Input: "A man with short black hair and a beard"
Output: {{"gender": "male", "face_shape": "round", "hair_length": "short", "hair_color": "black", "beard": "yes", "glasses": "no"}}

Description:
{description}
"""

    # Step 1: LLM attempt
    llm = get_llm()
    output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    json_text = clean_json(output)

    attrs = {}
    if json_text:
        try:
            parsed = json.loads(json_text)
            attrs = normalize(parsed)
        except:
            pass

    # Step 2: Always run keyword fallback
    attrs_fallback = keyword_fallback(description)

    # Step 3: Merge LLM + fallback (Fallback overrides LLM if present, or fill missing)
    # Actually, let's trust fallback for explicitly detected keywords, but keep LLM for others.
    # Logic: Start with LLM findings, update with Keyword findings.
    
    final_attrs = attrs.copy()
    final_attrs.update(attrs_fallback)
    
    # We DO NOT fill in defaults here anymore.
    # Missing keys will be None or absent.

    return final_attrs

# -----------------------------
# Optional test block
# -----------------------------
if __name__ == "__main__":
    test_descriptions = [
        "Male with short black hair and beard",
        "A woman with oval face and glasses",
        "He had long brown hair and no beard"
    ]

    for desc in test_descriptions:
        print("Description:", desc)
        print("Extracted:", extract_attributes_llm(desc))
        print("-" * 50)
