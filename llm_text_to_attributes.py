from transformers import pipeline
import json
import re

# -----------------------------
# Initialize LLM
# -----------------------------
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # can use flan-t5-large if enough GPU
    device_map="auto",
    do_sample=False               # deterministic output
)

# -----------------------------
# Default attribute values
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
    attrs = DEFAULT_ATTRS.copy()
    desc = description.lower()

    # Gender
    # Gender (use word boundaries to avoid matching "man" inside "woman")
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
    # Hair length (allow adjectives between the length word and 'hair', e.g. 'long brown hair')
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
    output = llm(prompt, max_new_tokens=200)[0]["generated_text"]
    json_text = clean_json(output)

    attrs = None
    if json_text:
        try:
            attrs = json.loads(json_text)
            attrs = normalize(attrs)
        except:
            attrs = None

    # Step 2: Always run keyword fallback
    attrs_fallback = keyword_fallback(description)

    # Step 3: Merge LLM + fallback
    if attrs:
        for key in DEFAULT_ATTRS:
            # Replace default or missing values with keyword fallback
            if key not in attrs or attrs[key] == DEFAULT_ATTRS[key]:
                attrs[key] = attrs_fallback[key]
    else:
        attrs = attrs_fallback

    return attrs

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
