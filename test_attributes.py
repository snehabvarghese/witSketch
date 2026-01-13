from llm_text_to_attributes import extract_attributes_llm

descriptions = [
    "Male with short black hair and beard",
    "A woman with oval face and glasses",
    "He had long brown hair and no beard"
]

for desc in descriptions:
    print("Description:", desc)
    print("Extracted:", extract_attributes_llm(desc))
    print("-" * 40)
