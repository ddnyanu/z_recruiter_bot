# parser_app/utils/semantic_field_extractor.py

from sentence_transformers import SentenceTransformer, util

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define alternate ways the fields can be written
semantic_field_map = {
    "date_of_birth": ["date of birth", "dob", "d.o.b", "birth date", "born on"],
    "residential_address": ["address", "permanent address", "current location", "residence", "home address"],
    "contact_number": ["contact number", "mobile", "phone", "telephone", "cell number"]
}

def extract_semantic_field(raw_text: str, field: str) -> str:
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    line_embeddings = model.encode(lines, convert_to_tensor=True)
    target_phrases = semantic_field_map.get(field, [])
    target_embedding = model.encode(target_phrases, convert_to_tensor=True)

    # Get best matching line across all target phrases
    scores = util.cos_sim(target_embedding, line_embeddings)
    max_score_idx = scores.max(dim=1).values.argmax().item()
    return lines[max_score_idx]
