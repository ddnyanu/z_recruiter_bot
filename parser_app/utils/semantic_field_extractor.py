# parser_app/utils/semantic_field_extractor.py

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

# Load lightweight transformer model
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

# Define semantic mappings for fallback fields
semantic_field_map = {
    "date_of_birth": ["date of birth", "dob", "d.o.b", "birth date", "born on"],
    "residential_address": ["address", "permanent address", "current location", "residence", "home address"],
    "contact_number": ["contact number", "mobile", "phone", "telephone", "cell number"]
}

def get_embedding(text):
    """Generate a normalized mean pooled embedding for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(embeddings, p=2, dim=1)

def extract_semantic_field(raw_text: str, field: str) -> str:
    """Extract the best matching line for a given field using semantic similarity."""
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    if not lines or field not in semantic_field_map:
        return ""

    line_embeddings = torch.cat([get_embedding(line) for line in lines], dim=0)

    best_line = ""
    highest_score = -1.0

    for target in semantic_field_map[field]:
        target_embedding = get_embedding(target)
        # Compute cosine similarity with all lines
        similarities = cosine_similarity(target_embedding, line_embeddings).squeeze(0)

        max_idx = similarities.argmax().item()
        max_score = similarities[max_idx].item()

        if max_score > highest_score:
            highest_score = max_score
            best_line = lines[max_idx]

    return best_line
