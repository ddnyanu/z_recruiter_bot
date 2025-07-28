from openai import OpenAI
from decouple import config
import numpy as np
import torch
import torch.nn.functional as F

client = OpenAI(api_key=config("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return torch.tensor(response.data[0].embedding)

def cosine_similarity(a, b):
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return torch.dot(a, b).item()

semantic_field_map = {
    "date_of_birth": ["date of birth", "dob", "d.o.b", "birth date", "born on"],
    "residential_address": ["address", "permanent address", "current location", "residence", "home address"],
    "contact_number": ["contact number", "mobile", "phone", "telephone", "cell number"]
}

def extract_semantic_field(raw_text: str, field: str) -> str:
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    line_embeddings = [get_embedding(line) for line in lines]

    target_phrases = semantic_field_map.get(field, [])
    target_embeddings = [get_embedding(phrase) for phrase in target_phrases]

    max_score = -1
    best_line = ""

    for line, line_emb in zip(lines, line_embeddings):
        for target_emb in target_embeddings:
            score = cosine_similarity(line_emb, target_emb)
            if score > max_score:
                max_score = score
                best_line = line

    return best_line
