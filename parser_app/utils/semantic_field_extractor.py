# parser_app/utils/semantic_field_extractor.py

import os
from openai import OpenAI
import numpy as np
import torch
import torch.nn.functional as F

from decouple import config

client = OpenAI(api_key=config("OPENAI_API_KEY"))

# Define alternate ways the fields can be written
semantic_field_map = {
    "date_of_birth": ["date of birth", "dob", "d.o.b", "birth date", "born on"],
    "residential_address": ["address", "permanent address", "current location", "residence", "home address"],
    "contact_number": ["contact number", "mobile", "phone", "telephone", "cell number"]
}

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip()
    )
    return response.data[0].embedding

def extract_semantic_field(raw_text: str, field: str) -> str:
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    if not lines:
        return ""

    line_embeddings = [get_embedding(line) for line in lines]
    target_phrases = semantic_field_map.get(field, [])
    target_embeddings = [get_embedding(phrase) for phrase in target_phrases]

    best_score = -1.0
    best_line = ""

    for target_emb in target_embeddings:
        for i, line_emb in enumerate(line_embeddings):
            score = F.cosine_similarity(
                torch.tensor(target_emb).unsqueeze(0),
                torch.tensor(line_emb).unsqueeze(0)
            ).item()

            if score > best_score:
                best_score = score
                best_line = lines[i]

    return best_line
