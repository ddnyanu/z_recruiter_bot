def truncate_text(text, max_chars=3000):
    if not isinstance(text, str):
        text = str(text or "")
    return text[:max_chars]
