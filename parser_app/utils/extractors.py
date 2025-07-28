import re
from dateutil.parser import parse

def extract_by_keywords(text, keywords):
    lines = text.split('\n')
    for line in lines:
        for keyword in keywords:
            if keyword.lower() in line.lower():
                clean_line = line.split(":")[-1].strip()
                return clean_line
    return ""

def extract_date(text):
    try:
        dt = parse(text, fuzzy=True, dayfirst=True)
        return dt.strftime("%d-%m-%Y")
    except:
        return ""
