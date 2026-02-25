import re

def extract_confidence(text):
    match = re.search(r"CONFIDENCE[: ]+([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None
