# utils/common_group.py
import re

def normalize_group(label: str) -> str:
    """
    Clean, normalize and extract the most common plant group name.
    Examples:
      Mangifera indica   → Mango
      Musa acuminata     → Banana
      Hibiscus rosa-sinensis → Hibiscus
      Aloe vera          → Aloe
    """
    if not label:
        return "Unknown"

    label = label.lower().strip()

    # SPLIT scientific names
    parts = re.split(r"[ _\-]+", label)

    # take FIRST scientific component
    word = parts[0]

    SPECIAL = {
        "musa": "Banana",
        "mangifera": "Mango",
        "rosa": "Rose",
        "hibiscus": "Hibiscus",
        "aloe": "Aloe",
        "ocimum": "Basil",
    }

    if word in SPECIAL:
        return SPECIAL[word]

    return word.title()
