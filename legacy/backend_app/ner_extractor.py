"""
DevGen Framework — NER Extraction Pipeline
Extracts structured fields from recognized Devanagari text using rule-based
and transformer-based Named Entity Recognition.

Target fields for government documents:
  - Names (Nepali personal names)
  - Dates (BS/AD formats)
  - ID Numbers (citizenship no., land plot no.)
  - Locations (districts, provinces, municipalities)
  - Roll/Certificate Numbers
"""

import re
from typing import Optional


# ── Regex patterns for Nepali administrative document fields ──────────────

# Nepali date formats:  २०७८/०३/१५  or  2078/03/15  or  2078-03-15
_NEPALI_DIGIT = "[\u0966-\u096f]"  # ०-९
_ASCII_DIGIT = r"\d"
_DIGITS = f"(?:{_NEPALI_DIGIT}|{_ASCII_DIGIT})"

DATE_PATTERN = re.compile(
    rf"(?:{_DIGITS}{{4}}[\/\-\.]{_DIGITS}{{1,2}}[\/\-\.]{_DIGITS}{{1,2}})",
)

# Citizenship numbers: e.g.  12-34-56-78901  or  123456789
CITIZENSHIP_PATTERN = re.compile(
    r"\b(?:\d{2}-\d{2}-\d{2}-\d{5}|\d{8,12})\b"
)

# Land plot (Kittanumber):  किtta नं.  followed by digits
KILLA_PATTERN = re.compile(
    r"(?:कित्ता|killa|plot|kittanumber)[^\d]*(\d+)",
    re.IGNORECASE
)

# Sheet number (Paana):  पाना नं.  or  sheet no.
PAANA_PATTERN = re.compile(
    r"(?:पाना|paana|sheet)[^\d]*(\d+)",
    re.IGNORECASE
)

# Districts of Nepal (common Romanized forms + Devanagari)
DISTRICTS = [
    "काठमाडौं", "ललितपुर", "भक्तपुर", "मकवानपुर", "रसुवा",
    "नुवाकोट", "धादिङ", "चितवन", "गोरखा", "तनहुँ",
    "Kathmandu", "Lalitpur", "Bhaktapur", "Makwanpur",
    "Chitwan", "Gorkha", "Tanahu", "Kaski", "Pokhara",
]
DISTRICT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in DISTRICTS) + r")\b",
    re.IGNORECASE
)

# Province numbers
PROVINCE_PATTERN = re.compile(
    r"(?:प्रदेश|province)\s*(?:नं\.?\s*)?([१२३४५६७८९0-9]+)",
    re.IGNORECASE
)

# Ward number
WARD_PATTERN = re.compile(
    r"(?:वडा|ward)\s*(?:नं\.?\s*)?(\d+)",
    re.IGNORECASE
)


def extract_entities(text: str) -> dict:
    """Extract named entities and structured fields from recognized text.

    Args:
        text: Raw recognized Devanagari/Unicode text from TrOCR

    Returns:
        dict of entity types → list of found values
    """
    entities = {
        "dates": [],
        "citizenship_numbers": [],
        "killa_numbers": [],
        "paana_numbers": [],
        "districts": [],
        "provinces": [],
        "wards": [],
        "raw_numbers": [],
    }

    # Dates
    entities["dates"] = list(set(DATE_PATTERN.findall(text)))

    # Citizenship numbers
    entities["citizenship_numbers"] = list(set(CITIZENSHIP_PATTERN.findall(text)))

    # Killa (plot) numbers
    entities["killa_numbers"] = [m.group(1) for m in KILLA_PATTERN.finditer(text)]

    # Paana (sheet) numbers
    entities["paana_numbers"] = [m.group(1) for m in PAANA_PATTERN.finditer(text)]

    # Districts
    entities["districts"] = list(set(DISTRICT_PATTERN.findall(text)))

    # Province numbers
    entities["provinces"] = [m.group(1) for m in PROVINCE_PATTERN.finditer(text)]

    # Ward numbers
    entities["wards"] = [m.group(1) for m in WARD_PATTERN.finditer(text)]

    # All standalone numbers (catch-all for unidentified IDs)
    all_nums = re.findall(r"\b\d{4,}\b", text)
    known = (
        entities["citizenship_numbers"]
        + entities["killa_numbers"]
        + entities["paana_numbers"]
        + entities["wards"]
    )
    entities["raw_numbers"] = [n for n in all_nums if n not in known]

    # Compute a simple confidence score based on how many fields were found
    found_count = sum(len(v) for v in entities.values())
    entities["_extraction_confidence"] = min(1.0, found_count / 5)
    entities["_field_count"] = found_count

    return entities


def summarize_entities(entities: dict) -> str:
    """Return a human-readable summary of extracted entities."""
    lines = []
    if entities["dates"]:
        lines.append(f"📅 Dates: {', '.join(entities['dates'])}")
    if entities["citizenship_numbers"]:
        lines.append(f"🪪 Citizenship Nos: {', '.join(entities['citizenship_numbers'])}")
    if entities["killa_numbers"]:
        lines.append(f"📋 Killa (Plot) Nos: {', '.join(entities['killa_numbers'])}")
    if entities["paana_numbers"]:
        lines.append(f"📄 Paana (Sheet) Nos: {', '.join(entities['paana_numbers'])}")
    if entities["districts"]:
        lines.append(f"📍 Districts: {', '.join(entities['districts'])}")
    if entities["provinces"]:
        lines.append(f"🗺️ Provinces: {', '.join(entities['provinces'])}")
    if entities["wards"]:
        lines.append(f"🏘️ Wards: {', '.join(entities['wards'])}")
    if entities["raw_numbers"]:
        lines.append(f"🔢 Other IDs: {', '.join(entities['raw_numbers'])}")
    return "\n".join(lines) if lines else "No structured entities found."
