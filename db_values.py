"""
db_values.py
============
Single source of truth for ALL exact values that exist in the crimes table.
Used by sql_query.py and analytics_query.py to normalize user input
BEFORE sending to the LLM — so the LLM never has to guess spellings.
"""

# ── Exact values from the database ───────────────────────────────────────────

CITIES = [
    "Agra", "Ahmedabad", "Bangalore", "Bhopal", "Chennai", "Delhi",
    "Faridabad", "Ghaziabad", "Hyderabad", "Indore", "Jaipur", "Kalyan",
    "Kanpur", "Kolkata", "Lucknow", "Ludhiana", "Meerut", "Mumbai",
    "Nagpur", "Nashik", "Patna", "Pune", "Rajkot", "Srinagar", "Surat",
    "Thane", "Varanasi", "Vasai", "Visakhapatnam",
]

CRIME_DESCRIPTIONS = [
    "ARSON", "ASSAULT", "BURGLARY", "COUNTERFEITING", "CYBERCRIME",
    "DOMESTIC VIOLENCE", "DRUG OFFENSE", "EXTORTION", "FIREARM OFFENSE",
    "FRAUD", "HOMICIDE", "IDENTITY THEFT", "ILLEGAL POSSESSION",
    "KIDNAPPING", "PUBLIC INTOXICATION", "ROBBERY", "SEXUAL ASSAULT",
    "SHOPLIFTING", "TRAFFIC VIOLATION", "VANDALISM", "VEHICLE - STOLEN",
]

WEAPONS = ["Blunt Object", "Explosives", "Firearm", "Knife", "Other", "Poison"]

CRIME_DOMAINS = ["Fire Accident", "Other Crime", "Traffic Fatality", "Violent Crime"]

GENDERS = {"M", "F", "X"}

CASE_CLOSED = {"Yes", "No"}

# ── Alias maps: what users type → exact DB value ─────────────────────────────

CITY_ALIASES = {
    "bengaluru":         "Bangalore",
    "bengalore":         "Bangalore",
    "banglore":          "Bangalore",
    "bombay":            "Mumbai",
    "calcutta":          "Kolkata",
    "vizag":             "Visakhapatnam",
    "visakha":           "Visakhapatnam",
    "vishakhapatnam":    "Visakhapatnam",
    "vizakhapatnam":     "Visakhapatnam",
    "new delhi":         "Delhi",
    "hydrabad":          "Hyderabad",
    "hyd":               "Hyderabad",
}

CRIME_ALIASES = {
    # theft variants → best match
    "theft":                  "SHOPLIFTING",   # generic theft → shoplifting
    "stealing":               "SHOPLIFTING",
    "vehicle theft":          "VEHICLE - STOLEN",
    "vehicle-stolen":         "VEHICLE - STOLEN",
    "vehicle stolen":         "VEHICLE - STOLEN",
    "car theft":              "VEHICLE - STOLEN",
    "car stolen":             "VEHICLE - STOLEN",
    "bike theft":             "VEHICLE - STOLEN",
    "bike stolen":            "VEHICLE - STOLEN",
    "auto theft":             "VEHICLE - STOLEN",
    "shop theft":             "SHOPLIFTING",
    "shoptheft":              "SHOPLIFTING",
    # other common aliases
    "murder":                 "HOMICIDE",
    "killing":                "HOMICIDE",
    "rape":                   "SEXUAL ASSAULT",
    "molestation":            "SEXUAL ASSAULT",
    "eve teasing":            "SEXUAL ASSAULT",
    "abduction":              "KIDNAPPING",
    "gun":                    "FIREARM OFFENSE",
    "firearm":                "FIREARM OFFENSE",
    "gunshot":                "FIREARM OFFENSE",
    "drugs":                  "DRUG OFFENSE",
    "drug":                   "DRUG OFFENSE",
    "narcotics":              "DRUG OFFENSE",
    "fire":                   "ARSON",
    "burning":                "ARSON",
    "drunk driving":          "TRAFFIC VIOLATION",
    "dui":                    "TRAFFIC VIOLATION",
    "dwi":                    "TRAFFIC VIOLATION",
    "hit and run":            "TRAFFIC VIOLATION",
    "road accident":          "TRAFFIC VIOLATION",
    "cyber":                  "CYBERCRIME",
    "hacking":                "CYBERCRIME",
    "online fraud":           "CYBERCRIME",
    "extort":                 "EXTORTION",
    "blackmail":              "EXTORTION",
    "counterfeit":            "COUNTERFEITING",
    "fake currency":          "COUNTERFEITING",
    "graffiti":               "VANDALISM",
    "destruction of property":"VANDALISM",
    "intoxication":           "PUBLIC INTOXICATION",
    "drunk":                  "PUBLIC INTOXICATION",
    "illegal weapon":         "ILLEGAL POSSESSION",
    "illegal arms":           "ILLEGAL POSSESSION",
}

GENDER_ALIASES = {
    "female": "F",
    "woman":  "F",
    "women":  "F",
    "girl":   "F",
    "lady":   "F",
    "ladies": "F",
    "male":   "M",
    "man":    "M",
    "men":    "M",
    "boy":    "M",
    "other":  "X",
    "x":      "X",
}

CASE_STATUS_ALIASES = {
    "open":         "No",
    "pending":      "No",
    "on hold":      "No",
    "left on hold": "No",
    "unsolved":     "No",
    "active":       "No",
    "not closed":   "No",
    "unresolved":   "No",
    "ongoing":      "No",
    "closed":       "Yes",
    "solved":       "Yes",
    "resolved":     "Yes",
    "completed":    "Yes",
    "done":         "Yes",
}

WEAPON_ALIASES = {
    "gun":          "Firearm",
    "pistol":       "Firearm",
    "rifle":        "Firearm",
    "revolver":     "Firearm",
    "blade":        "Knife",
    "dagger":       "Knife",
    "sword":        "Knife",
    "stick":        "Blunt Object",
    "rod":          "Blunt Object",
    "bat":          "Blunt Object",
    "hammer":       "Blunt Object",
    "stone":        "Blunt Object",
    "rock":         "Blunt Object",
    "bomb":         "Explosives",
    "explosive":    "Explosives",
    "grenade":      "Explosives",
    "chemical":     "Poison",
    "acid":         "Poison",
}


def normalize_query(user_query: str) -> tuple[str, dict]:
    """
    Scan user query for known aliases and replace them with exact DB values.
    Returns:
      - normalized_query: string with aliases replaced
      - resolved: dict of what was resolved, for logging/debugging
    """
    import re
    normalized = user_query
    resolved   = {}

    q_lower = user_query.lower()

    # --- Resolve cities ---
    for alias, canonical in CITY_ALIASES.items():
        if alias in q_lower:
            # Replace case-insensitively in the original query
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            normalized = pattern.sub(canonical, normalized)
            resolved[f"city_alias:{alias}"] = canonical

    # --- Resolve crime descriptions (longest match first to avoid partial hits) ---
    for alias in sorted(CRIME_ALIASES.keys(), key=len, reverse=True):
        if alias in q_lower:
            canonical = CRIME_ALIASES[alias]
            pattern = re.compile(re.escape(alias), re.IGNORECASE)
            normalized = pattern.sub(canonical, normalized)
            resolved[f"crime_alias:{alias}"] = canonical
            break  # only replace first/best match

    # --- Resolve gender ---
    for alias, canonical in GENDER_ALIASES.items():
        # Use word boundary to avoid "female" matching inside other words
        pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
        if pattern.search(normalized):
            normalized = pattern.sub(f"gender={canonical}", normalized)
            resolved[f"gender_alias:{alias}"] = canonical

    # --- Resolve case status ---
    for alias in sorted(CASE_STATUS_ALIASES.keys(), key=len, reverse=True):
        if alias in q_lower:
            canonical = CASE_STATUS_ALIASES[alias]
            resolved[f"case_status:{alias}"] = canonical
            break

    # --- Resolve weapons ---
    for alias, canonical in WEAPON_ALIASES.items():
        pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
        if pattern.search(normalized):
            normalized = pattern.sub(canonical, normalized)
            resolved[f"weapon_alias:{alias}"] = canonical

    return normalized, resolved


def build_context_hint(resolved: dict) -> str:
    """Build a short hint string to append to the LLM prompt so it uses exact values."""
    if not resolved:
        return ""
    lines = ["RESOLVED VALUES (use these EXACT strings in SQL):"]
    for key, val in resolved.items():
        kind = key.split(":")[0]
        original = key.split(":")[1]
        if kind == "city_alias":
            lines.append(f'  city = \'{val}\'  (user said "{original}")')
        elif kind == "crime_alias":
            lines.append(f'  crime_description = \'{val}\'  (user said "{original}")')
        elif kind == "gender_alias":
            lines.append(f'  victim_gender = \'{val}\'  (user said "{original}")')
        elif kind == "case_status":
            lines.append(f'  case_closed = \'{val}\'  (user said "{original}")')
        elif kind == "weapon_alias":
            lines.append(f'  weapon_used = \'{val}\'  (user said "{original}")')
    return "\n".join(lines)


if __name__ == "__main__":
    tests = [
        "show me vehicle stolen cases in bengaluru where female victim",
        "show theft cases in bombay where victim is male",
        "open cases in vizag where victim is a girl",
        "murder cases in calcutta",
        "drug cases that are pending in delhi",
        "show me cases in chennai marked as vehicle-stolen",
        "which city has more crimes that are happen by female",
    ]
    for q in tests:
        norm, res = normalize_query(q)
        hint = build_context_hint(res)
        print(f"\nOriginal : {q}")
        print(f"Normalized: {norm}")
        print(f"Resolved  : {res}")
        if hint:
            print(f"Hint:\n{hint}")
        print("-" * 60)
