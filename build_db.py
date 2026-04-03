"""
build_db.py
===========
Run this once to create cctns_crime_data.db from crime_dataset_india.csv.
Render runs this as part of the build command so the DB is always present.
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path

CSV_PATH = Path(__file__).parent / "crime_dataset_india.csv"
DB_PATH  = Path(__file__).parent / "cctns_crime_data.db"


def build():
    if not CSV_PATH.exists():
        print(f"[build_db] ERROR: CSV not found at {CSV_PATH}")
        return False

    # If DB already exists and is newer than CSV, skip rebuild
    if DB_PATH.exists():
        if DB_PATH.stat().st_mtime >= CSV_PATH.stat().st_mtime:
            print(f"[build_db] DB already up to date ({DB_PATH}), skipping rebuild.")
            return True

    print(f"[build_db] Building DB from {CSV_PATH} ...")

    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

    # Rename columns to snake_case to match app expectations
    df.columns = [
        'report_number', 'date_reported', 'date_of_occurrence', 'time_of_occurrence',
        'city', 'crime_code', 'crime_description', 'victim_age', 'victim_gender',
        'weapon_used', 'crime_domain', 'police_deployed', 'case_closed', 'date_case_closed'
    ]

    conn = sqlite3.connect(DB_PATH)
    df.to_sql('crimes', conn, if_exists='replace', index=False)

    # Add indexes for the most common query patterns
    conn.execute("CREATE INDEX IF NOT EXISTS idx_city ON crimes (city);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crime_desc ON crimes (crime_description);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_case_closed ON crimes (case_closed);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_victim_gender ON crimes (victim_gender);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_report_number ON crimes (report_number);")
    conn.commit()
    conn.close()

    size_mb = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"[build_db] Done! {len(df)} rows, {size_mb:.2f} MB → {DB_PATH}")
    return True


if __name__ == "__main__":
    success = build()
    if not success:
        exit(1)
