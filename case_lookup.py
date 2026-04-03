import os
import re
import sqlite3
import pandas as pd
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DB_PATH = Path(__file__).parent / "cctns_crime_data.db"

CASE_FORMAT_PROMPT = """You are a CCTNS police records officer assistant.
Given a case record from the database, format it into a clean, professional case summary report for a police officer.

Format as:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 CASE REPORT — #[report_number]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕒 Date Reported     : [date_reported]
📅 Date of Occurrence: [date_of_occurrence]
🏙️  City              : [city]
🔢 Crime Code        : [crime_code]
🚨 Crime Type        : [crime_description]
🎯 Crime Domain      : [crime_domain]
👤 Victim Age        : [victim_age]
⚧  Victim Gender     : [Male/Female based on M/F]
🔪 Weapon Used       : [weapon_used]
👮 Police Deployed   : [police_deployed] officers
📁 Case Status       : [Closed/Open] [date if closed]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Add a 1-2 line professional summary at the end.
"""


def extract_report_number(query: str) -> int | None:
    """Extract a report/case number from natural language query."""
    # Match patterns like: report number 1024, case 5031, #1024, no. 1024
    patterns = [
        r'(?:report\s*(?:number|no\.?|#)?\s*)(\d+)',
        r'(?:case\s*(?:number|no\.?|#)?\s*)(\d+)',
        r'(?:fir\s*(?:number|no\.?|#)?\s*)(\d+)',
        r'#\s*(\d+)',
        r'\bno\.?\s*(\d+)\b',
        r'\b(\d{4,6})\b',   # bare 4-6 digit number likely a report number
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def fetch_case(report_number: int) -> pd.DataFrame | None:
    """Fetch a case from the DB by report number."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM crimes WHERE report_number = ?",
            conn,
            params=(report_number,),
        )
        conn.close()
        return df
    except Exception as e:
        print(f"[case_lookup] DB error: {e}")
        return None


def format_case(record: dict) -> str:
    """Use LLM to format the case record into a readable report."""
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": CASE_FORMAT_PROMPT},
            {"role": "user", "content": f"Case record: {record}"},
        ],
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=512,
    )
    return resp.choices[0].message.content


def case_lookup_chain(query: str) -> dict:
    """
    Full pipeline: Extract report number → DB Lookup → Format Report
    Returns: answer, record, report_number, error
    """
    report_number = extract_report_number(query)

    if report_number is None:
        return {
            "answer": (
                "🔍 I couldn't find a report number in your query.\n"
                "Please mention the report number clearly, e.g.:\n"
                "  • 'Show report number 1024'\n"
                "  • 'Check case 5031'\n"
                "  • 'Lookup FIR #2048'"
            ),
            "record": None,
            "report_number": None,
            "error": "no_report_number",
        }

    df = fetch_case(report_number)

    if df is None:
        return {
            "answer": "❌ Database error while looking up the case. Please try again.",
            "record": None,
            "report_number": report_number,
            "error": "db_error",
        }

    if df.empty:
        return {
            "answer": f"🔍 No case found with report number **{report_number}**. Please verify the number.",
            "record": None,
            "report_number": report_number,
            "error": "not_found",
        }

    record = df.iloc[0].to_dict()
    formatted = format_case(record)

    return {
        "answer": formatted,
        "record": record,
        "report_number": report_number,
        "error": None,
    }


if __name__ == "__main__":
    tests = [
        "Show report number 1024",
        "Check case 5031",
        "Lookup FIR #200",
        "Show me case number 9999",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        result = case_lookup_chain(q)
        print(f"Report #: {result['report_number']}")
        print(f"Answer:\n{result['answer']}")