from groq import Groq
import os
import re
import time
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from db_values import normalize_query, build_context_hint, CRIME_DESCRIPTIONS, CITIES

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Retry Helper ──────────────────────────────────────────────────────────────
def groq_with_retry(messages, model, temperature, max_tokens, retries=3, backoff=2):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "rate_limit" in err or "429" in err or "too many" in err or "413" in err
            is_transient  = "connection" in err or "timeout" in err or "500" in err or "503" in err
            if (is_rate_limit or is_transient) and attempt < retries - 1:
                wait = backoff * (2 ** attempt)
                print(f"[groq_retry] Attempt {attempt+1} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# ── DB Path ───────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "cctns_crime_data.db"

# ── Schema ────────────────────────────────────────────────────────────────────
SQL_SYSTEM_PROMPT = """You are an expert SQLite assistant for the CCTNS police database.

Table: crimes
Columns: report_number, date_reported, date_of_occurrence, time_of_occurrence,
         city, crime_code, crime_description, victim_age, victim_gender,
         weapon_used, crime_domain, police_deployed, case_closed, date_case_closed

STRICT RULES — follow exactly:
1. Use LOWER() only for city and crime_description comparisons.
2. victim_gender: use EXACTLY 'M', 'F', or 'X' — never 'Male'/'Female'/'male'/'female'.
3. case_closed: use EXACTLY 'Yes' or 'No' — never 'open'/'closed'/'pending'.
4. If the user's message contains RESOLVED VALUES, you MUST use those exact strings in SQL.
5. LIMIT 20 rows always (unless user asks for count/aggregation).
6. Never use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE.
7. Return ONLY the SQL inside <SQL></SQL> tags.
8. ALWAYS select only these columns (never use SELECT *):
   report_number, city, crime_description, date_reported, victim_age, victim_gender, case_closed, weapon_used

Examples:
  Q: Show VEHICLE - STOLEN cases in Chennai where victim_gender = 'F'
  <SQL>SELECT report_number, city, crime_description, date_reported, victim_age, victim_gender, case_closed, weapon_used FROM crimes WHERE LOWER(crime_description) = 'vehicle - stolen' AND LOWER(city) = 'chennai' AND victim_gender = 'F' LIMIT 20;</SQL>

  Q: Open cases in Bangalore where victim_gender = 'F' aged 15-25
  <SQL>SELECT report_number, city, crime_description, date_reported, victim_age, victim_gender, case_closed, weapon_used FROM crimes WHERE LOWER(city) = 'bangalore' AND case_closed = 'No' AND victim_gender = 'F' AND victim_age BETWEEN 15 AND 25 LIMIT 20;</SQL>

  Q: Show SHOPLIFTING cases in Mumbai where victim_gender = 'M'
  <SQL>SELECT report_number, city, crime_description, date_reported, victim_age, victim_gender, case_closed, weapon_used FROM crimes WHERE LOWER(crime_description) = 'shoplifting' AND LOWER(city) = 'mumbai' AND victim_gender = 'M' LIMIT 20;</SQL>

  Q: Show robbery cases in Delhi
  <SQL>SELECT report_number, city, crime_description, date_reported, victim_age, victim_gender, case_closed, weapon_used FROM crimes WHERE LOWER(crime_description) = 'robbery' AND LOWER(city) = 'delhi' LIMIT 20;</SQL>
"""

COMPREHENSION_PROMPT = """You are a police data assistant for CCTNS (India).
Summarize the query results in plain English for a police officer.
- Start directly with the findings. No preamble.
- Mention total count, city, crime type, open/closed status.
- Max 80 words. Be concise and factual.
"""


def generate_sql(question: str, context_hint: str) -> str:
    user_content = question
    if context_hint:
        user_content = f"{question}\n\n{context_hint}"

    resp = groq_with_retry(
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        model=GROQ_MODEL,
        temperature=0.0,   # zero temp for deterministic SQL
        max_tokens=256,
    )
    return resp.choices[0].message.content


def run_query(sql: str) -> pd.DataFrame | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[sql_query] DB error: {e}")
        return None


def comprehend(question: str, records: list) -> str:
    sample  = records[:10]
    total   = len(records)
    payload = (
        f"QUESTION: {question}\n"
        f"TOTAL RESULTS: {total}\n"
        f"SAMPLE (first {len(sample)} of {total}): {sample}"
    )
    resp = groq_with_retry(
        messages=[
            {"role": "system", "content": COMPREHENSION_PROMPT},
            {"role": "user",   "content": payload},
        ],
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=200,
    )
    return resp.choices[0].message.content


def sql_query_chain(question: str) -> dict:
    """NL → normalize → SQL → Execute → NL Response"""

    # Step 1: Normalize — resolve aliases to exact DB values
    normalized, resolved = normalize_query(question)
    context_hint = build_context_hint(resolved)
    if resolved:
        print(f"[sql_query] Resolved: {resolved}")

    # Step 2: Generate SQL using normalized query + hint
    try:
        raw = generate_sql(normalized, context_hint)
    except Exception as e:
        print(f"[sql_query] generate_sql failed: {e}")
        return {
            "answer": "⚠️ Could not generate query right now. Please try again in a moment.",
            "records": [], "sql": None, "error": str(e),
        }

    matches = re.findall(r"<SQL>(.*?)</SQL>", raw, re.DOTALL)
    if not matches:
        return {
            "answer": "❌ Couldn't generate a valid query. Please rephrase.",
            "records": [], "sql": None, "error": "no_sql_generated",
        }

    sql = matches[0].strip()

    # Safety net: if LLM still generated SELECT *, replace with exact display columns
    DISPLAY_COLS = "report_number, city, crime_description, date_reported, victim_age, victim_gender, case_closed, weapon_used"
    if re.match(r'^\s*SELECT\s+\*\s+FROM', sql, re.IGNORECASE):
        sql = re.sub(r'^\s*SELECT\s+\*\s+FROM', f'SELECT {DISPLAY_COLS} FROM', sql, flags=re.IGNORECASE)
        print(f"[sql_query] Replaced SELECT * with specific columns")

    print(f"[sql_query] SQL: {sql}")

    # Step 3: Safety check
    if re.search(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)\b', sql, re.IGNORECASE):
        return {
            "answer": "⚠️ Query blocked for security reasons.",
            "records": [], "sql": sql, "error": "unsafe_query",
        }

    # Step 4: Execute
    df = run_query(sql)
    if df is None:
        return {
            "answer": "❌ Error executing the query. Please try again.",
            "records": [], "sql": sql, "error": "execution_error",
        }

    if df.empty:
        return {
            "answer": "🔍 No records found matching your criteria.",
            "records": [], "sql": sql, "error": None,
        }

    records = df.head(20).to_dict(orient="records")

    # Step 5: Comprehend — with solid fallback if LLM returns empty
    try:
        answer = comprehend(question, records)
        # If LLM somehow returns empty, build a fallback summary
        if not answer or not answer.strip():
            raise ValueError("Empty response from comprehend")
    except Exception as e:
        print(f"[sql_query] comprehend failed or empty: {e}")
        total = result_total = len(df) if 'df' in dir() else len(records)
        answer = f"Found {total} record(s) matching your query."

    return {
        "answer":     answer,
        "records":    records,
        "sql":        sql,
        "error":      None,
        "total_rows": len(df),
    }


if __name__ == "__main__":
    tests = [
        "show me cases in chennai marked as vehicle-stolen",
        "show me cases in chennai marked as vehicle stolen",
        "show theft cases in Surat where victim is male",
        "open cases in bengaluru where victim is female aged 15 to 25",
        "murder cases in calcutta",
        "drug cases pending in delhi",
        "show vandalism cases in Chennai where victim is female",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        result = sql_query_chain(q)
        print(f"SQL: {result['sql']}")
        print(f"Rows: {result.get('total_rows', 0)}")
        print(f"Answer: {result['answer']}")
        print("-" * 60)
