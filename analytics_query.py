from groq import Groq
import os
import re
import time
import sqlite3
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from db_values import normalize_query, build_context_hint

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DB_PATH = Path(__file__).parent / "cctns_crime_data.db"


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


ANALYTICS_SQL_PROMPT = """You are an expert SQLite analyst for the CCTNS police database.

Table: crimes
Columns: report_number, date_reported, date_of_occurrence, time_of_occurrence,
         city, crime_code, crime_description, victim_age, victim_gender (M/F/X),
         weapon_used, crime_domain, police_deployed, case_closed (Yes/No), date_case_closed

STRICT RULES:
1. Always use GROUP BY with COUNT(*), AVG(), SUM() for analytics.
2. victim_gender values are EXACTLY 'M', 'F', 'X' — never 'Male'/'Female'.
3. case_closed values are EXACTLY 'Yes' or 'No'.
4. If RESOLVED VALUES are provided, use those EXACT strings.
5. Use LOWER() for city and crime_description comparisons.
6. Limit to top 10 unless specified.
7. Never use DROP, DELETE, UPDATE, INSERT, ALTER.
8. Return ONLY SQL inside <SQL></SQL> tags.

Examples:
  Q: Which city has highest crimes by female victims (victim_gender = 'F')?
  <SQL>SELECT city, COUNT(*) as total FROM crimes WHERE victim_gender = 'F' GROUP BY city ORDER BY total DESC LIMIT 10;</SQL>

  Q: Cities where female crime rate > 30%
  <SQL>SELECT city,
    COUNT(*) as total_crimes,
    SUM(CASE WHEN victim_gender = 'F' THEN 1 ELSE 0 END) as female_crimes,
    ROUND(SUM(CASE WHEN victim_gender = 'F' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as female_percentage
  FROM crimes GROUP BY city
  HAVING female_percentage > 30
  ORDER BY female_percentage DESC;</SQL>

  Q: Top 5 crime types
  <SQL>SELECT crime_description, COUNT(*) as count FROM crimes GROUP BY crime_description ORDER BY count DESC LIMIT 5;</SQL>

  Q: Cases closed vs open
  <SQL>SELECT case_closed, COUNT(*) as count FROM crimes GROUP BY case_closed;</SQL>

  Q: Crime trend by month
  <SQL>SELECT substr(date_reported, 4, 2) as month, COUNT(*) as count FROM crimes GROUP BY month ORDER BY month;</SQL>
"""

ANALYTICS_COMPREHENSION_PROMPT = """You are a crime data analyst for Indian police (CCTNS).
Given aggregated query results, respond in this exact format:
SUMMARY: <one sentence, max 40 words>
INSIGHTS:
- <insight 1>
- <insight 2>
CHART_TYPE: <bar|pie|line|table>
Be factual and concise.
"""


def generate_analytics_sql(question: str, context_hint: str) -> str:
    user_content = question
    if context_hint:
        user_content = f"{question}\n\n{context_hint}"

    resp = groq_with_retry(
        messages=[
            {"role": "system", "content": ANALYTICS_SQL_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        model=GROQ_MODEL,
        temperature=0.0,
        max_tokens=300,
    )
    return resp.choices[0].message.content


def run_query(sql: str) -> pd.DataFrame | None:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[analytics_query] DB error: {e}")
        return None


def parse_comprehension(raw: str) -> dict:
    summary    = ""
    insights   = []
    chart_type = "table"
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("SUMMARY:"):
            summary = line.replace("SUMMARY:", "").strip()
        elif line.startswith("- "):
            insights.append(line[2:].strip())
        elif line.startswith("CHART_TYPE:"):
            chart_type = line.replace("CHART_TYPE:", "").strip().lower()
            if chart_type not in ["bar", "pie", "line", "table"]:
                chart_type = "table"
    return {"summary": summary, "insights": insights, "chart_type": chart_type}


def analytics_chain(question: str) -> dict:
    """NL → normalize → Aggregation SQL → Execute → Stats + Chart"""

    # Step 1: Normalize
    normalized, resolved = normalize_query(question)
    context_hint = build_context_hint(resolved)
    if resolved:
        print(f"[analytics_query] Resolved: {resolved}")

    # Step 2: Generate SQL
    try:
        raw = generate_analytics_sql(normalized, context_hint)
    except Exception as e:
        print(f"[analytics_query] generate_sql failed: {e}")
        return {
            "answer": "⚠️ Could not generate analytics query. Please try again.",
            "summary": "", "insights": [], "chart_type": "table",
            "records": [], "sql": None, "error": str(e),
        }

    matches = re.findall(r"<SQL>(.*?)</SQL>", raw, re.DOTALL)
    if not matches:
        return {
            "answer": "❌ Could not generate an analytics query. Please rephrase.",
            "summary": "", "insights": [], "chart_type": "table",
            "records": [], "sql": None, "error": "no_sql_generated",
        }

    sql = matches[0].strip()
    print(f"[analytics_query] SQL: {sql}")

    if re.search(r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)\b', sql, re.IGNORECASE):
        return {
            "answer": "⚠️ Query blocked for security reasons.",
            "summary": "", "insights": [], "chart_type": "table",
            "records": [], "sql": sql, "error": "unsafe_query",
        }

    # Step 3: Execute
    df = run_query(sql)
    if df is None:
        return {
            "answer": "❌ Error executing analytics query.",
            "summary": "", "insights": [], "chart_type": "table",
            "records": [], "sql": sql, "error": "execution_error",
        }

    if df.empty:
        return {
            "answer": "📊 No data found for this analytics query.",
            "summary": "", "insights": [], "chart_type": "table",
            "records": [], "sql": sql, "error": None,
        }

    records = df.to_dict(orient="records")

    # Step 4: Comprehend
    try:
        resp = groq_with_retry(
            messages=[
                {"role": "system", "content": ANALYTICS_COMPREHENSION_PROMPT},
                {"role": "user",   "content": f"QUESTION: {question}\nDATA: {records}"},
            ],
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=300,
        )
        parsed = parse_comprehension(resp.choices[0].message.content)
    except Exception as e:
        print(f"[analytics_query] comprehend failed: {e}")
        parsed = {"summary": "Analytics complete.", "insights": [], "chart_type": "table"}

    chart_data = None
    if len(df.columns) >= 2:
        label_col  = df.columns[0]
        value_col  = df.columns[1]
        chart_data = {
            "labels":    df[label_col].astype(str).tolist(),
            "values":    df[value_col].tolist(),
            "label_key": label_col,
            "value_key": value_col,
        }

    return {
        "answer":     parsed["summary"],
        "summary":    parsed["summary"],
        "insights":   parsed["insights"],
        "chart_type": parsed["chart_type"],
        "chart_data": chart_data,
        "records":    records,
        "sql":        sql,
        "error":      None,
        "total_rows": len(df),
    }


if __name__ == "__main__":
    tests = [
        "which city has more crimes that are happen by female",
        "which city has female crime rate is greater than 30%",
        "top 5 crime types",
        "how many cases are closed vs open",
        "which weapon is used the most",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        result = analytics_chain(q)
        print(f"SQL: {result['sql']}")
        print(f"Summary: {result['summary']}")
        print(f"Chart: {result['chart_type']}")
        print(f"Records: {result['records'][:3]}")
        print("-" * 60)