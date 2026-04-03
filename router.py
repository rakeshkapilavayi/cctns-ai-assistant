import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Route Names ───────────────────────────────────────────────────────────────
ROUTES = {
    "SQL_QUERY":       "Retrieve crime records from the database",
    "ANALYTICS_QUERY": "Aggregations, statistics, trends, charts",
    "CASE_LOOKUP":     "Look up a specific case by report/FIR number",
    "FAQ":             "Questions about the system, terminology, crime codes",
    "SMALL_TALK":      "Greetings, casual conversation, pleasantries",
    "HELP":            "Guide on how to use the system",
}

ROUTER_PROMPT = """You are an intent classifier for a CCTNS (Crime and Criminal Tracking Network & Systems) police AI assistant.

Classify the user query into EXACTLY ONE of these routes:

SQL_QUERY       - Retrieve specific crime records from the database
                  Examples: "Show robbery cases in Bangalore", "List crimes involving knife",
                            "Show open cases in Mumbai", "Crimes reported yesterday",
                            "Show female victims of theft"

ANALYTICS_QUERY - Aggregations, statistics, counts, trends, rankings, charts
                  Examples: "Which city has highest crimes?", "Top 5 crime types",
                            "How many cases are closed?", "Crime trend by month",
                            "Average victim age", "Crime distribution"

CASE_LOOKUP     - Look up a specific case by report number, FIR number, or case number
                  Examples: "Show report number 1024", "Check case 5031",
                            "Lookup FIR #2048", "Case no. 789"

FAQ             - Questions about the CCTNS system, terminology, fields, crime codes
                  Examples: "What is a crime code?", "What does case closed mean?",
                            "What is CCTNS?", "What is FIR?", "What is crime domain?"

SMALL_TALK      - Greetings, casual conversation, thanks, farewell
                  Examples: "Hello", "Hi", "Good morning", "How are you?",
                            "Thank you", "Bye", "నమస్కారం"

HELP            - Questions about system capabilities, how to use it
                  Examples: "What can you do?", "Help", "How do I query?",
                            "What can I ask?", "How does this work?"

Rules:
- Return ONLY the route name (e.g., SQL_QUERY). Nothing else.
- If a number like "1024" or "case 5031" appears with keywords like report/case/FIR → CASE_LOOKUP
- If asking for counts, trends, rankings, "how many", "which city has most" → ANALYTICS_QUERY
- If asking to "show", "list", "find" crimes with filters → SQL_QUERY
- If asking about system terminology or meaning → FAQ
"""


def route(query: str) -> str:
    """
    Classify the query into one of 6 routes.
    Returns: one of SQL_QUERY, ANALYTICS_QUERY, CASE_LOOKUP, FAQ, SMALL_TALK, HELP
    """
    # Fast rule-based overrides before calling LLM (saves latency)

    query_lower = query.lower().strip()

    # CASE_LOOKUP: presence of report/case/fir + number
    if re.search(r'\b(report|case|fir)\b.*\d{3,}', query_lower) or \
       re.search(r'#\s*\d{3,}', query_lower):
        return "CASE_LOOKUP"

    # SMALL_TALK: very short greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening",
                 "good afternoon", "bye", "goodbye", "thank you", "thanks",
                 "నమస్కారం", "నమస్తే", "ధన్యవాదాలు"]
    if query_lower.strip() in greetings or query_lower.strip() in [g + "!" for g in greetings]:
        return "SMALL_TALK"

    # HELP: very simple help keywords
    if query_lower.strip() in ["help", "?"]:
        return "HELP"

    # LLM classification for everything else
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=16,
        )
        result = resp.choices[0].message.content.strip().upper()

        # Validate result
        valid_routes = list(ROUTES.keys())
        for r in valid_routes:
            if r in result:
                return r

        # Fallback
        return "SQL_QUERY"

    except Exception as e:
        print(f"[router] Error: {e}")
        return "SQL_QUERY"


if __name__ == "__main__":
    test_cases = [
        # SQL_QUERY
        ("Show robbery cases in Bangalore",              "SQL_QUERY"),
        ("List crimes involving knife in Delhi",         "SQL_QUERY"),
        ("Show open cases in Mumbai",                    "SQL_QUERY"),
        ("List all female victims of domestic violence", "SQL_QUERY"),
        # ANALYTICS_QUERY
        ("Which city has highest number of crimes?",     "ANALYTICS_QUERY"),
        ("Top 5 most common crime types",                "ANALYTICS_QUERY"),
        ("How many cases are closed vs open?",           "ANALYTICS_QUERY"),
        ("Crime trend by month",                         "ANALYTICS_QUERY"),
        # CASE_LOOKUP
        ("Show report number 1024",                      "CASE_LOOKUP"),
        ("Check case 5031",                              "CASE_LOOKUP"),
        ("Lookup FIR #2048",                             "CASE_LOOKUP"),
        # FAQ
        ("What is a crime code?",                        "FAQ"),
        ("What does case closed mean?",                  "FAQ"),
        ("What is CCTNS?",                               "FAQ"),
        # SMALL_TALK
        ("Hello",                                        "SMALL_TALK"),
        ("Good morning",                                 "SMALL_TALK"),
        ("Thank you",                                    "SMALL_TALK"),
        # HELP
        ("What can you do?",                             "HELP"),
        ("How do I query crimes?",                       "HELP"),
        ("Help",                                         "HELP"),
    ]

    print("=" * 65)
    print(f"{'QUERY':<45} {'EXPECTED':<18} {'RESULT':<18}")
    print("=" * 65)

    correct = 0
    for query, expected in test_cases:
        result = route(query)
        match = "✅" if result == expected else "❌"
        if result == expected:
            correct += 1
        print(f"{match} {query[:43]:<45} {expected:<18} {result:<18}")

    print("=" * 65)
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct//len(test_cases)}%)")