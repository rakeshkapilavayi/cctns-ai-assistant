import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Static Help Content ───────────────────────────────────────────────────────
HELP_CONTENT = {
    "general": """
🤖 **CCTNS AI Voice Assistant — Help Guide**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

I am your AI-powered assistant for querying the CCTNS (Crime and Criminal Tracking Network & Systems) database using natural language or voice commands.

**What I can do:**

📌 **1. Crime Data Queries (SQL)**
   Ask questions to retrieve crime records:
   • "Show robbery cases in Bangalore"
   • "List crimes reported yesterday"
   • "Show crimes involving knife in Delhi"
   • "Show open cases in Mumbai"
   • "List female victims of domestic violence"

📊 **2. Analytics & Statistics**
   Ask for summaries and trends:
   • "Which city has the highest number of crimes?"
   • "Top 5 most common crime types"
   • "How many cases are closed vs open?"
   • "Crime breakdown by domain"
   • "Average victim age per crime type"

🔍 **3. Case Lookup**
   Look up a specific case by report number:
   • "Show report number 1024"
   • "Check case 5031"
   • "Lookup FIR #2048"

❓ **4. FAQ**
   Ask questions about the system or terminology:
   • "What is a crime code?"
   • "What does case closed mean?"
   • "What is CCTNS?"
   • "What is an FIR?"

💬 **5. Greetings**
   Let’s keep it casual and chat normally instead..

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Tip: You can use voice or text input in English or Telugu.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
    "sql": """
📌 **How to Query Crime Records**

You can ask questions like:
• "Show robbery cases in Bangalore"
• "List theft cases in Delhi last month"
• "Show crimes involving firearms"
• "List all open homicide cases"
• "Show crimes where victim is female"
• "Cases reported in Chennai in 2023"

**Tips:**
- Mention the city name for location-specific results
- Mention the crime type (robbery, theft, homicide, etc.)
- Mention weapons if relevant (knife, firearm, etc.)
- Ask for open/closed cases to filter by status
""",
    "analytics": """
📊 **How to Get Statistics & Analytics**

You can ask:
• "Which city has the highest number of crimes?"
• "Top 5 most common crime types"
• "How many cases are closed vs open?"
• "Crime distribution by domain"
• "Monthly crime trend for 2023"
• "Average victim age by crime type"
• "Which weapon is most commonly used?"

Results will include charts and visualizations!
""",
    "case_lookup": """
🔍 **How to Look Up a Specific Case**

Mention the report/case number:
• "Show report number 1024"
• "Check case 5031"
• "Lookup FIR #2048"
• "Show case no. 789"

You'll get a complete formatted case summary.
""",
}

HELP_CONTEXTUAL_PROMPT = f"""You are a helpful guide for police officers using the CCTNS AI Voice Assistant.
Based on the user's help request, provide the most relevant guidance from the help content below.
Be concise, professional, and helpful. Format using bullet points where appropriate.

HELP CONTENT:
{HELP_CONTENT['general']}

Additional context:
{HELP_CONTENT['sql']}
{HELP_CONTENT['analytics']}
{HELP_CONTENT['case_lookup']}
"""


def help_chain(query: str) -> dict:
    """
    Guide the officer on how to use the system.
    Returns: answer, help_type, error
    """
    query_lower = query.lower()

    # Detect specific help topic
    help_type = "general"
    if any(kw in query_lower for kw in ["crime query", "search crime", "find crime", "how to query", "sql"]):
        help_type = "sql"
    elif any(kw in query_lower for kw in ["statistic", "analytic", "chart", "trend", "report"]):
        help_type = "analytics"
    elif any(kw in query_lower for kw in ["case number", "report number", "lookup", "specific case"]):
        help_type = "case_lookup"

    # For very simple help requests, return static content directly (fast)
    simple_triggers = ["what can you do", "what can i ask", "help", "how to use", "guide", "capabilities"]
    if any(trigger in query_lower for trigger in simple_triggers):
        return {
            "answer": HELP_CONTENT["general"],
            "help_type": help_type,
            "error": None,
            "static": True,
        }

    # For more specific questions, use LLM
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": HELP_CONTEXTUAL_PROMPT},
                {"role": "user", "content": query},
            ],
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer, "help_type": help_type, "error": None, "static": False}

    except Exception as e:
        print(f"[help] Error: {e}")
        return {
            "answer": HELP_CONTENT["general"],
            "help_type": "general",
            "error": str(e),
            "static": True,
        }


if __name__ == "__main__":
    tests = [
        "What can you do?",
        "Help",
        "How do I query crimes?",
        "How do I look up a specific case?",
        "What kind of statistics can I get?",
        "I don't know how to use this system",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        result = help_chain(q)
        print(f"Help Type: {result['help_type']}")
        print(f"A: {result['answer']}")
        print("-" * 60)