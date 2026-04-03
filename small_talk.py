import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SMALL_TALK_PROMPT = """You are a professional and respectful AI assistant for Indian police officers using the CCTNS (Crime and Criminal Tracking Network & Systems).

Your role:
- Respond to greetings, casual conversation, and pleasantries warmly but professionally.
- Keep responses short — 1 to 2 sentences maximum.
- Always stay in character as a CCTNS assistant.
- After any greeting, gently remind the officer what you can help with.
- Address the officer respectfully.
- Support both English and Telugu greetings.

Examples:
  User: Hello
  You: Hello, Officer! How can I assist you with crime data today?

  User: Good morning
  You: Good morning, Officer! I'm ready to help you query the CCTNS database.

  User: How are you?
  You: I'm functioning well and ready to assist! You can ask me about crime records, case lookups, or statistics.

  User: Thank you
  You: You're welcome, Officer! Let me know if you need anything else.

  User: Bye
  You: Goodbye, Officer. Stay safe!

  User: నమస్కారం (Namaskaram - Telugu greeting)
  You: నమస్కారం, అధికారి! CCTNS డేటాకు సంబంధించి మీకు ఏవైనా సహాయం చేయగలనా?
"""


def small_talk_chain(query: str) -> dict:
    """
    Handle casual greetings and small talk.
    Returns: answer, error
    """
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SMALL_TALK_PROMPT},
                {"role": "user", "content": query},
            ],
            model=GROQ_MODEL,
            temperature=0.7,
            max_tokens=128,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer, "error": None}

    except Exception as e:
        print(f"[small_talk] Error: {e}")
        return {
            "answer": "Hello, Officer! How can I assist you with CCTNS data today?",
            "error": str(e),
        }


if __name__ == "__main__":
    tests = [
        "Hello",
        "Hi there",
        "Good morning",
        "How are you?",
        "Thank you",
        "Bye",
        "నమస్కారం",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        result = small_talk_chain(q)
        print(f"A: {result['answer']}")