import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Static FAQ Knowledge Base ─────────────────────────────────────────────────
CCTNS_FAQ_KNOWLEDGE = """
CCTNS SYSTEM KNOWLEDGE BASE
============================

1. What is CCTNS?
CCTNS (Crime and Criminal Tracking Network & Systems) is a digital system launched by the Government of India under the National Crime Records Bureau (NCRB). It connects all police stations across India to a centralized database for recording, tracking, and managing crime and criminal data.

2. What is a Crime Code?
A crime code is a standardized numerical identifier used to classify and categorize specific types of crimes. For example, code 576 may refer to Identity Theft, code 128 to Homicide. Crime codes help in uniformly recording and retrieving crime data across police stations.

3. What does 'Case Closed' mean?
A case is marked as 'Closed' (Yes) when the investigation is complete — either the accused has been arrested and chargesheeted, or the case has been resolved. 'Open' (No) means the case is still under active investigation.

4. What is Crime Domain?
Crime Domain is a broad classification of crimes into categories:
  - Violent Crime: Assault, Homicide, Robbery, Kidnapping
  - Other Crime: Fraud, Identity Theft, Cybercrime
  - Fire Accident: Arson-related incidents
  - Traffic Fatality: Road accidents with fatalities

5. What is Date Reported vs Date of Occurrence?
  - Date Reported: When the FIR or complaint was filed at the police station.
  - Date of Occurrence: The actual date and time the crime took place.
  These may differ if a crime is reported after the fact.

6. What is a Report Number?
A report number (also called FIR number or case number) is a unique identifier assigned to each crime report. You can use it to look up a specific case.

7. What does Police Deployed mean?
Police Deployed indicates the number of police personnel assigned/deployed to handle or investigate that particular case.

8. What is Victim Gender in the database?
  - M = Male
  - F = Female

9. What cities are covered in this database?
The database includes major Indian cities such as: Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Ahmedabad, Kolkata, Pune, Surat, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Bhopal, Patna, Ludhiana, Agra, and more.

10. What crimes are recorded?
Common crime types include: Robbery, Theft, Homicide, Kidnapping, Fraud, Burglary, Vandalism, Domestic Violence, Firearm Offense, Identity Theft, Arson, Hit and Run, DUI/DWI, Drug Offense, Sexual Assault, and others.

11. How do I query the system?
You can ask questions like:
  - "Show robbery cases in Delhi"
  - "Which city has the highest theft cases?"
  - "Show case number 1024"
  - "List all open cases in Mumbai"
  - "Show crimes involving knife last month"

12. What is FIR?
FIR (First Information Report) is a written document prepared by the police when they receive information about the commission of a cognizable offense. It is the first step in the criminal justice process.

13. What is charge sheet?
A charge sheet (also called a police report) is a formal document filed by police in court after completing the investigation, listing the accused, charges, and evidence.

14. What are cognizable offenses?
Cognizable offenses are crimes where police can arrest without a warrant (e.g., murder, robbery, kidnapping). Non-cognizable offenses require a court warrant for arrest.
"""

FAQ_SYSTEM_PROMPT = f"""You are a knowledgeable assistant for police officers using the CCTNS (Crime and Criminal Tracking Network & Systems) in India.

Use the knowledge base below to answer questions about CCTNS, crime terminology, database fields, and system usage.
If the answer is in the knowledge base, use it directly.
If not found, answer based on your knowledge of Indian law enforcement systems.
Keep responses concise, professional, and easy to understand for police personnel.

KNOWLEDGE BASE:
{CCTNS_FAQ_KNOWLEDGE}
"""


def faq_chain(query: str) -> dict:
    """
    Answer FAQ questions about CCTNS system, crime codes, terminology.
    Returns: answer, error
    """
    try:
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": FAQ_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=512,
        )
        answer = resp.choices[0].message.content
        return {"answer": answer, "error": None}

    except Exception as e:
        print(f"[faq] Error: {e}")
        return {
            "answer": "❌ Sorry, I couldn't retrieve the answer right now. Please try again.",
            "error": str(e),
        }


if __name__ == "__main__":
    tests = [
        "What is CCTNS?",
        "What is a crime code?",
        "What does case closed mean?",
        "What is the difference between date reported and date of occurrence?",
        "What is FIR?",
        "What cities are in the database?",
    ]
    for q in tests:
        print(f"\nQ: {q}")
        result = faq_chain(q)
        print(f"A: {result['answer']}")
        print("-" * 60)