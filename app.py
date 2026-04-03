"""
CCTNS AI Voice Assistant — Main Flask Application
==================================================
Routes:
  POST /chat           - Main chat endpoint
  GET  /health         - Health check
  GET  /schema         - Database schema info
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from router import route
from sql_query import sql_query_chain
from analytics_query import analytics_chain
from case_lookup import case_lookup_chain
from faq import faq_chain
from small_talk import small_talk_chain
from help import help_chain

load_dotenv()

app = Flask(__name__)
CORS(app)

print("🚔 CCTNS AI Voice Assistant — Starting...")
print("=" * 60)


# ── DB Schema Info (for frontend reference) ───────────────────────────────────
DB_SCHEMA = {
    "table": "crimes",
    "columns": [
        {"name": "report_number",       "type": "INTEGER", "description": "Unique crime report ID"},
        {"name": "date_reported",       "type": "TEXT",    "description": "When the crime was reported"},
        {"name": "date_of_occurrence",  "type": "TEXT",    "description": "When the crime occurred"},
        {"name": "time_of_occurrence",  "type": "TEXT",    "description": "Time of occurrence"},
        {"name": "city",                "type": "TEXT",    "description": "City where crime occurred"},
        {"name": "crime_code",          "type": "INTEGER", "description": "Numeric crime type code"},
        {"name": "crime_description",   "type": "TEXT",    "description": "Crime type description"},
        {"name": "victim_age",          "type": "INTEGER", "description": "Age of the victim"},
        {"name": "victim_gender",       "type": "TEXT",    "description": "M or F"},
        {"name": "weapon_used",         "type": "TEXT",    "description": "Weapon involved"},
        {"name": "crime_domain",        "type": "TEXT",    "description": "Broad crime category"},
        {"name": "police_deployed",     "type": "INTEGER", "description": "Officers deployed"},
        {"name": "case_closed",         "type": "TEXT",    "description": "Yes or No"},
        {"name": "date_case_closed",    "type": "TEXT",    "description": "Date case was closed"},
    ],
    "total_records": 40160,
}


# ── Serve Frontend ───────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "CCTNS AI Voice Assistant",
        "version": "1.0.0",
        "routes_available": ["SQL_QUERY", "ANALYTICS_QUERY", "CASE_LOOKUP", "FAQ", "SMALL_TALK", "HELP"],
    })


# ── Schema Info ───────────────────────────────────────────────────────────────
@app.route("/schema", methods=["GET"])
def schema():
    return jsonify(DB_SCHEMA)


# ── Update API Key at Runtime ─────────────────────────────────────────────────
@app.route("/update-key", methods=["POST"])
def update_key():
    """
    Allows changing the Groq API key without redeploying.
    Request: { "api_key": "gsk_xxx..." }
    """
    try:
        data = request.get_json(force=True)
        new_key = data.get("api_key", "").strip()

        if not new_key:
            return jsonify({"error": "api_key is required"}), 400
        if not new_key.startswith("gsk_"):
            return jsonify({"error": "Invalid key format — Groq keys start with 'gsk_'"}), 400

        # Update the environment variable and reinitialize all Groq clients
        os.environ["GROQ_API_KEY"] = new_key

        # Reload all modules that hold a Groq client instance
        import groq
        import sql_query, analytics_query, case_lookup, faq, small_talk, help, router

        new_client = groq.Groq(api_key=new_key)
        sql_query.client       = new_client
        analytics_query.client = new_client
        case_lookup.client     = new_client
        faq.client             = new_client
        small_talk.client      = new_client
        help.client            = new_client
        router.client          = new_client

        print(f"[update-key] API key updated to: {new_key[:8]}...")
        return jsonify({"status": "ok", "message": "API key updated successfully"})

    except Exception as e:
        print(f"[update-key] Error: {e}")
        return jsonify({"error": str(e)}), 500



@app.route("/chat", methods=["POST"])
def chat():
    """
    Request body:
    {
        "message": "Show robbery cases in Bangalore",
        "user_id": "officer_001",       (optional)
        "language": "en"                (optional: en / te)
    }

    Response:
    {
        "route": "SQL_QUERY",
        "answer": "...",
        "records": [...],               (for SQL_QUERY, ANALYTICS_QUERY, CASE_LOOKUP)
        "sql": "SELECT ...",            (for SQL_QUERY, ANALYTICS_QUERY)
        "chart_type": "bar",            (for ANALYTICS_QUERY)
        "chart_data": {...},            (for ANALYTICS_QUERY)
        "error": null
    }
    """
    try:
        data = request.get_json(force=True)
        query = data.get("message", "").strip()

        if not query:
            return jsonify({"error": "message field is required"}), 400

        user_id = data.get("user_id", "anonymous")
        print(f"\n[{user_id}] Query: {query}")

        # ── Step 1: Route the query ───────────────────────────────────────────
        intent = route(query)
        print(f"[{user_id}] Route: {intent}")

        # ── Step 2: Dispatch to the correct module ────────────────────────────
        if intent == "SQL_QUERY":
            result = sql_query_chain(query)
            return jsonify({
                "route":        intent,
                "answer":       result["answer"],
                "records":      result.get("records", []),
                "sql":          result.get("sql"),
                "total_rows":   result.get("total_rows", 0),
                "error":        result.get("error"),
            })

        elif intent == "ANALYTICS_QUERY":
            result = analytics_chain(query)
            return jsonify({
                "route":        intent,
                "answer":       result["answer"],
                "summary":      result.get("summary", ""),
                "insights":     result.get("insights", []),
                "chart_type":   result.get("chart_type", "table"),
                "chart_data":   result.get("chart_data"),
                "records":      result.get("records", []),
                "sql":          result.get("sql"),
                "total_rows":   result.get("total_rows", 0),
                "error":        result.get("error"),
            })

        elif intent == "CASE_LOOKUP":
            result = case_lookup_chain(query)
            return jsonify({
                "route":          intent,
                "answer":         result["answer"],
                "record":         result.get("record"),
                "report_number":  result.get("report_number"),
                "error":          result.get("error"),
            })

        elif intent == "FAQ":
            result = faq_chain(query)
            return jsonify({
                "route":  intent,
                "answer": result["answer"],
                "error":  result.get("error"),
            })

        elif intent == "SMALL_TALK":
            result = small_talk_chain(query)
            return jsonify({
                "route":  intent,
                "answer": result["answer"],
                "error":  result.get("error"),
            })

        elif intent == "HELP":
            result = help_chain(query)
            return jsonify({
                "route":      intent,
                "answer":     result["answer"],
                "help_type":  result.get("help_type", "general"),
                "error":      result.get("error"),
            })

        else:
            return jsonify({
                "route":  "UNKNOWN",
                "answer": "I'm not sure how to help with that. Try asking about crime records or type 'help'.",
                "error":  "unrecognized_route",
            })

    except Exception as e:
        err_str = str(e)
        print(f"[app] Unhandled error: {err_str}")
        if "decommissioned" in err_str or "model_decommissioned" in err_str:
            user_msg = "⚠️ AI model decommissioned. Set GROQ_MODEL=llama-3.1-8b-instant in .env and restart."
        elif "invalid_api_key" in err_str or "authentication" in err_str.lower():
            user_msg = "⚠️ Invalid Groq API key. Check your .env file."
        elif "413" in err_str or ("rate_limit" in err_str and "token" in err_str.lower()):
            user_msg = "⚠️ Request too large for Groq free tier. Please try again — the system will retry automatically."
        elif "rate_limit" in err_str or "429" in err_str or "too many" in err_str.lower():
            user_msg = "⚠️ Rate limit reached. Please wait 10–15 seconds and try again."
        elif "Connection" in err_str or "timeout" in err_str.lower():
            user_msg = "⚠️ Could not reach the Groq API. Check your internet connection."
        else:
            user_msg = f"⚠️ Backend error: {err_str[:300]}"
        return jsonify({"error": err_str, "answer": user_msg, "route": "ERROR"}), 500


# ── Global error handlers (always JSON, never HTML 404 pages) ─────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "answer": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": str(e), "answer": "⚠️ Internal server error."}), 500


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
