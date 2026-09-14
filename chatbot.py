import streamlit as st
import os
import re
import json
import inspect
from datetime import datetime, timedelta
from supabase import create_client
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from deep_translator import GoogleTranslator

# ------------------------------------------------
# Config
# ------------------------------------------------
st.set_page_config(page_title="Career AI", page_icon="💼", layout="centered")
os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = "false"
os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = "false"
st.title("💼 Career Intelligence Chatbot")
st.caption("RAG + Query Rewriting + Tool Calling + Risk Metrics + Hindi Support")

# ------------------------------------------------
# Credentials
# ------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY or not SUPABASE_KEY or not SUPABASE_URL:
    st.error("❌ Missing credentials. Add to Streamlit Secrets.")
    st.stop()

# ------------------------------------------------
# Supabase + LLM clients (must be created BEFORE auth check)
# ------------------------------------------------
# Add this temporarily to check if the keys are loading
if not SUPABASE_KEY:
    st.error("DEBUG: SUPABASE_KEY is empty! Environment variables are not loading.")
    st.stop()
elif len(SUPABASE_KEY) < 20:
    st.error(f"DEBUG: SUPABASE_KEY is too short, it might be corrupted: {SUPABASE_KEY}")
    st.stop()

# Then your client initialization
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-120b")

# ------------------------------------------------
# Auth: get token from URL params and verify user
# ------------------------------------------------
# If the user is already authenticated in this Streamlit session,
# don't require the ticket again on every rerun.
if "user_id" in st.session_state:
    current_user_id = st.session_state["user_id"]

else:
    # First load: get ticket from URL
    ticket_id = st.query_params.get("ticket")

    if not ticket_id:
        st.warning(
            "⌛ No auth ticket received. Please close and reopen the chatbot panel."
        )
        st.stop()

    # Fetch the real token from the database
    ticket_response = (
        supabase
        .table("auth_tickets")
        .select("access_token")
        .eq("id", ticket_id)
        .execute()
    )

    if not ticket_response.data:
        st.error(
            "❌ Invalid or expired session ticket. Please log out and back in."
        )
        st.stop()

    auth_token = ticket_response.data[0]["access_token"]

    # Burn the ticket immediately so it cannot be reused
    supabase.table("auth_tickets").delete().eq("id", ticket_id).execute()

    # Verify the token
    try:
        user_response = supabase.auth.get_user(auth_token)
        current_user_id = user_response.user.id

    except Exception:
        st.error("❌ Invalid or expired session. Please log in again.")
        st.stop()

    # Store authenticated user in Streamlit session
    st.session_state["user_id"] = current_user_id

    # Remove ticket from URL after successful authentication
    if "ticket" in st.query_params:
        del st.query_params["ticket"]

# ------------------------------------------------
# Chat History Persistence
# ------------------------------------------------
def load_chat_history(user_id: str, limit: int = 50):
    try:
        res = supabase.table("chat_history") \
            .select("role, message, created_at") \
            .eq("user_id", user_id) \
            .order("created_at", desc=False) \
            .limit(limit) \
            .execute()
        return [{"role": row["role"], "content": row["message"]} for row in (res.data or [])]
    except Exception:
        return []

def save_chat_message(user_id: str, role: str, message: str):
    try:
        supabase.table("chat_history").insert({
            "user_id": user_id,
            "role": role,
            "message": message
        }).execute()
    except Exception:
        pass  # Silently fail — don't break chat for a persistence error

# ------------------------------------------------
# TABLE SCHEMA
# ------------------------------------------------
# Single source table now — new_jobs_data replaces both the old
# naukri_jobs and jobs tables. Note: no industry/sector column exists
# here (unlike the old `jobs` table), and jobUploaded is relative text
# ("3 Days Ago") rather than a real date — days_ago (a generated
# integer column added via migration) is what filtering/sorting
# actually uses.
TABLE_SCHEMA = """
TABLE: new_jobs_data
Columns: jobId, title, companyName, companyId, location, experience,
         minimumExperience, maximumExperience, salary, minimumSalary,
         maximumSalary, currency, tagsAndSkills, jobDescription,
         jobUploaded (relative text, NOT sortable — use days_ago),
         days_ago (generated integer, 0 = most recent posting),
         ReviewsCount, AggregateRating

Note: pass the city name as the user said it (e.g. "Bangalore" or
"Bengaluru") — the tool functions already check known old/current
name variants (Bangalore/Bengaluru, Bombay/Mumbai, Madras/Chennai,
Calcutta/Kolkata, Gurgaon/Gurugram, etc.) automatically.
"""

# ------------------------------------------------
# SALARY FORMATTING
# ------------------------------------------------
# minimumSalary/maximumSalary are stored as TEXT, as raw rupee amounts
# (not LPA), and "0" means "not disclosed" — but "0" is a non-empty
# string (truthy), so a naive check would misreport it as a real
# ₹0–0 LPA range. Mirrors the same fix applied on the frontend.
def to_lakhs(rupees: float) -> str:
    lakhs = rupees / 100000
    if lakhs == int(lakhs):
        return str(int(lakhs))
    return f"{lakhs:.2f}".rstrip('0').rstrip('.')

def format_salary(min_raw, max_raw, currency=None) -> str:
    try:
        min_val = float(min_raw)
        max_val = float(max_raw)
    except (TypeError, ValueError):
        return "Not disclosed"
    if min_val <= 0 or max_val <= 0:
        return "Not disclosed"
    cur = currency or "₹"
    return f"{cur}{to_lakhs(min_val)}–{to_lakhs(max_val)} LPA"

# ------------------------------------------------
# CITY ALIASES
# ------------------------------------------------
# Several Indian cities have an old anglicized name and a current
# official one, and job listings mix both inconsistently. A plain
# ilike("location", "%bangalore%") finds zero rows if the data only
# says "Bengaluru" — they don't share that substring at all, even
# though they're the same city. RAG search doesn't have this problem
# (it matches on meaning, not literal substrings), which is why RAG
# found jobs a tool-calling query for the same city missed entirely.
CITY_ALIASES = {
    "bangalore": ["bangalore", "bengaluru"],
    "bengaluru": ["bangalore", "bengaluru"],
    "bombay": ["bombay", "mumbai"],
    "mumbai": ["bombay", "mumbai"],
    "madras": ["madras", "chennai"],
    "chennai": ["madras", "chennai"],
    "calcutta": ["calcutta", "kolkata"],
    "kolkata": ["calcutta", "kolkata"],
    "gurgaon": ["gurgaon", "gurugram"],
    "gurugram": ["gurgaon", "gurugram"],
    "cochin": ["cochin", "kochi"],
    "kochi": ["cochin", "kochi"],
    "trivandrum": ["trivandrum", "thiruvananthapuram"],
    "thiruvananthapuram": ["trivandrum", "thiruvananthapuram"],
    "pondicherry": ["pondicherry", "puducherry"],
    "puducherry": ["pondicherry", "puducherry"],
}

def apply_city_filter(query, city: str):
    """Filters by location, checking all known name variants of the
    given city so an old/new naming mismatch doesn't silently return
    zero results for a city that genuinely has listings."""
    if not city:
        return query
    variants = CITY_ALIASES.get(city.strip().lower(), [city])
    or_expr = ",".join(f"location.ilike.%{v}%" for v in variants)
    return query.or_(or_expr)

# ------------------------------------------------
# TRANSLATION
# ------------------------------------------------
def translate_to_english(text):
    try:
        if re.search("[\u0900-\u097F]", text):
            return GoogleTranslator(source="hi", target="en").translate(text)
        return text
    except Exception:
        try:
            r = llm.invoke(f"Translate to English. Return ONLY translation:\n{text}")
            return r.content.strip()
        except Exception:
            return text

# ================================================
# RAG SECTION
# ================================================

@st.cache_resource
def load_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def embeddings_exist():
    try:
        res = supabase.table("job_embeddings").select("jobId").limit(1).execute()
        return len(res.data) > 0
    except Exception:
        return False

def get_vector_store():
    """
    Only CHECKS whether embeddings are ready — does not build them.
    Deliberately NOT cached: this used to wrap the actual (heavy)
    embedding-building work, where caching made sense. Now it's just
    two cheap count() queries, and caching it with @st.cache_resource
    meant the result from whenever the server process first started
    (e.g. before any embeddings existed) would stick around until the
    whole process restarted — a browser reload alone wouldn't clear it.
    """
    try:
        total_res = supabase.table("new_jobs_data").select("jobId", count="exact").execute()
        total_jobs = total_res.count or 0

        saved_res = supabase.table("job_embeddings").select("jobId", count="exact").execute()
        total_saved = saved_res.count or 0

        if total_saved >= total_jobs and total_saved > 0:
            return True, total_saved, total_jobs, "ready"
        elif total_saved > 0:
            # RAG works over this subset — could be partial because the
            # build script hasn't finished, or a deliberate cap (e.g.
            # free-tier storage limits), so this isn't necessarily
            # something to "fix".
            return True, total_saved, total_jobs, "partial"
        else:
            return None, 0, total_jobs, "not_built"

    except Exception as e:
        return None, str(e), 0, "exception"

def rag_search(query, vector_store=None, k=5):
    try:
        embedding_model = load_embedding_model()
        query_vector = embedding_model.embed_query(query)

        res = supabase.rpc("match_jobs", {
            "query_embedding": query_vector,
            "match_count": k
        }).execute()

        return [
            Document(
                page_content=row["page_content"],
                metadata={
                    "title": row.get("title", ""),
                    "companyName": row.get("companyName", ""),
                    "location": row.get("location", ""),
                    "tagsAndSkills": row.get("tagsAndSkills", ""),
                    "experience": row.get("experience", ""),
                    "salary_display": row.get("salary_display", ""),
                    "rating": row.get("rating", "")
                }
            )
            for row in (res.data or [])
        ]
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# ------------------------------------------------
# QUERY REWRITER
# ------------------------------------------------
def rewrite_query(question, chat_history):
    if not chat_history or len(chat_history) < 2:
        return question

    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content'][:300]}\n"

    rewrite_prompt = f"""You are a search query optimizer for a job search engine.

CONVERSATION HISTORY:
{history_str}

CURRENT USER MESSAGE: {question}

Rewrite the current message into a STANDALONE search query for a job database.

Rules:
- If already clear and standalone, return as-is
- Replace pronouns like "that", "it", "those" with actual context from history
- Keep concise — 5-15 words max
- Focus on: role, skills, location, company
- Return ONLY the rewritten query, nothing else

Examples:
"Tell me more about that" → "Data Scientist jobs at Accenture Bengaluru"
"What about Mumbai?" → "Python Developer jobs in Mumbai"
"Show me similar ones" → "Machine Learning Engineer jobs Bangalore"
"What skills do I need?" → "Required skills for Data Analyst role"
"How much do they pay?" → "Data Scientist salary"
"""
    try:
        response = llm.invoke(rewrite_prompt)
        rewritten = response.content.strip()
        if len(rewritten) > 150 or "\n" in rewritten:
            return question
        return rewritten
    except Exception:
        return question

# ------------------------------------------------
# RAG ANSWER — compact card format
# ------------------------------------------------
def rag_answer(question, vector_store, chat_history, is_hindi=False):
    rewritten_query = rewrite_query(question, chat_history)
    relevant_docs = rag_search(rewritten_query, vector_store)

    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs]) \
        if relevant_docs else "No relevant jobs found."

    history_str = ""
    for msg in (chat_history or [])[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content'][:300]}\n"

    language = "IMPORTANT: Respond entirely in Hindi." if is_hindi else "Respond in English."

    prompt = f"""You are an expert career advisor with access to real Indian job market data.

CONVERSATION HISTORY:
{history_str}

ORIGINAL USER QUESTION: {question}
SEARCH QUERY USED: {rewritten_query}

RELEVANT JOB DATA (via semantic search):
{context}

Instructions:
- Format EACH job as a compact 3-line card exactly like this:

**1. Job Title** — Company Name
📍 Location | ⏳ Experience | 💰 Salary
🛠️ `skill1` `skill2` `skill3` `skill4`

**2. Job Title** — Company Name
📍 Location | ⏳ Experience | 💰 Salary
🛠️ `skill1` `skill2` `skill3`

Rules for cards:
- Strictly 3 lines per card, no more
- Keep location short — city names only, max 2-3 cities
- If salary is missing or "Not disclosed" → write "Not disclosed", don't invent a number
- Skills in backticks like tags — max 4-5 skills per card
- Separate each card with a blank line
- After ALL cards write ONE short summary line (max 20 words)
- End with ONE short follow-up question (max 15 words)
- NO long paragraphs anywhere
- {language}
"""
    try:
        response = llm.invoke(prompt)
        return response.content, relevant_docs, rewritten_query
    except Exception as e:
        return f"Error: {str(e)}", [], question

# ================================================
# RISK METRICS SECTION
# ================================================

def fetch_market_data_for_risk(role):
    data = {}
    try:
        base_q = supabase.table("new_jobs_data").select("*", count="exact")
        if role:
            base_q = base_q.ilike("title", f"%{role}%")
        data["total_jobs"] = base_q.execute().count or 0

        skills_q = supabase.table("new_jobs_data").select("tagsAndSkills")
        if role:
            skills_q = skills_q.ilike("title", f"%{role}%")
        skills_res = skills_q.limit(100).execute()
        freq = {}
        for row in skills_res.data or []:
            for s in (row.get("tagsAndSkills") or "").split(","):
                s = s.strip().lower()
                if s:
                    freq[s] = freq.get(s, 0) + 1
        data["top_skills"] = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]

        # No absolute postdate exists here — jobUploaded is relative
        # text, parsed into the days_ago integer column. Recent vs
        # previous period uses that instead of a calendar cutoff.
        # If jobUploaded turns out to have an unconfirmed cap for
        # older postings (similar to what showed up in another table),
        # the prev-period count could be inflated right at that
        # boundary — worth spot-checking once real numbers come in.
        recent_q = supabase.table("new_jobs_data").select("*", count="exact").lte("days_ago", 90)
        if role:
            recent_q = recent_q.ilike("title", f"%{role}%")
        data["recent_3m"] = recent_q.execute().count or 0

        prev_q = supabase.table("new_jobs_data").select("*", count="exact").gt("days_ago", 90).lte("days_ago", 180)
        if role:
            prev_q = prev_q.ilike("title", f"%{role}%")
        data["prev_3m"] = prev_q.execute().count or 0

        companies_q = supabase.table("new_jobs_data").select("companyName")
        if role:
            companies_q = companies_q.ilike("title", f"%{role}%")
        companies_res = companies_q.limit(100).execute()
        companies = list({r.get("companyName", "") for r in companies_res.data or [] if r.get("companyName")})
        data["unique_companies"] = len(companies)
        data["top_companies"] = companies[:10]

    except Exception as e:
        data["db_error"] = str(e)
    return data

def calculate_risk_metrics(role, job_description, market_data):
    skills_from_db = [s[0] for s in market_data.get("top_skills", [])]
    total_jobs = market_data.get("total_jobs", 0)
    recent_3m = market_data.get("recent_3m", 0)
    prev_3m = market_data.get("prev_3m", 0)
    unique_companies = market_data.get("unique_companies", 0)

    trend_str = f"{((recent_3m - prev_3m) / prev_3m * 100):+.1f}% change" \
        if prev_3m > 0 else "Insufficient historical data"

    risk_prompt = f"""You are an expert AI labor market analyst.

JOB ROLE: {role}
JOB DESCRIPTION: {job_description or "Not provided"}

REAL MARKET DATA:
- Total active listings: {total_jobs}
- Jobs last 3 months: {recent_3m}
- Jobs previous 3 months: {prev_3m}
- Trend: {trend_str}
- Unique companies hiring: {unique_companies}
- Top skills: {', '.join(skills_from_db[:10]) if skills_from_db else 'N/A'}

Calculate 4 risk metrics. Respond ONLY with JSON:
{{
  "task_automation_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "key_automatable_tasks": ["task1", "task2", "task3"],
    "explanation": "<2-3 sentences>"
  }},
  "ai_replacement_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "timeline": "<estimated timeline>",
    "explanation": "<2-3 sentences>"
  }},
  "market_saturation_risk": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "market_signal": "<growing/stable/declining>",
    "explanation": "<2-3 sentences>"
  }},
  "overall_risk_score": {{
    "score": <0-100>,
    "label": "<LOW/MEDIUM/HIGH/VERY HIGH>",
    "verdict": "<one line summary>",
    "top_recommendations": ["rec1", "rec2", "rec3"]
  }}
}}"""

    try:
        response = llm.invoke(risk_prompt)
        content = re.sub(r"```json|```", "", response.content.strip()).strip()
        return json.loads(content)
    except Exception as e:
        return {"error": str(e)}

def display_risk_metrics(metrics, role, market_data, is_hindi=False):
    if "error" in metrics:
        return f"Error calculating risk: {metrics['error']}"

    def score_color(s): return "🔴" if s >= 70 else "🟡" if s >= 40 else "🟢"
    def score_bar(s): return "█" * int(s/10) + "░" * (10 - int(s/10))

    tar = metrics.get("task_automation_risk", {})
    air = metrics.get("ai_replacement_risk", {})
    msr = metrics.get("market_saturation_risk", {})
    ovr = metrics.get("overall_risk_score", {})

    output = f"""## 🎯 Risk Analysis — **{role}**

---
### 1️⃣ Task Automation Risk
{score_color(tar.get('score',0))} **{tar.get('score',0)}/100** — {tar.get('label','')}
`{score_bar(tar.get('score',0))}` {tar.get('score',0)}%
{tar.get('explanation','')}
**Tasks at risk:** {' • '.join(tar.get('key_automatable_tasks',[]))}

---
### 2️⃣ AI Replacement Risk
{score_color(air.get('score',0))} **{air.get('score',0)}/100** — {air.get('label','')}
`{score_bar(air.get('score',0))}` {air.get('score',0)}%
{air.get('explanation','')}
⏱️ **Timeline:** {air.get('timeline','N/A')}

---
### 3️⃣ Market Saturation Risk
{score_color(msr.get('score',0))} **{msr.get('score',0)}/100** — {msr.get('label','')}
`{score_bar(msr.get('score',0))}` {msr.get('score',0)}%
{msr.get('explanation','')}
📈 **Signal:** {msr.get('market_signal','N/A')}

---
### 🏆 Overall Risk Score
{score_color(ovr.get('score',0))} **{ovr.get('score',0)}/100** — {ovr.get('label','')}
`{score_bar(ovr.get('score',0))}` {ovr.get('score',0)}%
**{ovr.get('verdict','')}**

💡 **Recommendations:**
{chr(10).join([f"• {r}" for r in ovr.get('top_recommendations',[])])}

---
📊 *{market_data.get('total_jobs',0)} listings | {market_data.get('unique_companies',0)} companies hiring*
"""

    if is_hindi:
        try:
            translated = llm.invoke(f"Translate to Hindi. Keep numbers, scores, emojis:\n{output}")
            return translated.content
        except Exception:
            return output
    return output

def handle_risk_conversation(user_message, is_hindi):
    if "risk_session" not in st.session_state:
        st.session_state.risk_session = {"active": False, "role": None, "job_description": None, "step": None}

    rs = st.session_state.risk_session
    risk_keywords = ["risk", "automation risk", "ai risk", "replacement risk",
                     "saturation", "risk score", "job risk", "career risk", "जोखिम", "खतरा"]
    is_risk_request = any(kw in user_message.lower() for kw in risk_keywords)

    if is_risk_request and not rs["active"]:
        st.session_state.risk_session = {"active": True, "role": None, "job_description": None, "step": "ask_role"}
        if is_hindi:
            return "बिल्कुल! 📊 अपना **जॉब रोल** बताएं (जैसे: Data Analyst, Developer):"
        return "Sure! 📊 Please tell me your **Job Role / Designation**:"

    if rs["active"] and rs["step"] == "ask_role":
        st.session_state.risk_session["role"] = translate_to_english(user_message)
        st.session_state.risk_session["step"] = "ask_description"
        if is_hindi:
            return f"रोल: **{user_message}** ✅\n\n**Job Description** पेस्ट करें:\n*(नहीं है तो 'skip' लिखें)*"
        return f"Role: **{user_message}** ✅\n\nPaste your **Job Description**:\n*(Type 'skip' if you don't have one)*"

    if rs["active"] and rs["step"] == "ask_description":
        jd = user_message if user_message.lower() not in ["skip", "no", "नहीं", "छोड़ो"] else ""
        st.session_state.risk_session["job_description"] = translate_to_english(jd)
        role = st.session_state.risk_session["role"]
        job_description = st.session_state.risk_session["job_description"]
        st.session_state.risk_session = {"active": False, "role": None, "job_description": None, "step": None}

        with st.spinner(f"📊 Fetching market data for '{role}'..."):
            market_data = fetch_market_data_for_risk(role)
        with st.spinner("🧠 Calculating risk metrics..."):
            metrics = calculate_risk_metrics(role, job_description, market_data)

        return display_risk_metrics(metrics, role, market_data, is_hindi)

    return None

# ================================================
# TOOL CALLING SECTION
# ================================================
# Consolidated to a single set of tools against new_jobs_data — the
# old naukri_jobs/jobs split no longer applies. industry_breakdown is
# gone entirely: there's no industry/sector column in this schema, so
# rather than leave a tool that silently returns nothing meaningful,
# it's removed. Add it back if you wire in a real industry data source.

TOOLS = {
    "count_jobs": {"description": "Count jobs matching role/city/recency", "params": ["role", "city", "months"]},
    "list_jobs": {"description": "List jobs for a role/city, with salary info", "params": ["role", "city", "months", "limit"]},
    "top_skills": {"description": "Get most in-demand skills for a role", "params": ["role"]},
    "salary_insights": {"description": "Show salary info for a role/city", "params": ["role", "city"]},
    "recent_jobs": {"description": "Get jobs posted in the last N months", "params": ["months", "role"]},
    "company_jobs": {"description": "List jobs from a specific company", "params": ["company", "role"]},
    "run_custom_sql": {"description": "Run custom PostgreSQL SELECT query", "params": ["sql"]},
    "general_advice": {"description": "Answer career questions using LLM knowledge", "params": ["question"]}
}

def count_jobs(role="", city="", months=None):
    try:
        q = supabase.table("new_jobs_data").select("*", count="exact")
        if role: q = q.ilike("title", f"%{role}%")
        if city: q = apply_city_filter(q, city)
        if months: q = q.lte("days_ago", 30 * int(months))
        return {"count": q.execute().count or 0}
    except Exception as e:
        return {"error": str(e)}

def list_jobs(role="", city="", months=None, limit=8):
    try:
        q = supabase.table("new_jobs_data").select(
            "title, companyName, location, tagsAndSkills, experience, jobUploaded, "
            "minimumSalary, maximumSalary, currency"
        )
        if role: q = q.ilike("title", f"%{role}%")
        if city: q = apply_city_filter(q, city)
        if months: q = q.lte("days_ago", 30 * int(months))
        rows = q.order("days_ago", desc=False).limit(int(limit)).execute().data or []
        for r in rows:
            r["salary_display"] = format_salary(r.get("minimumSalary"), r.get("maximumSalary"), r.get("currency"))
        return {"jobs": rows}
    except Exception as e:
        return {"error": str(e)}

def top_skills(role=""):
    try:
        q = supabase.table("new_jobs_data").select("tagsAndSkills")
        if role: q = q.ilike("title", f"%{role}%")
        res = q.limit(150).execute()
        freq = {}
        for row in res.data or []:
            for s in (row.get("tagsAndSkills") or "").split(","):
                s = s.strip()
                if s: freq[s] = freq.get(s, 0) + 1
        return {"skills": sorted(freq.items(), key=lambda x: x[1], reverse=True)[:12]}
    except Exception as e:
        return {"error": str(e)}

def salary_insights(role="", city=""):
    try:
        q = supabase.table("new_jobs_data").select(
            "title, companyName, location, minimumSalary, maximumSalary, currency"
        )
        if role: q = q.ilike("title", f"%{role}%")
        if city: q = apply_city_filter(q, city)
        res = q.limit(30).execute()
        rows = []
        for r in res.data or []:
            display = format_salary(r.get("minimumSalary"), r.get("maximumSalary"), r.get("currency"))
            if display != "Not disclosed":  # don't clutter results with undisclosed rows
                r["salary_display"] = display
                rows.append(r)
        return {"salary_data": rows}
    except Exception as e:
        return {"error": str(e)}

def recent_jobs(months=3, role=""):
    try:
        q = supabase.table("new_jobs_data").select(
            "title, companyName, location, jobUploaded, tagsAndSkills"
        ).lte("days_ago", 30 * int(months)).order("days_ago", desc=False)
        if role: q = q.ilike("title", f"%{role}%")
        return {"jobs": q.limit(15).execute().data or [], "months": months}
    except Exception as e:
        return {"error": str(e)}

def company_jobs(company="", role=""):
    try:
        q = supabase.table("new_jobs_data").select(
            "title, companyName, location, tagsAndSkills, experience, jobUploaded, "
            "minimumSalary, maximumSalary, currency"
        )
        if company: q = q.ilike("companyName", f"%{company}%")
        if role: q = q.ilike("title", f"%{role}%")
        rows = q.order("days_ago", desc=False).limit(10).execute().data or []
        for r in rows:
            r["salary_display"] = format_salary(r.get("minimumSalary"), r.get("maximumSalary"), r.get("currency"))
        return {"jobs": rows}
    except Exception as e:
        return {"error": str(e)}

def run_custom_sql(sql=""):
    try:
        if not sql.strip().lower().startswith("select"):
            return {"error": "Only SELECT allowed."}
        return {"results": supabase.rpc("run_query", {"query": sql}).execute().data or [], "sql_used": sql}
    except Exception as e:
        return {"error": str(e)}

def general_advice(question=""):
    try:
        response = llm.invoke(f"You are an expert Indian job market career advisor. Answer concisely:\n{question}")
        return {"answer": response.content}
    except Exception as e:
        return {"error": str(e)}

TOOL_FUNCTIONS = {
    "count_jobs": count_jobs,
    "list_jobs": list_jobs,
    "top_skills": top_skills,
    "salary_insights": salary_insights,
    "recent_jobs": recent_jobs,
    "company_jobs": company_jobs,
    "run_custom_sql": run_custom_sql,
    "general_advice": general_advice,
}

def should_use_rag(question):
    rag_signals = [
        "similar to", "match", "profile", "find jobs for me",
        "suitable", "recommend", "based on my skills",
        "like me", "fit", "relevant jobs", "jobs for someone who",
        "career change", "transition", "what jobs can i do",
        "i know", "i have experience", "i like working",
        "tell me more", "more about", "show me similar",
        "what about", "how about", "other options",
        "मेरे लिए", "मुझे कौन सी", "मेरे skills", "मेरी profile",
        "और बताओ", "इसके बारे में"
    ]
    return any(signal in question.lower() for signal in rag_signals)

def plan_tool_call(user_question, chat_history):
    history_str = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content'][:200]}\n"

    tools_desc = "\n".join([
        f"- {name}: {info['description']} | params: {info['params']}"
        for name, info in TOOLS.items()
    ])

    plan_prompt = f"""You are a tool-calling AI for an Indian job market database.

AVAILABLE TOOLS:
{tools_desc}

DATABASE SCHEMA:
{TABLE_SCHEMA}

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

Rules:
1. Use the structured tools (count_jobs, list_jobs, etc.) for anything they cover
2. Use run_custom_sql only for something the structured tools can't express
3. Use general_advice for career knowledge unrelated to this specific dataset
4. Pass limit/months as integers always

Respond ONLY with JSON array:
[{{"tool": "name", "params": {{}}, "reason": "why"}}]
"""
    try:
        response = llm.invoke(plan_prompt)
        content = re.sub(r"```json|```", "", response.content.strip()).strip()
        tool_calls = json.loads(content)
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        return tool_calls
    except Exception:
        return [{"tool": "general_advice", "params": {"question": user_question}, "reason": "fallback"}]

def execute_tool_calls(tool_calls):
    results = []
    for call in tool_calls:
        tool_name = call.get("tool")
        params = call.get("params", {})
        reason = call.get("reason", "")
        if tool_name in TOOL_FUNCTIONS:
            func = TOOL_FUNCTIONS[tool_name]
            valid_params = inspect.signature(func).parameters.keys()
            filtered = {k: v for k, v in params.items() if k in valid_params}
            data = func(**filtered)
        else:
            data = {"error": f"Unknown tool: {tool_name}"}
        results.append({"tool": tool_name, "reason": reason, "data": data})
    return results

def generate_final_answer(user_question, tool_results, chat_history, is_hindi=False):
    results_str = ""
    for r in tool_results:
        results_str += f"\n[Tool: {r['tool']} | Reason: {r['reason']}]\n"
        results_str += json.dumps(r["data"], indent=2, default=str)

    history_str = ""
    for msg in (chat_history or [])[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content'][:300]}\n"

    language = "IMPORTANT: Respond entirely in Hindi." if is_hindi else "Respond in English."

    answer_prompt = f"""You are an expert career advisor with real Indian job market data.

CONVERSATION HISTORY:
{history_str}

USER QUESTION: {user_question}

DATA FROM DATABASE:
{results_str}

Instructions:
- Use real data for specific, accurate answers
- Highlight numbers, companies, skills, trends
- If data empty, answer from general knowledge
- Be conversational and concise
- Use bullet points where helpful
- {language}
"""
    try:
        response = llm.invoke(answer_prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# ================================================
# BUILD RAG INDEX ON STARTUP
# ================================================
with st.spinner("⏳ Checking RAG index..."):
    vector_store, result, total_jobs, status = get_vector_store()

if status == "ready":
    st.success(f"✅ RAG ready — {result} jobs indexed.")
elif status == "partial":
    st.info(f"ℹ️ RAG active on {result:,} of {total_jobs:,} listings (a subset, not the full "
            f"table — e.g. due to storage limits). Semantic search still works, just over that subset.")
elif status == "not_built":
    st.error("❌ No embeddings found yet. Run `python build_job_embeddings.py` "
             "separately, then reload this app.")
else:
    st.error(f"❌ Could not check RAG status: {result}")

# ================================================
# DEBUG PANEL
# ================================================
with st.expander("🔧 Debug Panel"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Test new_jobs_data"):
            try:
                res = supabase.table("new_jobs_data").select("*").limit(2).execute()
                st.success(f"✅ {len(res.data)} rows")
                if res.data: st.dataframe(res.data)
            except Exception as e:
                st.error(f"❌ {e}")
    with col2:
        if st.button("Test job_embeddings"):
            try:
                res = supabase.table("job_embeddings") \
                    .select("jobId, title, companyName, location").limit(5).execute()
                st.success(f"✅ {len(res.data)} embeddings")
                if res.data: st.dataframe(res.data)
            except Exception as e:
                st.error(f"❌ {e}")
    with col3:
        if st.button("Test RAG Search"):
            if vector_store:
                docs = rag_search("python developer bangalore", vector_store)
                st.success(f"✅ {len(docs)} jobs found")
                for d in docs[:2]:
                    st.text(d.page_content[:200])
                    st.divider()
            else:
                st.error("❌ RAG not ready")

    if st.button("🔄 Clear RAG index"):
        try:
            supabase.table("job_embeddings").delete().neq("jobId", -1).execute()
            get_vector_store.clear()
            st.success("✅ Cleared! Run `python build_job_embeddings.py` again to rebuild, then reload this app.")
        except Exception as e:
            st.error(f"❌ {e}")

# ================================================
# CHAT UI
# ================================================
if "messages" not in st.session_state:
    saved_messages = load_chat_history(st.session_state["user_id"])
    greeting = {
        "role": "assistant",
        "content": (
            "👋 Hi! I'm your Career Intelligence Assistant.\n\n"
            "I can help you with:\n"
            "- 🔍 **Semantic search** — *Find jobs matching my Python and ML skills*\n"
            "- 💬 **Follow-ups** — *What about Mumbai? Tell me more about that*\n"
            "- 📊 **Queries** — *How many jobs in Delhi?*\n"
            "- 🎯 **Risk score** — *Calculate my AI risk score*\n"
            "- 💰 **Salary** — *Show salaries for data analysts*\n"
            "- 🏢 **Companies** — *Jobs at TCS or Infosys*\n"
            "- 🇮🇳 **Hindi** — *मुझे नौकरी ढूंढने में मदद करो* 🙏\n\n"
            f"{'✅ RAG Active — ' + str(result) + ' jobs indexed' if vector_store else '⚠️ RAG not available'}"
        )
    }
    if saved_messages:
        st.session_state.messages = [greeting] + saved_messages
    else:
        st.session_state.messages = [greeting]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask about jobs, skills, salaries, risk score..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    save_chat_message(st.session_state["user_id"], "user", prompt)

    is_hindi = bool(re.search("[\u0900-\u097F]", prompt))
    processing_prompt = translate_to_english(prompt) if is_hindi else prompt

    if is_hindi:
        with st.sidebar:
            st.info(f"🌐 Translated: *{processing_prompt}*")

    # ── Priority 1: Risk
    risk_answer = handle_risk_conversation(processing_prompt, is_hindi)

    if risk_answer:
        st.session_state.messages.append({"role": "assistant", "content": risk_answer})
        st.chat_message("assistant").write(risk_answer)
        save_chat_message(st.session_state["user_id"], "assistant", risk_answer)

    # ── Priority 2: RAG — semantic + follow-up
    elif should_use_rag(processing_prompt) and vector_store:
        with st.spinner("🔍 Searching..."):
            answer, relevant_docs, rewritten = rag_answer(
                processing_prompt,
                vector_store,
                st.session_state.messages[:-1],
                is_hindi
            )

        if rewritten != processing_prompt:
            st.caption(f"🔄 Searched for: _{rewritten}_")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
        save_chat_message(st.session_state["user_id"], "assistant", answer)

    # ── Priority 3: Tool calling — structured queries
    else:
        with st.spinner("🧠 Thinking..."):
            tool_calls = plan_tool_call(processing_prompt, st.session_state.messages[:-1])

            with st.sidebar:
                st.subheader("🔍 Tool Plan")
                for tc in tool_calls:
                    st.markdown(f"**🔧 {tc['tool']}**")
                    st.caption(tc.get("reason", ""))
                    st.json(tc.get("params", {}))
                    st.divider()

            tool_results = execute_tool_calls(tool_calls)
            answer = generate_final_answer(
                processing_prompt,
                tool_results,
                st.session_state.messages[:-1],
                is_hindi
            )

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
        save_chat_message(st.session_state["user_id"], "assistant", answer)