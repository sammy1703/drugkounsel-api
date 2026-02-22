import os
import re
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# ---------------- CONFIGURATION ----------------
load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
CORS(app)

# Logging configuration for production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MED_DB_PATH = os.path.join(os.path.dirname(__file__), "data/medicines.json")
INT_DB_PATH = os.path.join(os.path.dirname(__file__), "data/interactions.json")
GPT_MODEL = "gpt-4o-mini"
GPT_TEMP = 0.2

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- DATABASE ENGINE ----------------
MEDICINE_DB = []
INTERACTION_DB = []

def load_databases():
    """Charge les bases de données JSON au démarrage."""
    global MEDICINE_DB, INTERACTION_DB
    try:
        if os.path.exists(MED_DB_PATH):
            with open(MED_DB_PATH, "r", encoding="utf-8") as f:
                MEDICINE_DB = json.load(f)
        if os.path.exists(INT_DB_PATH):
            with open(INT_DB_PATH, "r", encoding="utf-8") as f:
                INTERACTION_DB = json.load(f)
        logger.info(f"DB Loaded: {len(MEDICINE_DB)} meds, {len(INTERACTION_DB)} interactions.")
    except Exception as e:
        logger.error(f"Failed to load DB: {e}")

load_databases()

# ---------------- CORE ENGINE HELPERS ----------------

def normalize(name):
    """Normalizes drug names for reliable matching."""
    if not name: return ""
    return re.sub(r'[^a-z0-9]', '', name.lower().strip())

def resolve_to_generic(drug_name):
    """Resolves a brand name to its generic counterpart using the DB."""
    norm_input = normalize(drug_name)
    for med in MEDICINE_DB:
        # Check generic name
        generic = med.get("generic") or med.get("name")
        if normalize(generic) == norm_input:
            return generic
        
        # Check brand names
        brands = med.get("brands", [])
        if any(normalize(b) == norm_input for b in brands):
            return generic or med.get("name")
            
    return drug_name  # Return original if not found

def severity_to_score(severity):
    """Ranks severity for finding the most dangerous interaction."""
    ranks = {"SEVERE": 3, "MODERATE": 2, "MILD": 1}
    return ranks.get(severity.upper(), 0)

# ---------------- INTERACTION ENGINE ----------------

def check_drug_interactions(target_drug, cabinet):
    """
    Checks the target drug against all drugs in the patient's cabinet.
    Implements bidirectional matching and highest severity detection.
    """
    if not cabinet:
        return {
            "status": "safe",
            "risk_level": "NONE",
            "message": "No other medicines to check.",
            "patient_advice": "No interactions found with current list."
        }

    target_gen = normalize(resolve_to_generic(target_drug))
    cabinet_gens = [normalize(resolve_to_generic(d)) for d in cabinet]
    
    found_interactions = []

    for other_gen in cabinet_gens:
        if not other_gen or other_gen == target_gen:
            continue
            
        for inter in INTERACTION_DB:
            db_drugs = [normalize(d) for d in inter.get("drugs", [])]
            # Bidirectional check: Is both the target and the other drug in the DB entry?
            if target_gen in db_drugs and other_gen in db_drugs:
                found_interactions.append(inter)

    if not found_interactions:
        return {
            "status": "safe",
            "risk_level": "NONE",
            "message": "No clinically significant interactions found in database.",
            "patient_advice": "Use as directed by your healthcare professional."
        }

    # Find the most severe interaction
    highest = max(found_interactions, key=lambda x: severity_to_score(x.get("severity", "MILD")))

    return {
        "status": "warning",
        "risk_level": highest.get("severity", "MODERATE").upper(),
        "involved": highest.get("drugs", []),
        "message": highest.get("description", "A potential interaction was detected."),
        "patient_advice": highest.get("advice", "Consult your doctor or pharmacist.")
    }

# ---------------- AI FALLBACK ENGINE ----------------

def generate_ai_counseling(drug_name, language="English", cabinet=None):
    """Generates clinical-grade counseling using GPT fallback (gpt-4o-mini)."""
    
    system_prompt = """You are RxKounsel AI, a clinical pharmacology engine.
    Generate a JSON object with strictly:
    - 'drug': Generic name
    - 'sections': List of {header, content} (headers: WHAT IS THIS MEDICINE FOR, HOW TO TAKE, IMPORTANT WARNINGS, COMMON SIDE EFFECTS, WHEN TO SEEK MEDICAL HELP, GENERAL ADVICE)
    - 'brands': List of common brands
    - 'language': The requested language
    Rules: Minimum 100 words per section. Simple language. Medical accuracy is mandatory."""

    user_prompt = f"Medicine: {drug_name}\nLanguage: {language}\nCabinet: {cabinet}"

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=GPT_TEMP,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            
            # Simple validation of structure
            if "sections" in data and len(data["sections"]) >= 4:
                # Store in DB for future use
                new_entry = {
                    "generic": data.get("drug", drug_name).capitalize(),
                    "brands": data.get("brands", []),
                    "indication": data["sections"][0]["content"],
                    "how_to_take": data["sections"][1].get("content", ""),
                    "warnings": data["sections"][2].get("content", ""),
                    "side_effects": data["sections"][3].get("content", ""),
                    "seek_help": data["sections"][4].get("content", "") if len(data["sections"]) > 4 else "",
                    "general_advice": data["sections"][5].get("content", "") if len(data["sections"]) > 5 else ""
                }
                save_to_db(new_entry)
                return data
                
        except Exception as e:
            logger.error(f"AI Attempt {attempt+1} failed: {e}")
            if attempt == 1: return None
            
    return None

def save_to_db(entry):
    """Appends a new medicine to the JSON database safely."""
    global MEDICINE_DB
    try:
        # Check if already exists to avoid duplicates
        norm_new = normalize(entry["generic"])
        if any(normalize(m.get("generic")) == norm_new for m in MEDICINE_DB):
            return

        MEDICINE_DB.append(entry)
        os.makedirs(os.path.dirname(MED_DB_PATH), exist_ok=True)
        with open(MED_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(MEDICINE_DB, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved new medicine to DB: {entry['generic']}")
    except Exception as e:
        logger.error(f"Failed to save to DB: {e}")

# ---------------- API ROUTES ----------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "RxKounsel Production API", "version": "2.0.0"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "db_size": len(MEDICINE_DB),
        "uptime_checkpoint": "active"
    }), 200

@app.route("/api/counseling", methods=["POST"])
def counseling():
    """Unified endpoint for Counseling + Interaction Detection."""
    try:
        payload = request.get_json(force=True)
        drug = payload.get("drug", "").strip()
        lang = payload.get("lang", "English").strip()
        cabinet = payload.get("existing_drugs", []) # Cabinet = currently taken drugs

        if not drug:
            return jsonify({"error": "Drug name is required"}), 400

        logger.info(f"Request: Drug={drug}, Lang={lang}, CabinetSize={len(cabinet)}")

        # 1. SEARCH LOCAL DB FIRST
        med_data = None
        db_match = None
        
        norm_target = normalize(drug)
        for m in MEDICINE_DB:
            if normalize(m.get("generic")) == norm_target or any(normalize(b) == norm_target for b in m.get("brands", [])):
                db_match = m
                break
        
        if db_match:
            logger.info(f"Database Hit: {drug}")
            # Build sections from DB format
            sections = [
                {"header": "WHAT IS THIS MEDICINE FOR", "content": db_match.get("indication", "N/A")},
                {"header": "HOW TO TAKE", "content": db_match.get("how_to_take", "N/A")},
                {"header": "IMPORTANT WARNINGS", "content": db_match.get("warnings", "N/A")},
                {"header": "COMMON SIDE EFFECTS", "content": db_match.get("side_effects", "N/A")},
                {"header": "WHEN TO SEEK MEDICAL HELP", "content": db_match.get("seek_help", "N/A")},
                {"header": "GENERAL ADVICE", "content": db_match.get("general_advice", "N/A")}
            ]
            med_data = {"drug": db_match.get("generic"), "sections": sections}
        
        # 2. AI FALLBACK IF NOT FOUND
        if not med_data:
            logger.info(f"Database Miss. Triggering AI Fallback for {drug}...")
            med_data = generate_ai_counseling(drug, lang, cabinet)

        if not med_data:
            return jsonify({"error": "Counseling could not be generated at this time."}), 503

        # 3. INTERACTION ENGINE
        interaction_report = check_drug_interactions(drug, cabinet)

        # 4. UNIFIED RESPONSE
        return jsonify({
            "status": "success",
            "medicine": med_data.get("drug"),
            "sections": med_data.get("sections"),
            "interaction_report": interaction_report,
            "metadata": {
                "language": lang,
                "source": "database" if db_match else "ai_generation"
            }
        }), 200

    except Exception as e:
        logger.error(f"Endpoint Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# ---------------- SERVER ----------------
if __name__ == "__main__":
    # Development Server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
