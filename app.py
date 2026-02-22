from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- DATABASES ----------------
MEDICINE_DB = []
INTERACTION_DB = []
MED_DB_PATH = "data/medicines.json"
INT_DB_PATH = "data/interactions.json"

def load_databases():
    global MEDICINE_DB, INTERACTION_DB
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Medicines
    med_path = os.path.join(base_dir, MED_DB_PATH)
    if os.path.exists(med_path):
        try:
            with open(med_path, "r", encoding="utf-8") as f:
                MEDICINE_DB = json.load(f)
            print(f"Loaded {len(MEDICINE_DB)} medicines successfully")
        except Exception as e:
            print(f"❌ Error loading medicines: {e}")
            
    # Load Interactions
    int_path = os.path.join(base_dir, INT_DB_PATH)
    if os.path.exists(int_path):
        try:
            with open(int_path, "r", encoding="utf-8") as f:
                INTERACTION_DB = json.load(f)
            print(f"Loaded {len(INTERACTION_DB)} interactions successfully")
        except Exception as e:
            print(f"❌ Error loading interactions: {e}")

load_databases()

# ---------------- HELPERS ----------------

def find_medicine(drug_name):
    drug_name = drug_name.lower().strip()
    normalized_input = re.sub(r'[^a-z0-9]', '', drug_name)
    
    for med in MEDICINE_DB:
        generic = med.get("generic") or med.get("name") or med.get("medicine")
        if generic:
            normalized_generic = re.sub(r'[^a-z0-9]', '', generic.lower())
            if normalized_input == normalized_generic:
                return med
        
        brands = med.get("brands", [])
        for b in brands:
            normalized_brand = re.sub(r'[^a-z0-9]', '', b.lower())
            if normalized_input == normalized_brand:
                return med
    return None

def severity_rank(level):
    order = {"SEVERE": 3, "MODERATE": 2, "MILD": 1}
    return order.get(level.upper(), 0)

def check_interactions(target_drug_name, existing_drugs):
    if not existing_drugs:
        return {
            "TITLE": "Drug–Drug Interaction Check",
            "RISK_LEVEL": "NONE",
            "MEDICINES_INVOLVED": "No other medicines listed",
            "MESSAGE": "No clinically significant interaction identified.",
            "PATIENT_ADVICE": "Continue medications as prescribed."
        }

    def get_generic(name):
        med = find_medicine(name)
        if med:
            return (med.get("generic") or med.get("name") or name).lower().strip()
        return name.lower().strip()

    target_generic = get_generic(target_drug_name)
    existing_generics = [get_generic(d) for d in existing_drugs]
    
    interactions_found = []
    
    for existing_generic in existing_generics:
        if target_generic == existing_generic:
            continue
            
        for inter in INTERACTION_DB:
            # Normalize drugs in DB entry for comparison
            db_drugs = [re.sub(r'[^a-z0-9]', '', d.lower()) for d in inter["drugs"]]
            norm_target = re.sub(r'[^a-z0-9]', '', target_generic)
            norm_existing = re.sub(r'[^a-z0-9]', '', existing_generic)
            
            if norm_target in db_drugs and norm_existing in db_drugs:
                interactions_found.append(inter)

    if not interactions_found:
        return {
            "TITLE": "Drug–Drug Interaction Check",
            "RISK_LEVEL": "NONE",
            "MEDICINES_INVOLVED": "No significant interaction detected",
            "MESSAGE": "No clinically significant interaction identified.",
            "PATIENT_ADVICE": "Continue medications as prescribed."
        }

    highest = max(interactions_found, key=lambda x: severity_rank(x["severity"]))

    return {
        "TITLE": "Drug–Drug Interaction Check",
        "RISK_LEVEL": highest["severity"].upper(),
        "MEDICINES_INVOLVED": ", ".join(highest["drugs"]),
        "MESSAGE": highest["description"],
        "PATIENT_ADVICE": highest["advice"]
    }

def generate_with_gpt(drug_name, language="English", existing_drugs=None):
    print(f"Medicine {drug_name} not found, generating with RxKounsel AI")
    
    system_prompt = """You are RxKounsel AI, a professional clinical medication counseling system.

STRICT RULES:
- Always generate detailed patient-friendly counseling.
- Minimum 120 words per section.
- Use simple language.
- No empty sections.
- Always return valid JSON only.
- Never return null sections.

FORMAT:
{
  "found": true,
  "drug": "<drug_name>",
  "sections": [
    {
      "header": "WHAT IS THIS MEDICINE FOR",
      "content": "<detailed explanation>"
    },
    {
      "header": "HOW TO TAKE",
      "content": "<clear dosing guidance>"
    },
    {
      "header": "IMPORTANT WARNINGS",
      "content": "<safety warnings>"
    },
    {
      "header": "COMMON SIDE EFFECTS",
      "content": "<common effects explained>"
    },
    {
      "header": "WHEN TO SEEK MEDICAL HELP",
      "content": "<emergency signs>"
    },
    {
      "header": "GENERAL ADVICE",
      "content": "<storage and general care>"
    }
  ],
  "brands": ["brand1", "brand2"]
}
"""

    user_content = f"Drug: {drug_name}\nLanguage: {language}\nExisting drugs: {existing_drugs}"

    # Try up to 2 times if validation fails
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            
            # Validation
            sections = result.get("sections")
            if not sections or not isinstance(sections, list) or len(sections) < 5:
                print(f"⚠️ Attempt {attempt+1} failed validation: sections too short or missing")
                continue

            # Auto-save to DB logic (We save the generic entry)
            # Find the index of sections if we want to store it in the old flat format, 
            # but user wants the new format to be consistent. 
            # I will store it semi-compatibly.
            new_med = {
                "generic": result.get("drug", drug_name).capitalize(),
                "brands": result.get("brands", []),
                "indication": sections[0]["content"],
                "how_to_take": sections[1]["content"],
                "warnings": sections[2]["content"],
                "side_effects": sections[3]["content"],
                "seek_help": sections[4]["content"] if len(sections) > 4 else "Seek help if symptoms worsen.",
                "general_advice": sections[5]["content"] if len(sections) > 5 else "Store safely.",
            }
            
            MEDICINE_DB.append(new_med)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(base_dir, MED_DB_PATH)
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(MEDICINE_DB, f, indent=2, ensure_ascii=False)
            print(f"💾 Auto-saved {drug_name} to medicines.json")
            
            return result
        except Exception as e:
            print(f"❌ Attempt {attempt+1} Error: {e}")
            if attempt == 1:
                raise e

    raise Exception("Failed to generate valid AI counseling after 2 attempts")

def build_structured_counseling(med, lang):
    # If the med object already has 'sections' (from GPT), return them
    if "sections" in med:
        return med["sections"]

    # Otherwise build from DB object
    sections = [
        {"header": "WHAT IS THIS MEDICINE FOR", "content": med.get("indication", "N/A")},
        {"header": "HOW TO TAKE", "content": med.get("how_to_take") or med.get("dosage", "Use as directed.")},
        {"header": "IMPORTANT WARNINGS", "content": med.get("warnings", "Consult physician.")},
        {"header": "COMMON SIDE EFFECTS", "content": med.get("side_effects", "N/A")},
        {"header": "WHEN TO SEEK MEDICAL HELP", "content": med.get("seek_help", "Seek immediate medical attention if you experience severe allergic reactions, difficulty breathing, or swelling of the face, lips, or tongue.")},
        {"header": "GENERAL ADVICE", "content": med.get("general_advice", "Keep away from children.")}
    ]
    
    if not lang or lang.lower() == "english":
        return sections
        
    try:
        instruction = f"Translate the following medical content to {lang}. Keep the exact headers. Return JSON object with 'sections' key (list of headers and content)."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional medical translator. Output JSON object with 'sections' list."},
                {"role": "user", "content": f"{instruction}\n\n{json.dumps(sections)}"}
            ],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        if isinstance(data, list): return data
        if "sections" in data: return data["sections"]
        return sections
    except Exception as e:
        print(f"❌ Translation error: {e}")
        return sections

# Voice system temporarily disabled

# ---------------- ROUTES ----------------

# Audio serving disabled

@app.route("/api/counseling", methods=["POST"])
def counseling():
    try:
        data = request.get_json(force=True)
        drug = data.get("drug", "").strip()
        lang = data.get("lang", "English").strip()
        existing_drugs = data.get("existing_drugs", [])

        print(f"📥 Received request for {drug} in {lang}")

        if not drug:
            return jsonify({
                "found": False, 
                "error": "No drug name provided."
            }), 200

        med = find_medicine(drug)
        if med:
            print("Medicine found in DB")
            sections = build_structured_counseling(med, lang)
        else:
            # Force AI generation if not in DB
            ai_data = generate_with_gpt(drug, lang, existing_drugs)
            print("Medicine generated via RxKounsel AI")
            sections = ai_data["sections"]
            # Update 'med' for interaction check below
            med = {"generic": ai_data.get("drug", drug)}

        interaction_metadata = check_interactions(
            med.get("generic") or med.get("name") or drug,
            existing_drugs
        )
        print("Interaction computed")

        # Voice system disabled
        # audio_url = generate_tts_file(sections, drug, lang)

        return jsonify({
            "found": True,
            "sections": sections,
            "interaction_metadata": interaction_metadata
        }), 200

    except Exception as e:
        print(f"❌ COUNSELING ERROR: {e}")
        return jsonify({
            "found": False, 
            "error": "AI counseling generation failed. Please try again."
        }), 200

@app.route("/")
def home():
    return {"status": "ok"}

@app.route("/health")
def health():
    return {"status": "ok", "db_size": len(MEDICINE_DB)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
