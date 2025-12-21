"""
Add diverse training samples to demonstrate RLHF model functionality.
This creates realistic medical Q&A pairs with varied quality levels.
"""

import os
import psycopg
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

db_config = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

# Diverse training samples with clear quality differences
training_samples = [
    # High quality (rating 5) - Comprehensive, medically accurate
    {
        "session_id": 1001,
        "prompt": "What are the symptoms of pneumonia?",
        "response": "Pneumonia symptoms typically include persistent cough with phlegm (yellow, green, or bloody), high fever (101-105°F), chest pain that worsens with breathing or coughing, shortness of breath, fatigue, confusion (especially in elderly patients), nausea, vomiting, and diarrhea. Severe cases may present with bluish lips or nails due to oxygen deprivation. Seek immediate medical attention if experiencing severe symptoms.",
        "rating": 5
    },
    {
        "session_id": 1002,
        "prompt": "What medications are used to treat asthma?",
        "response": "Asthma treatment includes two main categories: (1) Long-term control medications such as inhaled corticosteroids (fluticasone, budesonide), leukotriene modifiers (montelukast), and long-acting beta agonists (salmeterol). (2) Quick-relief medications including short-acting beta agonists (albuterol, levalbuterol). Treatment plans are personalized based on symptom severity, frequency, and individual response. Regular monitoring and adjustment are essential.",
        "rating": 5
    },
    {
        "session_id": 1003,
        "prompt": "How is diabetes diagnosed?",
        "response": "Diabetes is diagnosed using several criteria: (1) Fasting plasma glucose ≥126 mg/dL after 8-hour fast, (2) HbA1c ≥6.5%, (3) 2-hour plasma glucose ≥200 mg/dL during oral glucose tolerance test, or (4) Random plasma glucose ≥200 mg/dL with classic symptoms. Diagnosis requires confirmation with repeat testing unless unequivocal hyperglycemia exists. Prediabetes is indicated by HbA1c 5.7-6.4% or fasting glucose 100-125 mg/dL.",
        "rating": 5
    },
    
    # Medium quality (rating 3) - Correct but incomplete
    {
        "session_id": 1004,
        "prompt": "What are the symptoms of pneumonia?",
        "response": "Pneumonia symptoms include cough, fever, chest pain, and difficulty breathing. You should see a doctor if symptoms are severe.",
        "rating": 3
    },
    {
        "session_id": 1005,
        "prompt": "What medications are used to treat asthma?",
        "response": "Asthma is treated with inhalers. There are two types: controller medications you take daily and rescue inhalers for quick relief like albuterol.",
        "rating": 3
    },
    {
        "session_id": 1006,
        "prompt": "How is diabetes diagnosed?",
        "response": "Diabetes is diagnosed with blood tests that measure blood sugar levels. Your doctor will check fasting glucose or HbA1c levels.",
        "rating": 3
    },
    
    # Low quality (rating 1-2) - Too brief, vague, or incorrect
    {
        "session_id": 1007,
        "prompt": "What are the symptoms of pneumonia?",
        "response": "You might have a cough and feel tired.",
        "rating": 1
    },
    {
        "session_id": 1008,
        "prompt": "What medications are used to treat asthma?",
        "response": "Inhalers help with breathing.",
        "rating": 2
    },
    {
        "session_id": 1009,
        "prompt": "How is diabetes diagnosed?",
        "response": "A blood test will show if you have diabetes.",
        "rating": 2
    },
    
    # More diverse topics - High quality
    {
        "session_id": 1010,
        "prompt": "What is the treatment for hypertension?",
        "response": "Hypertension treatment begins with lifestyle modifications: reduced sodium intake (<2300mg/day), regular exercise (150min/week), weight loss if overweight, limited alcohol, and stress management. Pharmacological treatment includes first-line agents: ACE inhibitors (lisinopril), ARBs (losartan), calcium channel blockers (amlodipine), and thiazide diuretics (hydrochlorothiazide). Choice depends on patient factors, comorbidities, and blood pressure goals (<130/80 mmHg for most patients).",
        "rating": 5
    },
    {
        "session_id": 1011,
        "prompt": "What are the risk factors for stroke?",
        "response": "Major stroke risk factors include: hypertension (most significant), diabetes mellitus, atrial fibrillation, smoking, hyperlipidemia, obesity, sedentary lifestyle, excessive alcohol use, family history, age >55, prior stroke/TIA, carotid artery disease, and certain blood disorders. Modifiable factors should be aggressively managed through lifestyle changes and medications to reduce stroke risk by up to 80%.",
        "rating": 5
    },
    {
        "session_id": 1012,
        "prompt": "What is COPD and how is it managed?",
        "response": "COPD (Chronic Obstructive Pulmonary Disease) is progressive lung disease causing airflow limitation, primarily from smoking. Management includes: (1) Smoking cessation (most critical), (2) Bronchodilators (short/long-acting beta agonists, anticholinergics), (3) Inhaled corticosteroids for severe cases, (4) Pulmonary rehabilitation, (5) Oxygen therapy if hypoxemic, (6) Vaccinations (influenza, pneumococcal), and (7) Treatment of exacerbations. GOLD guidelines stratify treatment by symptom severity.",
        "rating": 5
    },
    
    # More diverse - Medium quality
    {
        "session_id": 1013,
        "prompt": "What is the treatment for hypertension?",
        "response": "High blood pressure is treated with medications like ACE inhibitors, beta blockers, and diuretics. Lifestyle changes like diet and exercise also help.",
        "rating": 3
    },
    {
        "session_id": 1014,
        "prompt": "What are the risk factors for stroke?",
        "response": "Risk factors for stroke include high blood pressure, diabetes, smoking, high cholesterol, and age.",
        "rating": 3
    },
    {
        "session_id": 1015,
        "prompt": "What is COPD and how is it managed?",
        "response": "COPD is a lung disease usually from smoking. Treatment includes inhalers, quitting smoking, and sometimes oxygen therapy.",
        "rating": 3
    },
    
    # More diverse - Low quality
    {
        "session_id": 1016,
        "prompt": "What is the treatment for hypertension?",
        "response": "Take blood pressure medication and eat healthy.",
        "rating": 2
    },
    {
        "session_id": 1017,
        "prompt": "What are the risk factors for stroke?",
        "response": "High blood pressure and smoking can cause strokes.",
        "rating": 2
    },
    {
        "session_id": 1018,
        "prompt": "What is COPD and how is it managed?",
        "response": "It's a breathing problem. Use inhalers.",
        "rating": 1
    },
]

def clear_existing_samples():
    """Remove duplicate samples to start fresh."""
    with psycopg.connect(**db_config, autocommit=True) as conn:
        with conn.cursor() as cur:
            # Keep only samples with distinct prompts/responses
            cur.execute("DELETE FROM rlhf_interactions WHERE rating IS NOT NULL")
            print(f"✅ Cleared existing rated samples")

def add_samples():
    """Add diverse training samples."""
    now = datetime.utcnow()
    
    with psycopg.connect(**db_config, autocommit=True) as conn:
        with conn.cursor() as cur:
            for sample in training_samples:
                cur.execute("""
                    INSERT INTO rlhf_interactions 
                        (session_id, user_prompt, ai_response, rating, feedback_comment, 
                         created_by, updated_by, created_dt, updated_dt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    sample["session_id"],
                    sample["prompt"],
                    sample["response"],
                    sample["rating"],
                    f"Quality rating: {sample['rating']}/5",
                    1001,  # created_by
                    1001,  # updated_by
                    now,
                    now
                ))
    
    print(f"✅ Added {len(training_samples)} diverse training samples")
    print(f"   High quality (rating 5): {sum(1 for s in training_samples if s['rating'] == 5)}")
    print(f"   Medium quality (rating 3): {sum(1 for s in training_samples if s['rating'] == 3)}")
    print(f"   Low quality (rating 1-2): {sum(1 for s in training_samples if s['rating'] <= 2)}")

if __name__ == "__main__":
    print("=" * 80)
    print("ADDING DIVERSE RLHF TRAINING SAMPLES")
    print("=" * 80)
    print()
    print("⚠️  Note: This will ADD samples without removing existing ones.")
    print("   The new diverse samples will help the model learn better patterns.")
    print()
    
    add_samples()
    
    print()
    print("=" * 80)
    print("✅ DONE! Now retrain the model with: python train_reward_sbert.py")
    print("=" * 80)
