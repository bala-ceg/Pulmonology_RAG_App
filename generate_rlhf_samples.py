"""
RLHF Sample Data Generator
This script populates the RLHF training database with sample sessions and interactions
for medical AI training purposes.
"""

import psycopg
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

# Sample sessions data
SESSIONS = [
    {
        "session_id": 1,
        "user_id": 1002,
        "model_version": "gpt-4o-v1",
        "session_start": datetime.now() - timedelta(days=7),
        "session_end": datetime.now() - timedelta(days=6),
        "status": "COMPLETED",
        "notes": "Cardiology training data",
        "created_by": 1001,
        "updated_by": 1001
    },
    {
        "session_id": 2,
        "user_id": 1003,
        "model_version": "gpt-4o-v1",
        "session_start": datetime.now() - timedelta(days=5),
        "session_end": datetime.now() - timedelta(days=4),
        "status": "COMPLETED",
        "notes": "Emergency medicine scenarios",
        "created_by": 1001,
        "updated_by": 1001
    },
    {
        "session_id": 3,
        "user_id": 1004,
        "model_version": "gpt-4o-v2",
        "session_start": datetime.now() - timedelta(days=3),
        "session_end": datetime.now() - timedelta(days=2),
        "status": "COMPLETED",
        "notes": "Pulmonology and respiratory cases",
        "created_by": 1001,
        "updated_by": 1001
    },
    {
        "session_id": 4,
        "user_id": 1005,
        "model_version": "gpt-4o-v2",
        "session_start": datetime.now() - timedelta(days=2),
        "session_end": datetime.now() - timedelta(days=1),
        "status": "COMPLETED",
        "notes": "Rheumatology and musculoskeletal",
        "created_by": 1001,
        "updated_by": 1001
    },
    {
        "session_id": 5,
        "user_id": 1006,
        "model_version": "gpt-4o-v2",
        "session_start": datetime.now() - timedelta(days=1),
        "session_end": datetime.now(),
        "status": "COMPLETED",
        "notes": "Primary care and preventive medicine",
        "created_by": 1001,
        "updated_by": 1001
    }
]

# Comprehensive medical training interactions
INTERACTIONS = [
    # Session 1 - Cardiology
    {
        "session_id": 1,
        "user_prompt": "What is the latest treatment guideline for hypertension?",
        "ai_response": "Based on ACC/AHA 2023 guidelines, hypertension management includes: 1) Lifestyle modifications first, 2) Initiate medication if BP ≥130/80 with ASCVD risk ≥10%, 3) First-line agents: ACE inhibitors, ARBs, CCBs, or thiazide diuretics, 4) Target BP <130/80 for most adults.",
        "rating": 5,
        "feedback_comment": "Accurate and up-to-date with current guidelines",
        "bias_flag": False
    },
    {
        "session_id": 1,
        "user_prompt": "How do I differentiate between STEMI and NSTEMI?",
        "ai_response": "STEMI (ST-Elevation MI): Shows ST elevation ≥1mm in contiguous leads or new LBBB. Requires immediate reperfusion. NSTEMI: Has elevated troponin but NO ST elevation. Both require urgent cardiology consultation and treatment.",
        "rating": 5,
        "feedback_comment": "Clear differentiation with actionable steps",
        "bias_flag": False
    },
    {
        "session_id": 1,
        "user_prompt": "What are the indications for dual antiplatelet therapy?",
        "ai_response": "DAPT (Aspirin + P2Y12 inhibitor) indicated for: 1) Post-ACS (minimum 12 months), 2) Post-PCI with stent (duration varies by stent type and bleeding risk), 3) High-risk atherosclerotic disease. Balance ischemic vs bleeding risk.",
        "rating": 4,
        "feedback_comment": "Good overview, could mention specific P2Y12 inhibitors",
        "bias_flag": False
    },
    
    # Session 2 - Emergency Medicine
    {
        "session_id": 2,
        "user_prompt": "What is the FAST exam and when should it be used?",
        "ai_response": "FAST (Focused Assessment with Sonography for Trauma) evaluates for free fluid in: 1) Perihepatic space, 2) Perisplenic space, 3) Pelvis, 4) Pericardium. Use in hemodynamically unstable trauma patients to detect hemoperitoneum or pericardial effusion.",
        "rating": 5,
        "feedback_comment": "Comprehensive explanation of FAST exam",
        "bias_flag": False
    },
    {
        "session_id": 2,
        "user_prompt": "How do you manage anaphylaxis?",
        "ai_response": "Immediate anaphylaxis management: 1) IM Epinephrine 0.3-0.5mg (1:1000) into anterolateral thigh, 2) Remove trigger, 3) Airway management/oxygen, 4) IV fluids for hypotension, 5) Antihistamines and corticosteroids as adjuncts. Monitor for biphasic reaction.",
        "rating": 5,
        "feedback_comment": "Excellent life-saving protocol clearly outlined",
        "bias_flag": False
    },
    {
        "session_id": 2,
        "user_prompt": "What are the Ottawa Ankle Rules?",
        "ai_response": "X-ray indicated if pain in malleolar zone AND: 1) Bone tenderness at posterior edge/tip of lateral malleolus, 2) Bone tenderness at posterior edge/tip of medial malleolus, OR 3) Inability to bear weight immediately and in ED (4 steps).",
        "rating": 5,
        "feedback_comment": "Accurate clinical decision rule",
        "bias_flag": False
    },
    
    # Session 3 - Pulmonology
    {
        "session_id": 3,
        "user_prompt": "What are the diagnostic criteria for ARDS?",
        "ai_response": "Berlin Definition of ARDS: 1) Timing: Within 1 week of known insult, 2) Bilateral opacities on chest imaging, 3) Not fully explained by cardiac failure, 4) PaO2/FiO2 ratio: Mild (200-300), Moderate (100-200), Severe (<100).",
        "rating": 5,
        "feedback_comment": "Precise diagnostic criteria with severity classification",
        "bias_flag": False
    },
    {
        "session_id": 3,
        "user_prompt": "How do you interpret spirometry results?",
        "ai_response": "Key spirometry parameters: 1) FEV1/FVC ratio: <0.7 suggests obstruction, 2) FVC: Reduced in restriction, 3) FEV1: Severity of obstruction. Obstructive: ↓FEV1/FVC. Restrictive: Normal/↑FEV1/FVC, ↓FVC. Requires proper technique and interpretation.",
        "rating": 4,
        "feedback_comment": "Good basic interpretation, could add reversibility testing",
        "bias_flag": False
    },
    {
        "session_id": 3,
        "user_prompt": "What is the management of acute PE?",
        "ai_response": "PE management based on risk: 1) Hemodynamically unstable: Thrombolysis or embolectomy, 2) High-risk stable: Consider thrombolysis, 3) Low-risk: Anticoagulation (LMWH, fondaparinux, or DOAC). CTPA for diagnosis. Risk stratify using PESI or sPESI scores.",
        "rating": 5,
        "feedback_comment": "Comprehensive risk-stratified approach",
        "bias_flag": False
    },
    
    # Session 4 - Rheumatology
    {
        "session_id": 4,
        "user_prompt": "What are the ACR/EULAR classification criteria for SLE?",
        "ai_response": "2019 EULAR/ACR SLE criteria: Requires ANA ≥1:80, then weighted criteria in 7 domains: 1) Constitutional, 2) Hematologic, 3) Neuropsychiatric, 4) Mucocutaneous, 5) Serosal, 6) Musculoskeletal, 7) Renal. Score ≥10 classifies as SLE.",
        "rating": 5,
        "feedback_comment": "Current classification criteria accurately described",
        "bias_flag": False
    },
    {
        "session_id": 4,
        "user_prompt": "How do you differentiate gout from pseudogout?",
        "ai_response": "Gout vs Pseudogout: 1) Crystal type: Gout has negatively birefringent MSU crystals, pseudogout has positively birefringent CPPD crystals, 2) Location: Gout often 1st MTP, pseudogout often knee/wrist, 3) Both diagnosed by synovial fluid analysis showing respective crystals.",
        "rating": 5,
        "feedback_comment": "Clear crystallographic differentiation",
        "bias_flag": False
    },
    {
        "session_id": 4,
        "user_prompt": "What are the red flags for inflammatory back pain?",
        "ai_response": "Inflammatory back pain features: 1) Age of onset <40 years, 2) Insidious onset, 3) Improvement with exercise, 4) No improvement with rest, 5) Night pain (second half), 6) Morning stiffness >30 minutes. Consider spondyloarthropathy.",
        "rating": 5,
        "feedback_comment": "Classic inflammatory features well articulated",
        "bias_flag": False
    },
    
    # Session 5 - Primary Care
    {
        "session_id": 5,
        "user_prompt": "What screening tests are recommended for a 50-year-old woman?",
        "ai_response": "Age 50 woman screening: 1) Mammography every 1-2 years, 2) Colonoscopy (start at 45-50), 3) Blood pressure, lipids, diabetes screening, 4) Bone density (if risk factors), 5) Cervical cancer screening per guidelines, 6) Mental health screening.",
        "rating": 4,
        "feedback_comment": "Good overview, could specify frequencies more precisely",
        "bias_flag": False
    },
    {
        "session_id": 5,
        "user_prompt": "How do you calculate cardiovascular risk?",
        "ai_response": "Pooled Cohort Equations (PCE) estimates 10-year ASCVD risk using: age, sex, race, total cholesterol, HDL, systolic BP, BP treatment status, diabetes, and smoking. Risk categories: Low (<5%), Borderline (5-7.5%), Intermediate (7.5-20%), High (≥20%).",
        "rating": 5,
        "feedback_comment": "Accurate risk calculator with proper categorization",
        "bias_flag": False
    },
    {
        "session_id": 5,
        "user_prompt": "What are the diagnostic criteria for metabolic syndrome?",
        "ai_response": "Metabolic syndrome (≥3 of 5 criteria): 1) Waist circumference: Men >40 inches, Women >35 inches, 2) Triglycerides ≥150 mg/dL, 3) HDL: Men <40, Women <50 mg/dL, 4) BP ≥130/85 mmHg, 5) Fasting glucose ≥100 mg/dL.",
        "rating": 5,
        "feedback_comment": "NCEP ATP III criteria correctly stated",
        "bias_flag": False
    },
    
    # Additional diverse scenarios
    {
        "session_id": 1,
        "user_prompt": "What is the role of beta-blockers after MI?",
        "ai_response": "Post-MI beta-blocker indications: 1) All patients with MI (especially if reduced EF), 2) Continue indefinitely unless contraindicated, 3) Reduces mortality and recurrent MI, 4) Start early (within 24 hours if stable), 5) Target heart rate 55-60 bpm.",
        "rating": 5,
        "feedback_comment": "Evidence-based recommendations clearly stated",
        "bias_flag": False
    },
    {
        "session_id": 2,
        "user_prompt": "How do you assess altered mental status?",
        "ai_response": "Systematic AMS evaluation: 1) ABCs first, 2) Check glucose immediately, 3) Consider AEIOU TIPS mnemonic, 4) Medications/toxins, 5) Vital signs, 6) Neurologic exam, 7) Labs: CBC, BMP, toxicology, 8) Imaging if indicated. Don't miss hypoglycemia, hypoxia, or increased ICP.",
        "rating": 5,
        "feedback_comment": "Systematic approach to critical presentation",
        "bias_flag": False
    },
    {
        "session_id": 3,
        "user_prompt": "What are the indications for mechanical ventilation?",
        "ai_response": "Intubation indications: 1) Airway protection (GCS <8), 2) Failure to oxygenate (PaO2 <50 on high-flow O2), 3) Failure to ventilate (PaCO2 >50 with pH <7.25), 4) Excessive work of breathing, 5) Anticipated clinical course. Consider non-invasive ventilation first if appropriate.",
        "rating": 5,
        "feedback_comment": "Critical decision-making criteria outlined",
        "bias_flag": False
    },
    {
        "session_id": 4,
        "user_prompt": "What labs should I order for a suspected rheumatoid arthritis patient?",
        "ai_response": "RA laboratory workup: 1) RF (Rheumatoid Factor), 2) Anti-CCP antibodies (more specific), 3) ESR and CRP (acute phase reactants), 4) CBC (check for anemia), 5) CMP (baseline kidney/liver), 6) ANA (rule out overlap syndromes). Anti-CCP most specific for RA.",
        "rating": 5,
        "feedback_comment": "Complete diagnostic workup specified",
        "bias_flag": False
    },
    {
        "session_id": 5,
        "user_prompt": "When should I refer a patient to a specialist?",
        "ai_response": "Consider specialist referral when: 1) Diagnosis uncertain after initial workup, 2) Treatment failure with standard therapies, 3) Complex comorbidities, 4) Need for specialized procedures, 5) Patient preference, 6) Medico-legal concerns. Document indication and urgency clearly.",
        "rating": 4,
        "feedback_comment": "Good general guidance but context-dependent",
        "bias_flag": False
    },
]


def insert_sessions(conn):
    """Insert sample RLHF sessions"""
    print("Inserting RLHF sessions...")
    with conn.cursor() as cur:
        for session in SESSIONS:
            try:
                cur.execute("""
                    INSERT INTO rlhf_sessions 
                    (session_id, user_id, model_version, session_start, session_end, 
                     status, notes, created_by, updated_by, created_dt, updated_dt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (session_id) DO UPDATE
                    SET user_id = EXCLUDED.user_id,
                        model_version = EXCLUDED.model_version,
                        notes = EXCLUDED.notes,
                        updated_dt = CURRENT_TIMESTAMP
                """, (
                    session['session_id'],
                    session['user_id'],
                    session['model_version'],
                    session['session_start'],
                    session['session_end'],
                    session['status'],
                    session['notes'],
                    session['created_by'],
                    session['updated_by']
                ))
                print(f"  ✓ Session {session['session_id']} inserted/updated")
            except Exception as e:
                print(f"  ✗ Error inserting session {session['session_id']}: {e}")
        conn.commit()


def insert_interactions(conn):
    """Insert sample RLHF interactions"""
    print("\nInserting RLHF interactions...")
    with conn.cursor() as cur:
        count = 0
        for interaction in INTERACTIONS:
            try:
                cur.execute("""
                    INSERT INTO rlhf_interactions 
                    (session_id, user_prompt, ai_response, rating, feedback_comment, 
                     bias_flag, created_by, updated_by, created_dt, updated_dt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    interaction['session_id'],
                    interaction['user_prompt'],
                    interaction['ai_response'],
                    interaction['rating'],
                    interaction['feedback_comment'],
                    interaction['bias_flag'],
                    1001,
                    1001
                ))
                count += 1
            except Exception as e:
                print(f"  ✗ Error inserting interaction: {e}")
        conn.commit()
        print(f"  ✓ {count} interactions inserted successfully")


def get_statistics(conn):
    """Display statistics about inserted data"""
    print("\n" + "="*60)
    print("RLHF DATABASE STATISTICS")
    print("="*60)
    
    with conn.cursor() as cur:
        # Total sessions
        cur.execute("SELECT COUNT(*) FROM rlhf_sessions")
        total_sessions = cur.fetchone()[0]
        print(f"Total Sessions: {total_sessions}")
        
        # Total interactions
        cur.execute("SELECT COUNT(*) FROM rlhf_interactions")
        total_interactions = cur.fetchone()[0]
        print(f"Total Interactions: {total_interactions}")
        
        # Average rating
        cur.execute("SELECT AVG(rating)::numeric(3,2) FROM rlhf_interactions WHERE rating IS NOT NULL")
        avg_rating = cur.fetchone()[0]
        print(f"Average Rating: {avg_rating}/5.0")
        
        # Rating distribution
        cur.execute("""
            SELECT rating, COUNT(*) as count 
            FROM rlhf_interactions 
            WHERE rating IS NOT NULL 
            GROUP BY rating 
            ORDER BY rating DESC
        """)
        print("\nRating Distribution:")
        for row in cur.fetchall():
            stars = '★' * row[0]
            print(f"  {stars} ({row[0]}): {row[1]} interactions")
        
        # Bias flags
        cur.execute("SELECT COUNT(*) FROM rlhf_interactions WHERE bias_flag = TRUE")
        bias_count = cur.fetchone()[0]
        print(f"\nBias Flagged: {bias_count} interactions")
        
        # Sessions by model version
        cur.execute("""
            SELECT model_version, COUNT(*) as count 
            FROM rlhf_sessions 
            GROUP BY model_version 
            ORDER BY model_version
        """)
        print("\nSessions by Model Version:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} sessions")


def main():
    """Main execution function"""
    print("="*60)
    print("RLHF SAMPLE DATA GENERATOR")
    print("="*60)
    print()
    
    try:
        # Connect to database
        print("Connecting to database...")
        with psycopg.connect(**db_config) as conn:
            print("✓ Connected successfully\n")
            
            # Insert data
            insert_sessions(conn)
            insert_interactions(conn)
            
            # Show statistics
            get_statistics(conn)
            
            print("\n" + "="*60)
            print("✓ Sample data generation completed successfully!")
            print("="*60)
            print("\nYou can now access the RLHF admin panel at:")
            print("http://localhost:3000/admin/rlhf")
            print()
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
