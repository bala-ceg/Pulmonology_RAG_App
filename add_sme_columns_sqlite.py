#!/usr/bin/env python3
"""
Add SME review columns and doctors table to local SQLite database.
Run this script to add the new SME columns and doctors table to local_sft.db
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'local_sft.db')

# Sample doctors data: 3 doctors per department (30 departments = 90 doctors)
SAMPLE_DOCTORS = [
    # Cardiology
    ('Dr. Sarah Chen', 'sarah.chen@hospital.org', 'Cardiology', 'Interventional Cardiology'),
    ('Dr. Michael Roberts', 'michael.roberts@hospital.org', 'Cardiology', 'Electrophysiology'),
    ('Dr. Emily Johnson', 'emily.johnson@hospital.org', 'Cardiology', 'Heart Failure'),
    # Neurology
    ('Dr. James Wilson', 'james.wilson@hospital.org', 'Neurology', 'Stroke'),
    ('Dr. Lisa Park', 'lisa.park@hospital.org', 'Neurology', 'Epilepsy'),
    ('Dr. David Martinez', 'david.martinez@hospital.org', 'Neurology', 'Movement Disorders'),
    # Diabetes
    ('Dr. Amanda Thompson', 'amanda.thompson@hospital.org', 'Diabetes', 'Type 1 Diabetes'),
    ('Dr. Robert Kim', 'robert.kim@hospital.org', 'Diabetes', 'Type 2 Diabetes'),
    ('Dr. Jennifer Lee', 'jennifer.lee@hospital.org', 'Diabetes', 'Diabetic Complications'),
    # Pulmonology
    ('Dr. Christopher Brown', 'chris.brown@hospital.org', 'Pulmonology', 'COPD'),
    ('Dr. Michelle Garcia', 'michelle.garcia@hospital.org', 'Pulmonology', 'Asthma'),
    ('Dr. Andrew Taylor', 'andrew.taylor@hospital.org', 'Pulmonology', 'Interstitial Lung Disease'),
    # Gastroenterology
    ('Dr. Patricia White', 'patricia.white@hospital.org', 'Gastroenterology', 'IBD'),
    ('Dr. Daniel Anderson', 'daniel.anderson@hospital.org', 'Gastroenterology', 'Hepatology'),
    ('Dr. Rachel Moore', 'rachel.moore@hospital.org', 'Gastroenterology', 'Endoscopy'),
    # Nephrology
    ('Dr. Steven Clark', 'steven.clark@hospital.org', 'Nephrology', 'Dialysis'),
    ('Dr. Laura Rodriguez', 'laura.rodriguez@hospital.org', 'Nephrology', 'Transplant'),
    ('Dr. Kevin Wright', 'kevin.wright@hospital.org', 'Nephrology', 'Glomerular Disease'),
    # Oncology
    ('Dr. Nancy Lewis', 'nancy.lewis@hospital.org', 'Oncology', 'Breast Cancer'),
    ('Dr. Thomas Hall', 'thomas.hall@hospital.org', 'Oncology', 'Lung Cancer'),
    ('Dr. Karen Young', 'karen.young@hospital.org', 'Oncology', 'Hematologic Oncology'),
    # Hematology
    ('Dr. Brian King', 'brian.king@hospital.org', 'Hematology', 'Coagulation'),
    ('Dr. Susan Scott', 'susan.scott@hospital.org', 'Hematology', 'Anemia'),
    ('Dr. Joseph Green', 'joseph.green@hospital.org', 'Hematology', 'Blood Disorders'),
    # Orthopedics
    ('Dr. Elizabeth Adams', 'elizabeth.adams@hospital.org', 'Orthopedics', 'Joint Replacement'),
    ('Dr. Richard Baker', 'richard.baker@hospital.org', 'Orthopedics', 'Sports Medicine'),
    ('Dr. Maria Nelson', 'maria.nelson@hospital.org', 'Orthopedics', 'Spine Surgery'),
    # Dermatology
    ('Dr. Charles Hill', 'charles.hill@hospital.org', 'Dermatology', 'Skin Cancer'),
    ('Dr. Dorothy Carter', 'dorothy.carter@hospital.org', 'Dermatology', 'Psoriasis'),
    ('Dr. Frank Mitchell', 'frank.mitchell@hospital.org', 'Dermatology', 'Dermatitis'),
    # Ophthalmology
    ('Dr. Helen Perez', 'helen.perez@hospital.org', 'Ophthalmology', 'Retina'),
    ('Dr. George Roberts', 'george.roberts@hospital.org', 'Ophthalmology', 'Glaucoma'),
    ('Dr. Betty Turner', 'betty.turner@hospital.org', 'Ophthalmology', 'Cataract Surgery'),
    # Psychiatry
    ('Dr. Edward Phillips', 'edward.phillips@hospital.org', 'Psychiatry', 'Depression'),
    ('Dr. Margaret Campbell', 'margaret.campbell@hospital.org', 'Psychiatry', 'Anxiety Disorders'),
    ('Dr. Ronald Parker', 'ronald.parker@hospital.org', 'Psychiatry', 'Bipolar Disorder'),
    # Pediatrics
    ('Dr. Sandra Evans', 'sandra.evans@hospital.org', 'Pediatrics', 'General Pediatrics'),
    ('Dr. Kenneth Edwards', 'kenneth.edwards@hospital.org', 'Pediatrics', 'Neonatology'),
    ('Dr. Carol Collins', 'carol.collins@hospital.org', 'Pediatrics', 'Pediatric Cardiology'),
    # Rheumatology
    ('Dr. Mark Stewart', 'mark.stewart@hospital.org', 'Rheumatology', 'Rheumatoid Arthritis'),
    ('Dr. Diane Sanchez', 'diane.sanchez@hospital.org', 'Rheumatology', 'Lupus'),
    ('Dr. Paul Morris', 'paul.morris@hospital.org', 'Rheumatology', 'Fibromyalgia'),
    # Urology
    ('Dr. Angela Rogers', 'angela.rogers@hospital.org', 'Urology', 'Prostate'),
    ('Dr. Timothy Reed', 'timothy.reed@hospital.org', 'Urology', 'Kidney Stones'),
    ('Dr. Sharon Cook', 'sharon.cook@hospital.org', 'Urology', 'Urologic Oncology'),
    # Immunology & Allergy
    ('Dr. Larry Morgan', 'larry.morgan@hospital.org', 'Immunology & Allergy', 'Food Allergies'),
    ('Dr. Virginia Bell', 'virginia.bell@hospital.org', 'Immunology & Allergy', 'Asthma/Allergy'),
    ('Dr. Raymond Murphy', 'raymond.murphy@hospital.org', 'Immunology & Allergy', 'Immunodeficiency'),
    # Infectious Disease
    ('Dr. Joyce Bailey', 'joyce.bailey@hospital.org', 'Infectious Disease', 'HIV/AIDS'),
    ('Dr. Dennis Rivera', 'dennis.rivera@hospital.org', 'Infectious Disease', 'Hospital Infections'),
    ('Dr. Judith Cooper', 'judith.cooper@hospital.org', 'Infectious Disease', 'Tropical Diseases'),
    # Emergency Medicine
    ('Dr. Gerald Richardson', 'gerald.richardson@hospital.org', 'Emergency Medicine', 'Trauma'),
    ('Dr. Teresa Cox', 'teresa.cox@hospital.org', 'Emergency Medicine', 'Critical Care'),
    ('Dr. Jerry Howard', 'jerry.howard@hospital.org', 'Emergency Medicine', 'Toxicology'),
    # Hepatology
    ('Dr. Debra Ward', 'debra.ward@hospital.org', 'Hepatology', 'Viral Hepatitis'),
    ('Dr. Wayne Torres', 'wayne.torres@hospital.org', 'Hepatology', 'Liver Transplant'),
    ('Dr. Gloria Peterson', 'gloria.peterson@hospital.org', 'Hepatology', 'Cirrhosis'),
    # Nutrition & Dietetics
    ('Dr. Roy Gray', 'roy.gray@hospital.org', 'Nutrition & Dietetics', 'Clinical Nutrition'),
    ('Dr. Alice Ramirez', 'alice.ramirez@hospital.org', 'Nutrition & Dietetics', 'Obesity Medicine'),
    ('Dr. Eugene James', 'eugene.james@hospital.org', 'Nutrition & Dietetics', 'Metabolic Disorders'),
    # Pharmacology
    ('Dr. Ann Watson', 'ann.watson@hospital.org', 'Pharmacology', 'Clinical Pharmacology'),
    ('Dr. Russell Brooks', 'russell.brooks@hospital.org', 'Pharmacology', 'Drug Interactions'),
    ('Dr. Frances Kelly', 'frances.kelly@hospital.org', 'Pharmacology', 'Pharmacokinetics'),
    # General Medicine
    ('Dr. Jack Sanders', 'jack.sanders@hospital.org', 'General Medicine', 'Internal Medicine'),
    ('Dr. Ruby Price', 'ruby.price@hospital.org', 'General Medicine', 'Primary Care'),
    ('Dr. Albert Bennett', 'albert.bennett@hospital.org', 'General Medicine', 'Preventive Medicine'),
    # ENT (Otolaryngology)
    ('Dr. Phyllis Wood', 'phyllis.wood@hospital.org', 'ENT (Otolaryngology)', 'Head & Neck Surgery'),
    ('Dr. Jesse Barnes', 'jesse.barnes@hospital.org', 'ENT (Otolaryngology)', 'Hearing Disorders'),
    ('Dr. Lillian Ross', 'lillian.ross@hospital.org', 'ENT (Otolaryngology)', 'Sinus Surgery'),
    # Obstetrics & Gynecology
    ('Dr. Howard Henderson', 'howard.henderson@hospital.org', 'Obstetrics & Gynecology', 'High-Risk Pregnancy'),
    ('Dr. Jean Coleman', 'jean.coleman@hospital.org', 'Obstetrics & Gynecology', 'Reproductive Medicine'),
    ('Dr. Arthur Jenkins', 'arthur.jenkins@hospital.org', 'Obstetrics & Gynecology', 'Gynecologic Oncology'),
    # Radiology
    ('Dr. Catherine Perry', 'catherine.perry@hospital.org', 'Radiology', 'Diagnostic Radiology'),
    ('Dr. Henry Powell', 'henry.powell@hospital.org', 'Radiology', 'Interventional Radiology'),
    ('Dr. Ruth Long', 'ruth.long@hospital.org', 'Radiology', 'Neuroradiology'),
    # Anesthesiology
    ('Dr. Philip Patterson', 'philip.patterson@hospital.org', 'Anesthesiology', 'Cardiac Anesthesia'),
    ('Dr. Marie Hughes', 'marie.hughes@hospital.org', 'Anesthesiology', 'Pediatric Anesthesia'),
    ('Dr. Ralph Flores', 'ralph.flores@hospital.org', 'Anesthesiology', 'Pain Management'),
    # Pain Management
    ('Dr. Evelyn Washington', 'evelyn.washington@hospital.org', 'Pain Management', 'Chronic Pain'),
    ('Dr. Louis Butler', 'louis.butler@hospital.org', 'Pain Management', 'Interventional Pain'),
    ('Dr. Mildred Simmons', 'mildred.simmons@hospital.org', 'Pain Management', 'Cancer Pain'),
    # Sleep Medicine
    ('Dr. Johnny Foster', 'johnny.foster@hospital.org', 'Sleep Medicine', 'Sleep Apnea'),
    ('Dr. Doris Gonzales', 'doris.gonzales@hospital.org', 'Sleep Medicine', 'Insomnia'),
    ('Dr. Earl Bryant', 'earl.bryant@hospital.org', 'Sleep Medicine', 'Narcolepsy'),
    # Sports Medicine
    ('Dr. Martha Alexander', 'martha.alexander@hospital.org', 'Sports Medicine', 'Athletic Injuries'),
    ('Dr. Bruce Russell', 'bruce.russell@hospital.org', 'Sports Medicine', 'Rehabilitation'),
    ('Dr. Irene Griffin', 'irene.griffin@hospital.org', 'Sports Medicine', 'Performance Medicine'),
    # Geriatrics
    ('Dr. Stanley Diaz', 'stanley.diaz@hospital.org', 'Geriatrics', 'Dementia Care'),
    ('Dr. Hazel Hayes', 'hazel.hayes@hospital.org', 'Geriatrics', 'Palliative Care'),
    ('Dr. Fred Myers', 'fred.myers@hospital.org', 'Geriatrics', 'Geriatric Assessment'),
]


def add_sme_columns():
    """Add SME review columns to sft_ranked_data table in SQLite."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        print("   The database will be created automatically when you use the app.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    try:
        # Check if table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sft_ranked_data'")
        if not cur.fetchone():
            print(f"❌ Table 'sft_ranked_data' does not exist in {DB_PATH}")
            print("   Run the app first to create the tables.")
            conn.close()
            return
        
        # Check existing columns
        cur.execute("PRAGMA table_info(sft_ranked_data)")
        existing_columns = {row[1] for row in cur.fetchall()}
        
        columns_to_add = [
            ('domain', 'TEXT'),
            ('sme_score', 'INTEGER CHECK (sme_score >= 1 AND sme_score <= 5)'),
            ('sme_score_reason', 'TEXT'),
            ('sme_reviewed_by', 'TEXT'),
            ('sme_reviewed_at', 'TIMESTAMP')
        ]
        
        added_count = 0
        for column_name, column_type in columns_to_add:
            if column_name not in existing_columns:
                print(f"➕ Adding column: {column_name} ({column_type})")
                cur.execute(f"ALTER TABLE sft_ranked_data ADD COLUMN {column_name} {column_type}")
                added_count += 1
            else:
                print(f"✅ Column already exists: {column_name}")
        
        # Create indexes
        print("\n📊 Creating indexes...")
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_domain ON sft_ranked_data(domain)")
            print("✅ Created index: idx_sft_ranked_data_domain")
        except Exception as e:
            print(f"⚠️  Index already exists or error: {e}")
        
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sft_ranked_data_sme_score ON sft_ranked_data(sme_score)")
            print("✅ Created index: idx_sft_ranked_data_sme_score")
        except Exception as e:
            print(f"⚠️  Index already exists or error: {e}")
        
        conn.commit()
        
        print(f"\n✅ Migration complete!")
        print(f"   Added {added_count} new column(s)")
        print(f"   Database: {DB_PATH}")
        
        # Verify final schema
        print("\n📋 Final schema for sft_ranked_data:")
        cur.execute("PRAGMA table_info(sft_ranked_data)")
        for row in cur.fetchall():
            print(f"   - {row[1]}: {row[2]}")
        
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def add_doctors_table():
    """Create sme_doctors table and seed with sample data."""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        print("   The database will be created automatically when you use the app.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    try:
        # Create sme_doctors table
        print("\n👨‍⚕️ Creating sme_doctors table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sme_doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                department TEXT NOT NULL,
                specialty TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Table sme_doctors created")
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sme_doctors_department ON sme_doctors(department)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sme_doctors_active ON sme_doctors(is_active, department)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sme_doctors_name ON sme_doctors(name)")
        print("✅ Indexes created")
        
        # Check if table has data
        cur.execute("SELECT COUNT(*) FROM sme_doctors")
        count = cur.fetchone()[0]
        
        if count == 0:
            print("\n📥 Seeding sample doctors data...")
            cur.executemany("""
                INSERT INTO sme_doctors (name, email, department, specialty)
                VALUES (?, ?, ?, ?)
            """, SAMPLE_DOCTORS)
            print(f"✅ Inserted {len(SAMPLE_DOCTORS)} sample doctors (3 per department)")
        else:
            print(f"\n✅ Table already has {count} doctors, skipping seed data")
        
        conn.commit()
        
        # Show summary by department
        print("\n📊 Doctors by department:")
        cur.execute("""
            SELECT department, COUNT(*) as count 
            FROM sme_doctors 
            WHERE is_active = 1
            GROUP BY department 
            ORDER BY department
        """)
        for row in cur.fetchall():
            print(f"   - {row[0]}: {row[1]} doctors")
        
    except Exception as e:
        print(f"\n❌ Error creating doctors table: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("SME Review Columns & Doctors Table - SQLite Migration")
    print("=" * 60)
    print()
    add_sme_columns()
    print()
    add_doctors_table()
