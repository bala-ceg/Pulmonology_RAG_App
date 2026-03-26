-- ============================================================
-- SME Doctors Table Migration
-- ============================================================
-- Adds a doctors management table for department-specific SME review.
-- Each doctor is assigned to a specific medical department.
--
-- Usage (PostgreSQL):
--   psql -h <host> -p 5432 -U <admin_user> -d pces_base -f migrations/add_doctors_table.sql
-- ============================================================

-- 1. Create sme_doctors table
CREATE TABLE IF NOT EXISTS sme_doctors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT,
    department TEXT NOT NULL,
    specialty TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 2. Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_sme_doctors_department ON sme_doctors(department);
CREATE INDEX IF NOT EXISTS idx_sme_doctors_active ON sme_doctors(is_active, department);
CREATE INDEX IF NOT EXISTS idx_sme_doctors_name ON sme_doctors(name);

-- 3. Grant permissions to pcesuser (if applicable)
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pcesuser') THEN
        GRANT ALL ON sme_doctors TO pcesuser;
        GRANT USAGE, SELECT ON SEQUENCE sme_doctors_id_seq TO pcesuser;
    END IF;
END $$;

-- 4. Insert sample doctors (3 per department = 90 doctors total)
-- Cardiology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Sarah Chen', 'sarah.chen@hospital.org', 'Cardiology', 'Interventional Cardiology'),
    ('Dr. Michael Roberts', 'michael.roberts@hospital.org', 'Cardiology', 'Electrophysiology'),
    ('Dr. Emily Johnson', 'emily.johnson@hospital.org', 'Cardiology', 'Heart Failure');

-- Neurology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. James Wilson', 'james.wilson@hospital.org', 'Neurology', 'Stroke'),
    ('Dr. Lisa Park', 'lisa.park@hospital.org', 'Neurology', 'Epilepsy'),
    ('Dr. David Martinez', 'david.martinez@hospital.org', 'Neurology', 'Movement Disorders');

-- Diabetes
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Amanda Thompson', 'amanda.thompson@hospital.org', 'Diabetes', 'Type 1 Diabetes'),
    ('Dr. Robert Kim', 'robert.kim@hospital.org', 'Diabetes', 'Type 2 Diabetes'),
    ('Dr. Jennifer Lee', 'jennifer.lee@hospital.org', 'Diabetes', 'Diabetic Complications');

-- Pulmonology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Christopher Brown', 'chris.brown@hospital.org', 'Pulmonology', 'COPD'),
    ('Dr. Michelle Garcia', 'michelle.garcia@hospital.org', 'Pulmonology', 'Asthma'),
    ('Dr. Andrew Taylor', 'andrew.taylor@hospital.org', 'Pulmonology', 'Interstitial Lung Disease');

-- Gastroenterology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Patricia White', 'patricia.white@hospital.org', 'Gastroenterology', 'IBD'),
    ('Dr. Daniel Anderson', 'daniel.anderson@hospital.org', 'Gastroenterology', 'Hepatology'),
    ('Dr. Rachel Moore', 'rachel.moore@hospital.org', 'Gastroenterology', 'Endoscopy');

-- Nephrology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Steven Clark', 'steven.clark@hospital.org', 'Nephrology', 'Dialysis'),
    ('Dr. Laura Rodriguez', 'laura.rodriguez@hospital.org', 'Nephrology', 'Transplant'),
    ('Dr. Kevin Wright', 'kevin.wright@hospital.org', 'Nephrology', 'Glomerular Disease');

-- Oncology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Nancy Lewis', 'nancy.lewis@hospital.org', 'Oncology', 'Breast Cancer'),
    ('Dr. Thomas Hall', 'thomas.hall@hospital.org', 'Oncology', 'Lung Cancer'),
    ('Dr. Karen Young', 'karen.young@hospital.org', 'Oncology', 'Hematologic Oncology');

-- Hematology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Brian King', 'brian.king@hospital.org', 'Hematology', 'Coagulation'),
    ('Dr. Susan Scott', 'susan.scott@hospital.org', 'Hematology', 'Anemia'),
    ('Dr. Joseph Green', 'joseph.green@hospital.org', 'Hematology', 'Blood Disorders');

-- Orthopedics
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Elizabeth Adams', 'elizabeth.adams@hospital.org', 'Orthopedics', 'Joint Replacement'),
    ('Dr. Richard Baker', 'richard.baker@hospital.org', 'Orthopedics', 'Sports Medicine'),
    ('Dr. Maria Nelson', 'maria.nelson@hospital.org', 'Orthopedics', 'Spine Surgery');

-- Dermatology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Charles Hill', 'charles.hill@hospital.org', 'Dermatology', 'Skin Cancer'),
    ('Dr. Dorothy Carter', 'dorothy.carter@hospital.org', 'Dermatology', 'Psoriasis'),
    ('Dr. Frank Mitchell', 'frank.mitchell@hospital.org', 'Dermatology', 'Dermatitis');

-- Ophthalmology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Helen Perez', 'helen.perez@hospital.org', 'Ophthalmology', 'Retina'),
    ('Dr. George Roberts', 'george.roberts@hospital.org', 'Ophthalmology', 'Glaucoma'),
    ('Dr. Betty Turner', 'betty.turner@hospital.org', 'Ophthalmology', 'Cataract Surgery');

-- Psychiatry
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Edward Phillips', 'edward.phillips@hospital.org', 'Psychiatry', 'Depression'),
    ('Dr. Margaret Campbell', 'margaret.campbell@hospital.org', 'Psychiatry', 'Anxiety Disorders'),
    ('Dr. Ronald Parker', 'ronald.parker@hospital.org', 'Psychiatry', 'Bipolar Disorder');

-- Pediatrics
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Sandra Evans', 'sandra.evans@hospital.org', 'Pediatrics', 'General Pediatrics'),
    ('Dr. Kenneth Edwards', 'kenneth.edwards@hospital.org', 'Pediatrics', 'Neonatology'),
    ('Dr. Carol Collins', 'carol.collins@hospital.org', 'Pediatrics', 'Pediatric Cardiology');

-- Rheumatology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Mark Stewart', 'mark.stewart@hospital.org', 'Rheumatology', 'Rheumatoid Arthritis'),
    ('Dr. Diane Sanchez', 'diane.sanchez@hospital.org', 'Rheumatology', 'Lupus'),
    ('Dr. Paul Morris', 'paul.morris@hospital.org', 'Rheumatology', 'Fibromyalgia');

-- Urology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Angela Rogers', 'angela.rogers@hospital.org', 'Urology', 'Prostate'),
    ('Dr. Timothy Reed', 'timothy.reed@hospital.org', 'Urology', 'Kidney Stones'),
    ('Dr. Sharon Cook', 'sharon.cook@hospital.org', 'Urology', 'Urologic Oncology');

-- Immunology & Allergy
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Larry Morgan', 'larry.morgan@hospital.org', 'Immunology & Allergy', 'Food Allergies'),
    ('Dr. Virginia Bell', 'virginia.bell@hospital.org', 'Immunology & Allergy', 'Asthma/Allergy'),
    ('Dr. Raymond Murphy', 'raymond.murphy@hospital.org', 'Immunology & Allergy', 'Immunodeficiency');

-- Infectious Disease
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Joyce Bailey', 'joyce.bailey@hospital.org', 'Infectious Disease', 'HIV/AIDS'),
    ('Dr. Dennis Rivera', 'dennis.rivera@hospital.org', 'Infectious Disease', 'Hospital Infections'),
    ('Dr. Judith Cooper', 'judith.cooper@hospital.org', 'Infectious Disease', 'Tropical Diseases');

-- Emergency Medicine
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Gerald Richardson', 'gerald.richardson@hospital.org', 'Emergency Medicine', 'Trauma'),
    ('Dr. Teresa Cox', 'teresa.cox@hospital.org', 'Emergency Medicine', 'Critical Care'),
    ('Dr. Jerry Howard', 'jerry.howard@hospital.org', 'Emergency Medicine', 'Toxicology');

-- Hepatology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Debra Ward', 'debra.ward@hospital.org', 'Hepatology', 'Viral Hepatitis'),
    ('Dr. Wayne Torres', 'wayne.torres@hospital.org', 'Hepatology', 'Liver Transplant'),
    ('Dr. Gloria Peterson', 'gloria.peterson@hospital.org', 'Hepatology', 'Cirrhosis');

-- Nutrition & Dietetics
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Roy Gray', 'roy.gray@hospital.org', 'Nutrition & Dietetics', 'Clinical Nutrition'),
    ('Dr. Alice Ramirez', 'alice.ramirez@hospital.org', 'Nutrition & Dietetics', 'Obesity Medicine'),
    ('Dr. Eugene James', 'eugene.james@hospital.org', 'Nutrition & Dietetics', 'Metabolic Disorders');

-- Pharmacology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Ann Watson', 'ann.watson@hospital.org', 'Pharmacology', 'Clinical Pharmacology'),
    ('Dr. Russell Brooks', 'russell.brooks@hospital.org', 'Pharmacology', 'Drug Interactions'),
    ('Dr. Frances Kelly', 'frances.kelly@hospital.org', 'Pharmacology', 'Pharmacokinetics');

-- General Medicine
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Jack Sanders', 'jack.sanders@hospital.org', 'General Medicine', 'Internal Medicine'),
    ('Dr. Ruby Price', 'ruby.price@hospital.org', 'General Medicine', 'Primary Care'),
    ('Dr. Albert Bennett', 'albert.bennett@hospital.org', 'General Medicine', 'Preventive Medicine');

-- ENT (Otolaryngology)
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Phyllis Wood', 'phyllis.wood@hospital.org', 'ENT (Otolaryngology)', 'Head & Neck Surgery'),
    ('Dr. Jesse Barnes', 'jesse.barnes@hospital.org', 'ENT (Otolaryngology)', 'Hearing Disorders'),
    ('Dr. Lillian Ross', 'lillian.ross@hospital.org', 'ENT (Otolaryngology)', 'Sinus Surgery');

-- Obstetrics & Gynecology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Howard Henderson', 'howard.henderson@hospital.org', 'Obstetrics & Gynecology', 'High-Risk Pregnancy'),
    ('Dr. Jean Coleman', 'jean.coleman@hospital.org', 'Obstetrics & Gynecology', 'Reproductive Medicine'),
    ('Dr. Arthur Jenkins', 'arthur.jenkins@hospital.org', 'Obstetrics & Gynecology', 'Gynecologic Oncology');

-- Radiology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Catherine Perry', 'catherine.perry@hospital.org', 'Radiology', 'Diagnostic Radiology'),
    ('Dr. Henry Powell', 'henry.powell@hospital.org', 'Radiology', 'Interventional Radiology'),
    ('Dr. Ruth Long', 'ruth.long@hospital.org', 'Radiology', 'Neuroradiology');

-- Anesthesiology
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Philip Patterson', 'philip.patterson@hospital.org', 'Anesthesiology', 'Cardiac Anesthesia'),
    ('Dr. Marie Hughes', 'marie.hughes@hospital.org', 'Anesthesiology', 'Pediatric Anesthesia'),
    ('Dr. Ralph Flores', 'ralph.flores@hospital.org', 'Anesthesiology', 'Pain Management');

-- Pain Management
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Evelyn Washington', 'evelyn.washington@hospital.org', 'Pain Management', 'Chronic Pain'),
    ('Dr. Louis Butler', 'louis.butler@hospital.org', 'Pain Management', 'Interventional Pain'),
    ('Dr. Mildred Simmons', 'mildred.simmons@hospital.org', 'Pain Management', 'Cancer Pain');

-- Sleep Medicine
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Johnny Foster', 'johnny.foster@hospital.org', 'Sleep Medicine', 'Sleep Apnea'),
    ('Dr. Doris Gonzales', 'doris.gonzales@hospital.org', 'Sleep Medicine', 'Insomnia'),
    ('Dr. Earl Bryant', 'earl.bryant@hospital.org', 'Sleep Medicine', 'Narcolepsy');

-- Sports Medicine
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Martha Alexander', 'martha.alexander@hospital.org', 'Sports Medicine', 'Athletic Injuries'),
    ('Dr. Bruce Russell', 'bruce.russell@hospital.org', 'Sports Medicine', 'Rehabilitation'),
    ('Dr. Irene Griffin', 'irene.griffin@hospital.org', 'Sports Medicine', 'Performance Medicine');

-- Geriatrics
INSERT INTO sme_doctors (name, email, department, specialty) VALUES
    ('Dr. Stanley Diaz', 'stanley.diaz@hospital.org', 'Geriatrics', 'Dementia Care'),
    ('Dr. Hazel Hayes', 'hazel.hayes@hospital.org', 'Geriatrics', 'Palliative Care'),
    ('Dr. Fred Myers', 'fred.myers@hospital.org', 'Geriatrics', 'Geriatric Assessment');

-- Verify insertion
SELECT department, COUNT(*) as doctor_count 
FROM sme_doctors 
GROUP BY department 
ORDER BY department;
