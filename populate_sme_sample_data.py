#!/usr/bin/env python3
"""
Populate sample SME review data with domain field.
This script adds sample prompts to the sft_ranked_data table with domain values
so that SMEs can review them.
"""

import psycopg2
from datetime import datetime
import uuid

# Database connection
DB_CONFIG = {
    'host': '4.155.102.23',
    'port': 5432,
    'database': 'pces_base',
    'user': 'pcesuser',
    'password': 'your_password_here'  # Update this
}

SAMPLE_DATA = [
    {
        'domain': 'Cardiology',
        'prompt': 'Effect of COVID to heart patient male 70+ based out of Nigeria',
        'responses': [
            {
                'rank': 1,
                'text': 'There is direct effect on 70+ male and female population in Africa (including Nigeria) that lead to chronic diseases. This is very much prevalent in Ghettos where cleanliness are compromised.',
                'reason': 'Comprehensive response covering demographics, geography, and social determinants'
            },
            {
                'rank': 2,
                'text': 'No records found on Covid effecting Heart patient. It is highly recommended to be aware of COVID repercussion.',
                'reason': 'Lacks specific information but provides cautionary advice'
            },
            {
                'rank': 3,
                'text': 'Decent number of cases found with 70+ heart patient in Egypt and Libya because of COVID. It started with Pneumonia and if not proper care taken at initial stage may lead to Fatal.',
                'reason': 'Focuses on different geographic region, not Nigeria-specific'
            }
        ]
    },
    {
        'domain': 'Pulmonology',
        'prompt': 'What are the early warning signs of COPD in patients over 50?',
        'responses': [
            {
                'rank': 1,
                'text': 'Early warning signs include persistent cough (often called "smoker\'s cough"), increased breathlessness during routine activities, frequent chest infections, and wheezing. Risk factors include smoking history, occupational exposure to dust or chemicals, and genetic factors like Alpha-1 antitidase deficiency.',
                'reason': 'Complete clinical picture with risk factors and specific symptoms'
            },
            {
                'rank': 2,
                'text': 'Chronic cough and shortness of breath are the main signs. Patients may also experience fatigue and weight loss in advanced stages.',
                'reason': 'Covers main symptoms but lacks detail on early detection'
            }
        ]
    },
    {
        'domain': 'Diabetes',
        'prompt': 'How should a Type-2 Diabetes patient manage blood sugar during fasting?',
        'responses': [
            {
                'rank': 1,
                'text': 'Type-2 Diabetes patients planning to fast should consult their physician 6-8 weeks before Ramadan. Management includes adjusting medication timing, monitoring blood glucose 4-5 times daily, breaking fast immediately if glucose <70 mg/dL or >300 mg/dL, maintaining hydration during non-fasting hours, and choosing low-GI foods at Suhoor.',
                'reason': 'Evidence-based approach with specific clinical parameters and culturally sensitive'
            },
            {
                'rank': 2,
                'text': 'Patients should monitor blood sugar regularly and adjust insulin doses. They should eat healthy foods and stay hydrated.',
                'reason': 'Generic advice without specific clinical guidelines'
            },
            {
                'rank': 3,
                'text': 'Fasting is generally not recommended for diabetic patients.',
                'reason': 'Overly restrictive without considering individual patient circumstances'
            }
        ]
    },
    {
        'domain': 'Rheumatology',
        'prompt': 'What is the first-line treatment for newly diagnosed rheumatoid arthritis?',
        'responses': [
            {
                'rank': 1,
                'text': 'Methotrexate is the first-line DMARD for rheumatoid arthritis, typically started at 10-15 mg weekly with folic acid supplementation. NSAIDs and low-dose corticosteroids may be used for symptom control. Early aggressive treatment within 3 months of symptom onset improves long-term outcomes. Regular monitoring includes CBC, liver function tests, and creatinine.',
                'reason': 'Evidence-based standard of care with monitoring protocols'
            },
            {
                'rank': 2,
                'text': 'DMARDs like methotrexate are used along with anti-inflammatory medications.',
                'reason': 'Correct but lacks dosing and monitoring details'
            }
        ]
    },
    {
        'domain': 'Cardiology',
        'prompt': 'What lifestyle modifications are recommended post-myocardial infarction?',
        'responses': [
            {
                'rank': 1,
                'text': 'Post-MI lifestyle modifications include: smoking cessation (most critical), cardiac rehabilitation program, Mediterranean diet, limiting sodium to <2g/day, regular exercise (150 min/week moderate intensity after clearance), weight management (BMI <25), stress reduction, limiting alcohol to 1-2 drinks/day, and strict medication adherence for secondary prevention.',
                'reason': 'Comprehensive evidence-based recommendations with specific targets'
            },
            {
                'rank': 2,
                'text': 'Patients should quit smoking, eat healthy, exercise regularly, and take their medications.',
                'reason': 'Correct general direction but lacks specificity'
            },
            {
                'rank': 3,
                'text': 'Rest and avoid all strenuous activities.',
                'reason': 'Outdated advice; cardiac rehabilitation is now standard'
            }
        ]
    },
    {
        'domain': 'Pulmonology',
        'prompt': 'When should asthma patients use their rescue inhaler versus controller medication?',
        'responses': [
            {
                'rank': 1,
                'text': 'Rescue inhalers (short-acting beta-agonists like albuterol) are used for acute symptoms and before exercise. If needed >2 times/week, asthma is poorly controlled. Controller medications (inhaled corticosteroids) are taken daily regardless of symptoms to prevent attacks. Using rescue inhaler >2 times/week indicates need to step up controller therapy.',
                'reason': 'Clear distinction with clinical indicators for treatment adjustment'
            },
            {
                'rank': 2,
                'text': 'Rescue inhalers are for quick relief, controllers are for prevention.',
                'reason': 'Correct basic concept but lacks actionable guidance'
            }
        ]
    }
]


def populate_sme_review_data():
    """Insert sample data into sft_ranked_data table with domains."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("🚀 Starting to populate SME review data...\n")
        
        for item in SAMPLE_DATA:
            group_id = f"group_{uuid.uuid4().hex[:8]}"
            domain = item['domain']
            prompt = item['prompt']
            
            print(f"📝 Adding prompt for {domain}:")
            print(f"   {prompt[:80]}...")
            
            for response in item['responses']:
                cur.execute("""
                    INSERT INTO sft_ranked_data 
                    (prompt, response_text, rank, reason, group_id, domain, created_at, updated_at, created_by, updated_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    prompt,
                    response['text'],
                    response['rank'],
                    response['reason'],
                    group_id,
                    domain,
                    datetime.now(),
                    datetime.now(),
                    1001,  # System user
                    1001
                ))
                print(f"   ✓ Added Rank {response['rank']} response")
            
            print()
        
        conn.commit()
        
        # Show summary
        cur.execute("SELECT COUNT(DISTINCT group_id) FROM sft_ranked_data WHERE domain IS NOT NULL")
        prompt_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM sft_ranked_data WHERE domain IS NOT NULL")
        total_count = cur.fetchone()[0]
        
        cur.execute("SELECT domain, COUNT(*) FROM sft_ranked_data WHERE domain IS NOT NULL GROUP BY domain")
        by_domain = cur.fetchall()
        
        print("✅ Data population complete!")
        print(f"\n📊 Summary:")
        print(f"   Total prompts: {prompt_count}")
        print(f"   Total responses: {total_count}")
        print(f"\n   By domain:")
        for domain, count in by_domain:
            print(f"   - {domain}: {count} responses")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("SME Review Sample Data Populator")
    print("=" * 60)
    print()
    
    # Check if user wants to proceed
    response = input("This will add sample data to sft_ranked_data table. Continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        populate_sme_review_data()
    else:
        print("Cancelled.")
