#!/usr/bin/env python3
"""
Database Schema Inspector
=========================

This script inspects the p_diagnosis table structure to understand
the actual column names available.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def inspect_database_schema():
    """Inspect the p_diagnosis table structure"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            cursor_factory=RealDictCursor
        )
        
        with conn.cursor() as cursor:
            # Get table structure
            print("🔍 Inspecting p_diagnosis table structure...")
            print("=" * 50)
            
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = 'p_diagnosis'
                ORDER BY ordinal_position;
            """)
            
            columns = cursor.fetchall()
            
            if columns:
                print("📋 Table columns found:")
                for col in columns:
                    print(f"   - {col['column_name']} ({col['data_type']}) - Nullable: {col['is_nullable']}")
            else:
                print("❌ No columns found or table doesn't exist")
                return
            
            # Get sample data
            print("\n📊 Sample data from p_diagnosis table:")
            print("=" * 50)
            
            # Try to get first few rows with any columns that exist
            cursor.execute("SELECT * FROM p_diagnosis LIMIT 3;")
            sample_data = cursor.fetchall()
            
            if sample_data:
                print(f"Found {len(sample_data)} sample records:")
                for i, row in enumerate(sample_data, 1):
                    print(f"\n   Record {i}:")
                    for key, value in row.items():
                        print(f"     {key}: {value}")
            else:
                print("❌ No data found in table")
            
            # Check if description column exists and has data
            print("\n🔍 Focusing on description column...")
            print("=" * 50)
            
            if any(col['column_name'] == 'description' for col in columns):
                cursor.execute("SELECT description FROM p_diagnosis WHERE description IS NOT NULL LIMIT 5;")
                descriptions = cursor.fetchall()
                
                if descriptions:
                    print("Sample descriptions found:")
                    for i, desc in enumerate(descriptions, 1):
                        print(f"   {i}. {desc['description'][:100]}...")
                else:
                    print("❌ No descriptions found")
            else:
                print("❌ Description column not found")
    
    except Exception as e:
        print(f"❌ Error inspecting database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_database_schema()