#!/usr/bin/env python3
"""
PostgreSQL Database Connection Guide
====================================

This script demonstrates multiple ways to connect to your PostgreSQL databases.
Based on your .env configuration, you have two databases available:

1. pces_base (legacy database)
2. pces_ehr_ccm (diagnosis tool database with p_diagnosis table)
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_legacy_database_connection():
    """Test connection to the legacy pces_base database"""
    print("🔗 CONNECTING TO LEGACY DATABASE (pces_base)")
    print("=" * 50)
    
    try:
        # Legacy database connection parameters
        conn_params = {
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD')
        }
        
        print(f"Host: {conn_params['host']}")
        print(f"Port: {conn_params['port']}")
        print(f"Database: {conn_params['database']}")
        print(f"User: {conn_params['user']}")
        print(f"Password: {'*' * len(conn_params['password'])}")
        
        # Establish connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connection successful!")
        print(f"PostgreSQL version: {version['version']}")
        
        # List tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"📊 Available tables: {len(tables)}")
        for table in tables[:10]:  # Show first 10 tables
            print(f"   - {table['table_name']}")
        if len(tables) > 10:
            print(f"   ... and {len(tables) - 10} more tables")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

def test_diagnosis_database_connection():
    """Test connection to the diagnosis database (pces_ehr_ccm)"""
    print("\n\n🏥 CONNECTING TO DIAGNOSIS DATABASE (pces_ehr_ccm)")
    print("=" * 50)
    
    try:
        # Diagnosis database connection parameters
        conn_params = {
            'host': os.getenv('PG_TOOL_HOST'),
            'port': os.getenv('PG_TOOL_PORT', '5432'),
            'database': os.getenv('PG_TOOL_NAME'),
            'user': os.getenv('PG_TOOL_USER'),
            'password': os.getenv('PG_TOOL_PASSWORD')
        }
        
        print(f"Host: {conn_params['host']}")
        print(f"Port: {conn_params['port']}")
        print(f"Database: {conn_params['database']}")
        print(f"User: {conn_params['user']}")
        print(f"Password: {'*' * len(conn_params['password'])}")
        
        # Establish connection
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Connection successful!")
        print(f"PostgreSQL version: {version['version']}")
        
        # Check p_diagnosis table specifically
        cursor.execute("""
            SELECT COUNT(*) as total_records
            FROM p_diagnosis 
            WHERE is_active = TRUE AND deleted_at IS NULL;
        """)
        count = cursor.fetchone()
        print(f"📋 Active diagnosis records: {count['total_records']}")
        
        # Sample diagnosis data
        cursor.execute("""
            SELECT diagnosis_id, code, description, created_at
            FROM p_diagnosis 
            WHERE is_active = TRUE AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT 3;
        """)
        samples = cursor.fetchall()
        print(f"📄 Sample diagnosis records:")
        for i, sample in enumerate(samples, 1):
            print(f"   {i}. Code: {sample['code']}, Description: {sample['description']}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    return True

def connection_examples():
    """Show different ways to connect to PostgreSQL"""
    print("\n\n🔧 CONNECTION METHODS & EXAMPLES")
    print("=" * 50)
    
    print("1️⃣ **Using psycopg2 directly:**")
    print("""
import psycopg2
from psycopg2.extras import RealDictCursor

# For diagnosis database (recommended)
conn = psycopg2.connect(
    host="4.155.102.23",
    port="5432", 
    database="pces_ehr_ccm",
    user="pcesuser",
    password="Pcesuser101",
    cursor_factory=RealDictCursor
)
cursor = conn.cursor()
cursor.execute("SELECT * FROM p_diagnosis LIMIT 5;")
results = cursor.fetchall()
cursor.close()
conn.close()
""")

    print("\n2️⃣ **Using environment variables:**")
    print("""
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('PG_TOOL_HOST'),
    port=os.getenv('PG_TOOL_PORT'),
    database=os.getenv('PG_TOOL_NAME'),
    user=os.getenv('PG_TOOL_USER'),
    password=os.getenv('PG_TOOL_PASSWORD')
)
""")

    print("\n3️⃣ **Using your existing PostgreSQL tool:**")
    print("""
from postgres_tool import postgres_tool

# Get all diagnoses
result = postgres_tool.fetch_diagnosis_descriptions(limit=10)
print(result['content'])

# Search for specific diagnosis
result = postgres_tool.search_diagnosis_by_keyword("diabetes", limit=5)
print(result['summary'])
""")

    print("\n4️⃣ **Using psql command line:**")
    print("""
# Connect to diagnosis database
psql -h 4.155.102.23 -p 5432 -U pcesuser -d pces_ehr_ccm

# Connect to legacy database  
psql -h 4.155.102.23 -p 5432 -U pcesuser -d pces_base

# Once connected, you can run SQL queries:
SELECT COUNT(*) FROM p_diagnosis;
SELECT * FROM p_diagnosis LIMIT 5;
""")

def database_schema_info():
    """Show p_diagnosis table schema"""
    print("\n\n📊 P_DIAGNOSIS TABLE SCHEMA")
    print("=" * 50)
    
    try:
        conn = psycopg2.connect(
            host=os.getenv('PG_TOOL_HOST'),
            port=os.getenv('PG_TOOL_PORT'),
            database=os.getenv('PG_TOOL_NAME'),
            user=os.getenv('PG_TOOL_USER'),
            password=os.getenv('PG_TOOL_PASSWORD')
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get table schema
        cursor.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'p_diagnosis'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        
        print("Table: p_diagnosis")
        print("-" * 30)
        for col in columns:
            nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
            default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
            print(f"  {col['column_name']:<20} {col['data_type']:<25} {nullable}{default}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error getting schema: {e}")

def main():
    """Main function to test all connections"""
    print("🔗 POSTGRESQL DATABASE CONNECTION GUIDE")
    print("=" * 60)
    print("This script will test connections to both of your PostgreSQL databases.")
    print()
    
    # Test both database connections
    legacy_success = test_legacy_database_connection()
    diagnosis_success = test_diagnosis_database_connection()
    
    # Show connection examples
    connection_examples()
    
    # Show database schema
    if diagnosis_success:
        database_schema_info()
    
    print("\n\n" + "=" * 60)
    print("📋 CONNECTION SUMMARY")
    print("=" * 60)
    print(f"Legacy Database (pces_base): {'✅ Connected' if legacy_success else '❌ Failed'}")
    print(f"Diagnosis Database (pces_ehr_ccm): {'✅ Connected' if diagnosis_success else '❌ Failed'}")
    print()
    print("📝 Next Steps:")
    if diagnosis_success:
        print("  ✅ Your diagnosis database is ready to use!")
        print("  ✅ Use the examples above to connect in your code")
        print("  ✅ The PostgreSQL tool is configured and working")
    else:
        print("  ❌ Check your database credentials and network connectivity")
        print("  ❌ Ensure PostgreSQL server is running and accessible")

if __name__ == "__main__":
    main()