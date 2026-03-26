#!/usr/bin/env python3
"""
Local PostgreSQL Database Setup Script for PCES Application
This script creates a local PostgreSQL database and populates it with sample data.
"""

import psycopg
import sys
import os

def create_database():
    """Create the local PCES database."""
    
    # Default PostgreSQL connection (usually to 'postgres' database)
    default_config = {
        "host": "localhost",
        "port": "5432",
        "dbname": "postgres",  # Connect to default postgres db first
        "user": os.getenv("USER", "bseetharaman"),    # Use current user
        "password": ""  # No password needed for local user
    }
    
    try:
        print("Connecting to PostgreSQL server...")
        with psycopg.connect(**default_config) as conn:
            # Set autocommit to create database
            conn.autocommit = True
            with conn.cursor() as cursor:
                # Check if database exists
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'pces_local'")
                if cursor.fetchone():
                    print("Database 'pces_local' already exists.")
                else:
                    print("Creating database 'pces_local'...")
                    cursor.execute("CREATE DATABASE pces_local")
                    print("Database 'pces_local' created successfully!")
                    
    except psycopg.Error as e:
        print(f"Error creating database: {e}")
        return False
    
    return True

def setup_tables_and_data():
    """Setup tables and insert sample data."""
    
    # Configuration for the new database
    local_config = {
        "host": "localhost",
        "port": "5432",
        "dbname": "pces_local",
        "user": os.getenv("USER", "bseetharaman"),
        "password": ""  # No password needed for local user
    }
    
    try:
        print("Connecting to pces_local database...")
        with psycopg.connect(**local_config) as conn:
            with conn.cursor() as cursor:
                
                # Read and execute the SQL file
                sql_file_path = os.path.join(os.path.dirname(__file__), 'setup_database.sql')
                
                if not os.path.exists(sql_file_path):
                    print(f"SQL file not found: {sql_file_path}")
                    return False
                
                print("Reading SQL setup file...")
                with open(sql_file_path, 'r') as file:
                    sql_content = file.read()
                
                # Split by CREATE and INSERT statements and execute them
                print("Creating tables...")
                
                # Create pces_users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pces_users (
                        user_id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        pces_role VARCHAR(20),
                        first_name VARCHAR(50),
                        last_name VARCHAR(50),
                        middle_name VARCHAR(50),
                        email VARCHAR(100) UNIQUE,
                        phone VARCHAR(20),
                        addr1 VARCHAR(50),
                        addr2 VARCHAR(50),
                        city VARCHAR(50),
                        state VARCHAR(50),
                        zipcode VARCHAR(20),
                        country VARCHAR(20),
                        comments VARCHAR(200),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create patient table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS patient (
                        patient_id SERIAL PRIMARY KEY,
                        first_name VARCHAR(50),
                        last_name VARCHAR(50),
                        middle_name VARCHAR(50),
                        email VARCHAR(100) UNIQUE,
                        phone VARCHAR(20),
                        addr1 VARCHAR(50),
                        addr2 VARCHAR(50),
                        city VARCHAR(50),
                        state VARCHAR(50),
                        zipcode VARCHAR(20),
                        country VARCHAR(20),
                        comments VARCHAR(1000),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                print("Tables created successfully!")
                
                # Insert sample doctors
                print("Inserting sample doctors...")
                doctors_data = [
                    ('ssmith', '$2b$12$hash1', 'Doctor', 'Sarah', 'Smith', 'Jane', 'sarah.smith@hospital.com', '555-0101', '123 Medical Ave', 'Boston', 'MA', '02101', 'USA', 'Cardiologist'),
                    ('mjohnson', '$2b$12$hash2', 'Doctor', 'Michael', 'Johnson', 'Robert', 'michael.johnson@hospital.com', '555-0102', '456 Health St', 'New York', 'NY', '10001', 'USA', 'Pulmonologist'),
                    ('awilliams', '$2b$12$hash3', 'Doctor', 'Amanda', 'Williams', 'Grace', 'amanda.williams@hospital.com', '555-0103', '789 Care Blvd', 'Chicago', 'IL', '60601', 'USA', 'Neurologist'),
                    ('dbrown', '$2b$12$hash4', 'Doctor', 'David', 'Brown', 'Lee', 'david.brown@hospital.com', '555-0104', '321 Wellness Dr', 'Los Angeles', 'CA', '90001', 'USA', 'Family Medicine'),
                    ('ejones', '$2b$12$hash5', 'Doctor', 'Emily', 'Jones', 'Marie', 'emily.jones@hospital.com', '555-0105', '654 Hospital Rd', 'Houston', 'TX', '77001', 'USA', 'Pediatrician'),
                    ('rmiller', '$2b$12$hash6', 'Doctor', 'Robert', 'Miller', 'James', 'robert.miller@hospital.com', '555-0106', '987 Medical Plaza', 'Phoenix', 'AZ', '85001', 'USA', 'Orthopedic Surgeon'),
                    ('ldavis', '$2b$12$hash7', 'Doctor', 'Linda', 'Davis', 'Ann', 'linda.davis@hospital.com', '555-0107', '147 Health Center', 'Philadelphia', 'PA', '19101', 'USA', 'Dermatologist'),
                    ('jwilson', '$2b$12$hash8', 'Doctor', 'James', 'Wilson', 'Thomas', 'james.wilson@hospital.com', '555-0108', '258 Care Lane', 'San Antonio', 'TX', '78201', 'USA', 'Gastroenterologist'),
                    ('kmurray', '$2b$12$hash9', 'Doctor', 'Karen', 'Murray', 'Elizabeth', 'karen.murray@hospital.com', '555-0109', '369 Medical Way', 'San Diego', 'CA', '92101', 'USA', 'Oncologist'),
                    ('ctaylor', '$2b$12$hash10', 'Doctor', 'Christopher', 'Taylor', 'Michael', 'christopher.taylor@hospital.com', '555-0110', '741 Wellness Ave', 'Dallas', 'TX', '75201', 'USA', 'Psychiatrist'),
                    ('sanderson', '$2b$12$hash11', 'Doctor', 'Susan', 'Anderson', 'Rose', 'susan.anderson@hospital.com', '555-0111', '852 Health St', 'San Jose', 'CA', '95101', 'USA', 'Endocrinologist'),
                    ('twhite', '$2b$12$hash12', 'Doctor', 'Thomas', 'White', 'Andrew', 'thomas.white@hospital.com', '555-0112', '963 Medical Dr', 'Austin', 'TX', '73301', 'USA', 'Urologist'),
                    ('nmartin', '$2b$12$hash13', 'Doctor', 'Nancy', 'Martin', 'Claire', 'nancy.martin@hospital.com', '555-0113', '159 Care Circle', 'Jacksonville', 'FL', '32201', 'USA', 'Rheumatologist'),
                    ('pthompson', '$2b$12$hash14', 'Doctor', 'Paul', 'Thompson', 'Daniel', 'paul.thompson@hospital.com', '555-0114', '357 Hospital Blvd', 'Fort Worth', 'TX', '76101', 'USA', 'Anesthesiologist'),
                    ('mgarcia', '$2b$12$hash15', 'Doctor', 'Maria', 'Garcia', 'Isabel', 'maria.garcia@hospital.com', '555-0115', '468 Wellness Rd', 'Columbus', 'OH', '43201', 'USA', 'Emergency Medicine'),
                ]
                
                cursor.executemany("""
                    INSERT INTO pces_users (username, password_hash, pces_role, first_name, last_name, middle_name, email, phone, addr1, city, state, zipcode, country, comments) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, doctors_data)
                
                # Insert sample patients
                print("Inserting sample patients...")
                patients_data = [
                    ('John', 'Doe', 'William', 'john.doe@email.com', '555-1001', '100 Oak Street', 'Boston', 'MA', '02101', 'USA', 'Regular checkup patient, no known allergies'),
                    ('Jane', 'Smith', 'Elizabeth', 'jane.smith@email.com', '555-1002', '200 Pine Avenue', 'Cambridge', 'MA', '02139', 'USA', 'Diabetic patient, requires insulin monitoring'),
                    ('Robert', 'Johnson', 'Michael', 'robert.johnson@email.com', '555-1003', '300 Elm Road', 'Somerville', 'MA', '02143', 'USA', 'Hypertension, takes daily medication'),
                    ('Mary', 'Williams', 'Grace', 'mary.williams@email.com', '555-1004', '400 Maple Drive', 'Newton', 'MA', '02458', 'USA', 'Pregnant patient, second trimester'),
                    ('James', 'Brown', 'Richard', 'james.brown@email.com', '555-1005', '500 Cedar Lane', 'Brookline', 'MA', '02445', 'USA', 'Post-surgery recovery, knee replacement'),
                    ('Patricia', 'Davis', 'Ann', 'patricia.davis@email.com', '555-1006', '600 Birch Street', 'Medford', 'MA', '02155', 'USA', 'Asthma patient, uses inhaler'),
                    ('Christopher', 'Miller', 'Joseph', 'christopher.miller@email.com', '555-1007', '700 Spruce Avenue', 'Malden', 'MA', '02148', 'USA', 'Heart patient, scheduled for cardiac stress test'),
                    ('Sarah', 'Wilson', 'Marie', 'sarah.wilson@email.com', '555-1008', '800 Willow Road', 'Everett', 'MA', '02149', 'USA', 'Migraine sufferer, neurological consultation'),
                    ('Daniel', 'Moore', 'Thomas', 'daniel.moore@email.com', '555-1009', '900 Poplar Drive', 'Chelsea', 'MA', '02150', 'USA', 'Arthritis patient, joint pain management'),
                    ('Lisa', 'Taylor', 'Claire', 'lisa.taylor@email.com', '555-1010', '1000 Chestnut Lane', 'Revere', 'MA', '02151', 'USA', 'Mental health patient, anxiety and depression'),
                    ('Mark', 'Anderson', 'David', 'mark.anderson@email.com', '555-1011', '1100 Hickory Street', 'Waltham', 'MA', '02451', 'USA', 'Chronic back pain, physical therapy'),
                    ('Susan', 'Thomas', 'Rose', 'susan.thomas@email.com', '555-1012', '1200 Ash Avenue', 'Arlington', 'MA', '02474', 'USA', 'Skin condition, dermatology follow-up'),
                    ('Paul', 'Jackson', 'Andrew', 'paul.jackson@email.com', '555-1013', '1300 Sycamore Road', 'Belmont', 'MA', '02478', 'USA', 'Digestive issues, gastroenterology referral'),
                    ('Nancy', 'White', 'Lynn', 'nancy.white@email.com', '555-1014', '1400 Beech Drive', 'Watertown', 'MA', '02472', 'USA', 'Cancer survivor, regular oncology check-ups'),
                    ('Kevin', 'Harris', 'Scott', 'kevin.harris@email.com', '555-1015', '1500 Magnolia Lane', 'Lexington', 'MA', '02421', 'USA', 'Sleep apnea, uses CPAP machine'),
                    ('Michelle', 'Clark', 'Beth', 'michelle.clark@email.com', '555-1016', '1600 Dogwood Street', 'Concord', 'MA', '01742', 'USA', 'Thyroid condition, endocrinology treatment'),
                    ('Brian', 'Lewis', 'Patrick', 'brian.lewis@email.com', '555-1017', '1700 Redwood Avenue', 'Lincoln', 'MA', '01773', 'USA', 'Kidney stones, urology consultation'),
                    ('Jennifer', 'Walker', 'Nicole', 'jennifer.walker@email.com', '555-1018', '1800 Sequoia Road', 'Bedford', 'MA', '01730', 'USA', 'Rheumatoid arthritis, immunology treatment'),
                    ('Steven', 'Hall', 'Robert', 'steven.hall@email.com', '555-1019', '1900 Palm Drive', 'Burlington', 'MA', '01803', 'USA', 'Recent surgery, post-operative care'),
                    ('Amy', 'Young', 'Dawn', 'amy.young@email.com', '555-1020', '2000 Olive Lane', 'Woburn', 'MA', '01801', 'USA', 'Pediatric patient, routine vaccinations'),
                ]
                
                cursor.executemany("""
                    INSERT INTO patient (first_name, last_name, middle_name, email, phone, addr1, city, state, zipcode, country, comments) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, patients_data)
                
                # Commit all changes
                conn.commit()
                
                print("Sample data inserted successfully!")
                
                # Verify data
                cursor.execute("SELECT COUNT(*) FROM pces_users")
                doctor_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM patient")
                patient_count = cursor.fetchone()[0]
                
                print(f"Database setup complete!")
                print(f"- Doctors in pces_users table: {doctor_count}")
                print(f"- Patients in patient table: {patient_count}")
                
                return True
                
    except psycopg.Error as e:
        print(f"Error setting up tables and data: {e}")
        return False

def test_connection():
    """Test the database connection and queries."""
    local_config = {
        "host": "localhost",
        "port": "5432",
        "dbname": "pces_local",
        "user": os.getenv("USER", "bseetharaman"),
        "password": ""
    }
    
    try:
        print("\n=== Testing Database Connection ===")
        with psycopg.connect(**local_config) as conn:
            with conn.cursor() as cursor:
                
                # Test doctor search
                print("\nTesting doctor search for 'S':")
                cursor.execute("""
                    SELECT DISTINCT first_name, last_name 
                    FROM pces_users 
                    WHERE LOWER(first_name) LIKE %s 
                       OR LOWER(last_name) LIKE %s 
                       OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                    ORDER BY first_name, last_name
                    LIMIT 5
                """, ('%s%', '%s%', '%s%'))
                
                doctors = cursor.fetchall()
                for doctor in doctors:
                    print(f"  - Dr. {doctor[0]} {doctor[1]}")
                
                # Test patient search
                print("\nTesting patient search for 'J':")
                cursor.execute("""
                    SELECT DISTINCT first_name, last_name 
                    FROM patient 
                    WHERE LOWER(first_name) LIKE %s 
                       OR LOWER(last_name) LIKE %s 
                       OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                    ORDER BY first_name, last_name
                    LIMIT 5
                """, ('%j%', '%j%', '%j%'))
                
                patients = cursor.fetchall()
                for patient in patients:
                    print(f"  - {patient[0]} {patient[1]}")
                
                print("\nDatabase connection test successful!")
                return True
                
    except psycopg.Error as e:
        print(f"Error testing database connection: {e}")
        return False

def main():
    """Main function to set up the database."""
    print("=== PCES Local PostgreSQL Database Setup ===")
    print("This script will create a local PostgreSQL database with sample data.")
    print("\nMake sure you have PostgreSQL installed and running locally.")
    print("Default settings: host=localhost, port=5432, user=postgres")
    print("\nNote: You may need to modify the password in this script.")
    
    # Check if PostgreSQL is available
    try:
        import psycopg
    except ImportError:
        print("\nError: psycopg package not found. Please install it:")
        print("pip install psycopg[binary]")
        return
    
    proceed = input("\nProceed with database setup? (y/N): ")
    if proceed.lower() not in ['y', 'yes']:
        print("Setup cancelled.")
        return
    
    # Create database
    if create_database():
        # Setup tables and data
        if setup_tables_and_data():
            # Test the setup
            test_connection()
            print("\n=== Setup Complete! ===")
            print("You can now update your .env file to use the local database:")
            print("DB_HOST=localhost")
            print("DB_PORT=5432")
            print("DB_NAME=pces_local")
            print("DB_USER=postgres")
            print("DB_PASSWORD=your_postgres_password")
        else:
            print("\nFailed to setup tables and data.")
    else:
        print("\nFailed to create database.")

if __name__ == "__main__":
    main()
