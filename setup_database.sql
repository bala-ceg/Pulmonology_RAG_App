-- Setup Local PostgreSQL Database for PCES Application
-- Create database and tables with sample data

-- Create the database (run this as superuser first)
-- CREATE DATABASE pces_local;

-- Connect to pces_local database and run the following:

-- Create pces_users table
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
);

-- Create patient table
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
);

-- Insert sample doctors into pces_users table
INSERT INTO pces_users (username, password_hash, pces_role, first_name, last_name, middle_name, email, phone, addr1, city, state, zipcode, country, comments) VALUES
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
('mgarcia', '$2b$12$hash15', 'Doctor', 'Maria', 'Garcia', 'Isabel', 'maria.garcia@hospital.com', '555-0115', '468 Wellness Rd', 'Columbus', 'OH', '43201', 'USA', 'Emergency Medicine');

-- Insert sample patients into patient table
INSERT INTO patient (first_name, last_name, middle_name, email, phone, addr1, city, state, zipcode, country, comments) VALUES
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
('Amy', 'Young', 'Dawn', 'amy.young@email.com', '555-1020', '2000 Olive Lane', 'Woburn', 'MA', '01801', 'USA', 'Pediatric patient, routine vaccinations');
