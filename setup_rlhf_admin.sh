#!/bin/bash

# RLHF Admin Panel - Quick Setup Script
# This script helps you set up and run the RLHF admin panel

echo "=================================================="
echo "🎓 RLHF Admin Panel - Quick Setup"
echo "=================================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env with database credentials:"
    echo "  DB_HOST=your_host"
    echo "  DB_PORT=5432"
    echo "  DB_NAME=pces_base"
    echo "  DB_USER=your_user"
    echo "  DB_PASSWORD=your_password"
    exit 1
fi

echo "✓ Found .env file"
echo ""

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    echo "Please run this script from the Pulmonology_RAG_App directory"
    exit 1
fi

echo "✓ Found main.py"
echo ""

# Menu
echo "What would you like to do?"
echo ""
echo "1) Generate sample RLHF training data"
echo "2) Start Flask application (with admin panel)"
echo "3) Both (generate data then start app)"
echo "4) Exit"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Generating sample data..."
        python3 generate_rlhf_samples.py
        echo ""
        echo "✓ Sample data generated!"
        echo ""
        echo "To access the admin panel, run:"
        echo "  python main.py"
        echo "Then open: http://localhost:3000/admin/rlhf"
        ;;
    2)
        echo ""
        echo "Starting Flask application..."
        echo "Admin panel will be available at: http://localhost:3000/admin/rlhf"
        echo ""
        python3 main.py
        ;;
    3)
        echo ""
        echo "Step 1: Generating sample data..."
        python3 generate_rlhf_samples.py
        echo ""
        echo "✓ Sample data generated!"
        echo ""
        echo "Step 2: Starting Flask application..."
        echo "Admin panel will be available at: http://localhost:3000/admin/rlhf"
        echo ""
        python3 main.py
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
