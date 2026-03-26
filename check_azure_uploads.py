#!/usr/bin/env python3
"""
Script to check Azure Blob Storage uploads for PCES application
Run this script to see all files uploaded to Azure
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from azure_storage import get_storage_manager
    AZURE_AVAILABLE = True
except ImportError as e:
    print(f"Azure storage not available: {e}")
    AZURE_AVAILABLE = False
    sys.exit(1)

def format_file_size(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def check_azure_connection():
    """Check if Azure connection is working"""
    print("🔧 Checking Azure Storage connection...")
    
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found in environment variables")
        print("   Please check your .env file")
        return False
    
    try:
        storage_manager = get_storage_manager()
        print("✅ Azure Storage connection successful")
        
        # List available containers
        print("\n📦 Available containers:")
        for container in storage_manager.blob_service_client.list_containers():
            print(f"   - {container.name}")
        
        return True
    except Exception as e:
        print(f"❌ Azure connection failed: {e}")
        return False

def check_research_files():
    """Check research PDF files"""
    print("\n🔬 Research Files (Chat PDFs):")
    print("   Location: contoso/pces/documents/research/")
    
    try:
        storage_manager = get_storage_manager()
        files = storage_manager.list_files_in_container("contoso", "pces/documents/research/")
        
        if not files:
            print("   ❌ No research files found")
            return
        
        print(f"   ✅ Found {len(files)} research files:")
        for file in files:
            size = format_file_size(file['size'])
            modified = file['last_modified'].strftime("%Y-%m-%d %H:%M:%S")
            print(f"   📄 {file['name']}")
            print(f"      Size: {size} | Modified: {modified}")
            if file['metadata']:
                print(f"      Metadata: {file['metadata']}")
            print()
            
    except Exception as e:
        print(f"   ❌ Error checking research files: {e}")

def check_patient_summary_files():
    """Check patient summary PDF files"""
    print("\n👤 Patient Summary Files (Patient Notes PDFs):")
    print("   Location: contoso/pces/documents/doc-patient-summary/")
    
    try:
        storage_manager = get_storage_manager()
        files = storage_manager.list_files_in_container("contoso", "pces/documents/doc-patient-summary/")
        
        if not files:
            print("   ❌ No patient summary files found")
            return
        
        print(f"   ✅ Found {len(files)} patient summary files:")
        for file in files:
            size = format_file_size(file['size'])
            modified = file['last_modified'].strftime("%Y-%m-%d %H:%M:%S")
            print(f"   📄 {file['name']}")
            print(f"      Size: {size} | Modified: {modified}")
            if file['metadata']:
                print(f"      Metadata: {file['metadata']}")
            print()
            
    except Exception as e:
        print(f"   ❌ Error checking patient summary files: {e}")

def check_conversation_files():
    """Check conversation PDF files"""
    print("\n💬 Conversation Files (Doctor-Patient Conversation PDFs):")
    print("   Location: contoso/pces/documents/conversations/")
    
    try:
        storage_manager = get_storage_manager()
        files = storage_manager.list_files_in_container("contoso", "pces/documents/conversations/")
        
        if not files:
            print("   ❌ No conversation files found")
            return
        
        print(f"   ✅ Found {len(files)} conversation files:")
        for file in files:
            size = format_file_size(file['size'])
            modified = file['last_modified'].strftime("%Y-%m-%d %H:%M:%S")
            print(f"   📄 {file['name']}")
            print(f"      Size: {size} | Modified: {modified}")
            if file['metadata']:
                print(f"      Metadata: {file['metadata']}")
            print()
            
    except Exception as e:
        print(f"   ❌ Error checking conversation files: {e}")

def check_specific_file(filename, file_type="research"):
    """Check if a specific file exists"""
    print(f"\n🔍 Checking specific file: {filename}")
    
    try:
        storage_manager = get_storage_manager()
        
        # Determine the path based on file type
        paths = {
            "research": f"pces/documents/research/{filename}",
            "patient_summary": f"pces/documents/doc-patient-summary/{filename}",
            "conversation": f"pces/documents/conversations/{filename}"
        }
        
        file_path = paths.get(file_type, f"pces/documents/research/{filename}")
        
        exists = storage_manager.check_file_exists("contoso", file_path)
        
        if exists:
            print(f"   ✅ File exists: {file_path}")
            file_info = storage_manager.get_file_metadata("contoso", file_path)
            if file_info:
                size = format_file_size(file_info['size'])
                modified = file_info['last_modified'].strftime("%Y-%m-%d %H:%M:%S")
                print(f"   📊 Size: {size}")
                print(f"   📅 Last Modified: {modified}")
                print(f"   🔗 URL: {file_info['url']}")
                if file_info['metadata']:
                    print(f"   📋 Metadata: {file_info['metadata']}")
        else:
            print(f"   ❌ File not found: {file_path}")
            
    except Exception as e:
        print(f"   ❌ Error checking file: {e}")

def main():
    print("🚀 PCES Azure Storage Upload Checker")
    print("=" * 50)
    
    if not AZURE_AVAILABLE:
        print("❌ Azure storage utilities not available")
        return
    
    # Check connection
    if not check_azure_connection():
        return
    
    # Check all file types
    check_research_files()
    check_patient_summary_files()
    check_conversation_files()
    
    print("\n" + "=" * 50)
    print("✅ Azure storage check complete!")
    print("\n💡 Tips:")
    print("   - Use Azure Storage Explorer for a GUI interface")
    print("   - Check your Flask app logs for upload confirmations")
    print("   - API endpoints: /check_azure_files, /azure_storage_info")

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        file_type = sys.argv[2] if len(sys.argv) > 2 else "research"
        
        print("🚀 PCES Azure Storage - Specific File Check")
        print("=" * 50)
        
        if check_azure_connection():
            check_specific_file(filename, file_type)
    else:
        main()
