#!/usr/bin/env python3
"""
Organization KB Setup Script
This script initializes the Organization Knowledge Base with sample data.
"""

import os
import sys
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_organization_kb():
    """Initialize Organization KB with sample data."""
    
    print("🏥 Setting up Organization Knowledge Base...")
    
    # Check if config file exists
    if not os.path.exists("config/disciplines.json"):
        print("❌ Config file not found: config/disciplines.json")
        return
    
    # Check environment variables
    required_env_vars = ['openai_api_key', 'base_url', 'embedding_model_name']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return
    
    try:
        # Initialize embeddings
        print("🔧 Initializing embeddings...")
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv('openai_api_key'),
            base_url=os.getenv('base_url'),
            model=os.getenv('embedding_model_name')
        )
        print("✅ Embeddings initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=128,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # Load disciplines configuration
    with open("config/disciplines.json", "r") as f:
        config = json.load(f)
    
    for discipline in config["disciplines"]:
        discipline_id = discipline["id"]
        discipline_name = discipline["name"]
        kb_path = discipline["kb_path"]
        vector_db_path = discipline["vector_db_path"]
        
        print(f"\n📚 Processing {discipline_name}...")
        
        # Look for markdown files in the discipline folder
        discipline_dir = os.path.join(".", kb_path)
        if not os.path.exists(discipline_dir):
            print(f"   ⚠️  Directory not found: {discipline_dir}")
            continue
            
        documents = []
        
        # Process all .md files in the discipline directory
        for filename in os.listdir(discipline_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(discipline_dir, filename)
                print(f"   📄 Processing {filename}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into chunks
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "type": "organization_knowledge",
                            "discipline": discipline_id,
                            "chunk_id": i
                        }
                    ))
        
        if documents:
            # Create vector database
            persist_dir = os.path.join(".", vector_db_path)
            os.makedirs(persist_dir, exist_ok=True)
            
            print(f"   🔄 Creating vector database with {len(documents)} documents...")
            
            vector_store = Chroma.from_documents(
                documents,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            
            print(f"   ✅ Vector database created for {discipline_name}")
        else:
            print(f"   ⚠️  No documents found for {discipline_name}")
    
    print("\n🎉 Organization KB setup complete!")

if __name__ == "__main__":
    setup_organization_kb()
