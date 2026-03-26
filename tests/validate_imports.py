#!/usr/bin/env python3
"""
Import validation script to check if all required packages are installed correctly.
Run this script to verify your environment has all necessary dependencies.
"""

import sys
import importlib

# List of modules that should be importable
REQUIRED_MODULES = [
    # Core web framework
    'flask',

    # LangChain ecosystem
    'langchain',
    'langchain_openai',
    'langchain_chroma',
    'langchain_community',
    'langchain_text_splitters',  # langchain-text-splitters

    # Vector database
    'chromadb',

    # ML / Data science
    'numpy',
    'pandas',
    'sklearn',          # scikit-learn
    'sentence_transformers',
    'joblib',

    # Document processing
    'pypdf',
    'tiktoken',
    'fitz',             # PyMuPDF
    'pdfplumber',

    # Web scraping
    'bs4',              # beautifulsoup4
    'requests',
    'selenium',

    # Audio processing
    'whisper',          # openai-whisper
    'torch',
    'torchaudio',
    'librosa',
    'soundfile',
    'pydub',
    'pyannote.audio',

    # Database
    'psycopg',          # psycopg[pool]  — v3 driver used in main.py
    'psycopg2',         # psycopg2-binary — v2 driver used in postgres_tool.py

    # ORM
    'sqlalchemy',

    # PDF generation
    'reportlab',

    # Azure
    'azure.storage.blob',
    'azure.core',

    # External knowledge sources
    'wikipedia',
    'arxiv',
    'tavily',           # tavily-python

    # LLM / fine-tuning
    'openai',
    'transformers',
    'peft',
    'trl',
    'datasets',
    'accelerate',
    'safetensors',

    # Other utilities
    'apify_client',     # apify-client
    'dotenv',           # python-dotenv
]

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    """Run import validation for all required modules."""
    print("🔍 Validating Python package imports...")
    print("=" * 60)
    
    missing_packages = []
    import_errors = []
    
    for module in REQUIRED_MODULES:
        success, error = check_import(module)
        if success:
            print(f"✅ {module}")
        else:
            print(f"❌ {module} - {error}")
            missing_packages.append(module)
            import_errors.append((module, error))
    
    print("\n" + "=" * 60)
    
    if not missing_packages:
        print("🎉 All required packages are installed correctly!")
        return True
    else:
        print(f"⚠️  {len(missing_packages)} packages are missing or have issues:")
        
        # Group by likely PyPI package names
        pypi_suggestions = {
            'fitz': 'PyMuPDF',
            'bs4': 'beautifulsoup4',
            'dotenv': 'python-dotenv',
            'sklearn': 'scikit-learn',
            'whisper': 'openai-whisper',
            'langchain_text_splitters': 'langchain-text-splitters',
            'sentence_transformers': 'sentence-transformers',
            'psycopg2': 'psycopg2-binary',
            'apify_client': 'apify-client',
            'tavily': 'tavily-python',
            'pyannote.audio': 'pyannote.audio',
        }
        
        print("\n📦 Install missing packages with:")
        print("pip install", end="")
        
        for module in missing_packages:
            pypi_name = pypi_suggestions.get(module, module)
            print(f" {pypi_name}", end="")
        
        print("\n")
        
        # Show detailed errors
        print("\n🔍 Detailed import errors:")
        for module, error in import_errors:
            print(f"   {module}: {error}")
        
        print("\n💡 Notes:")
        print("   - Some packages may require system dependencies (e.g., ffmpeg for pydub)")
        print("   - torch/torchaudio may need platform-specific versions")
        print("   - pyannote.audio may require additional setup")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)