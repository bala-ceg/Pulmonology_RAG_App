#!/usr/bin/env python3
"""
Clear and rebuild external KB with better medical content
"""

import os
import shutil
import sys

def clear_external_kb():
    """Clear the existing external knowledge base."""
    kb_external_path = "./vector_dbs/kb_external"
    
    if os.path.exists(kb_external_path):
        try:
            shutil.rmtree(kb_external_path)
            print(f"✅ Cleared existing external KB at {kb_external_path}")
            return True
        except Exception as e:
            print(f"❌ Error clearing external KB: {e}")
            return False
    else:
        print("ℹ️  No existing external KB found")
        return True

def main():
    """Clear existing external KB and prompt for rebuild."""
    print("🧹 External KB Cleanup and Rebuild")
    print("=" * 40)
    
    # Clear existing external KB
    if clear_external_kb():
        print("\n📋 Next steps:")
        print("1. Run: python3 enhanced_external_kb.py setup")
        print("2. Wait for the enhanced setup to complete (may take 10-15 minutes)")
        print("3. Test with: python3 enhanced_external_kb.py test")
        print("4. Restart your Flask app and try the diabetes query again")
        
        # Create the directory structure
        os.makedirs("./vector_dbs/kb_external", exist_ok=True)
        print("\n✅ Ready for enhanced external KB setup!")
        return True
    else:
        print("❌ Failed to clear external KB")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)