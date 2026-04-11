#!/usr/bin/env python3
"""
Test the new user-friendly display names for citations
"""

import sys
import os
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_display_names():
    """Test the user-friendly display names in citations."""
    
    try:
        from rag_architecture import TwoStoreRAGManager
        from langchain_core.documents import Document
        
        print("🎨 Testing User-Friendly Display Names")
        print("=" * 40)
        
        # Test the _format_citations method directly
        rag_manager = TwoStoreRAGManager(None, None)  # We don't need real embeddings for this test
        
        # Create mock search results
        mock_local_docs = [
            Document(
                page_content="This is content from a local PDF",
                metadata={"source": "/path/to/sample1.pdf", "type": "pdf"}
            )
        ]
        
        mock_external_wikipedia = [
            Document(
                page_content="This is content from Wikipedia",
                metadata={
                    "source_type": "wikipedia",
                    "title": "Type 2 Diabetes",
                    "source": "https://en.wikipedia.org/wiki/Type_2_diabetes"
                }
            )
        ]
        
        mock_external_arxiv = [
            Document(
                page_content="This is content from arXiv",
                metadata={
                    "source_type": "arxiv", 
                    "Title": "Medical Diagnosis Using AI",
                    "Authors": "John Doe, Jane Smith",
                    "source": "https://arxiv.org/abs/1234.5678"
                }
            )
        ]
        
        print("🔍 Testing citation formatting...")
        
        # Test Local KB citations (should show as "Adhoc Documents")
        local_citations = rag_manager._format_citations(mock_local_docs, "Local KB")
        print("\n📄 Local KB Citations (should show 'Adhoc Documents'):")
        for citation in local_citations:
            print(f"  {citation}")
        
        # Test External KB Wikipedia citations (should show as "Third Party Research")
        external_wiki_citations = rag_manager._format_citations(mock_external_wikipedia, "External KB")
        print("\n🌐 External KB Wikipedia Citations (should show 'Third Party Research'):")
        for citation in external_wiki_citations:
            print(f"  {citation}")
        
        # Test External KB arXiv citations (should show as "Third Party Research")
        external_arxiv_citations = rag_manager._format_citations(mock_external_arxiv, "External KB")
        print("\n🔬 External KB arXiv Citations (should show 'Third Party Research'):")
        for citation in external_arxiv_citations:
            print(f"  {citation}")
        
        # Verify the display names are correct
        success = True
        
        if "Adhoc Documents" not in local_citations[0]:
            print("❌ Local KB display name not working")
            success = False
        else:
            print("✅ Local KB display name working: 'Adhoc Documents'")
        
        if "Third Party Research" not in external_wiki_citations[0]:
            print("❌ External KB Wikipedia display name not working")
            success = False
        else:
            print("✅ External KB Wikipedia display name working: 'Third Party Research'")
        
        if "Third Party Research" not in external_arxiv_citations[0]:
            print("❌ External KB arXiv display name not working")
            success = False
        else:
            print("✅ External KB arXiv display name working: 'Third Party Research'")
        
        print("\n" + "=" * 40)
        if success:
            print("🎉 All display names are working correctly!")
            print("\n📋 Expected behavior in your app:")
            print("• Local documents will show as 'Adhoc Documents' in citations")
            print("• Wikipedia/arXiv content will show as 'Third Party Research' in citations") 
            print("• Backend names (Local KB, External KB) remain unchanged")
        else:
            print("❌ Some display names need fixing")
        
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure rag_architecture.py is available")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_display_names()
    sys.exit(0 if success else 1)