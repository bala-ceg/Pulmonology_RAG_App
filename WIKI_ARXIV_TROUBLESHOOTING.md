# Wikipedia and ArXiv Not Working - Troubleshooting Guide
## Date: November 24, 2025

## Quick Diagnosis

Run the diagnostic script first to identify the exact issue:

```bash
cd /Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App
source ../.venv/bin/activate
python diagnose_wiki_arxiv.py
```

This will tell you exactly what's failing.

## Common Causes & Solutions

### 1. **Package Version Issues** (Most Common)

The `langchain-community` package may have been updated with breaking changes.

**Solution:**
```bash
source ../.venv/bin/activate
pip install --upgrade langchain-community
pip install --upgrade wikipedia
pip install --upgrade arxiv
```

If that doesn't work, try pinning to known working versions:
```bash
pip install langchain-community==0.0.38
pip install wikipedia==1.4.0
pip install arxiv==2.1.0
```

### 2. **Wikipedia/ArXiv Package Installation Missing**

Sometimes the `wikipedia` and `arxiv` packages get uninstalled accidentally.

**Solution:**
```bash
pip install wikipedia arxiv
```

### 3. **SSL/Certificate Issues**

macOS sometimes has certificate issues.

**Solution:**
```bash
# Install certificates for Python
/Applications/Python\ 3.*/Install\ Certificates.command

# Or use pip to fix SSL
pip install --upgrade certifi
```

### 4. **Network/Firewall Blocking**

Corporate networks or VPNs might block API access.

**Solution:**
- Disconnect from VPN and try again
- Check if you can access https://en.wikipedia.org in your browser
- Check if you can access http://export.arxiv.org/api/query?search_query=test&max_results=1

### 5. **Signal Module Issues (macOS specific)**

The timeout handler uses `signal.SIGALRM` which doesn't work on Windows and can have issues on macOS.

**Solution - Disable Timeout (Temporary):**

Edit `enhanced_tools.py` and comment out the timeout wrapper:

```python
# Around line 507 (Wikipedia):
try:
    # with timeout_handler(30):  # COMMENT THIS OUT
    loader = WikipediaLoader(query=processed_query, load_max_docs=3)
    docs = loader.load()
    print(f"Enhanced Wikipedia_Search: Loaded {len(docs)} documents")
# except TimeoutException as te:  # COMMENT THIS OUT
except Exception as e:  # CHANGE THIS
    print(f"⏱️ Wikipedia search error: {e}")
```

Do the same for ArXiv around line 578.

### 6. **API Rate Limiting**

Wikipedia or ArXiv might be rate-limiting your requests.

**Solution:**
- Wait 5-10 minutes and try again
- Reduce `load_max_docs` from 3 to 1 in the loaders

### 7. **Alternative: Use Backup API Methods**

If loaders consistently fail, we can implement direct API calls.

**Create a new file `wiki_arxiv_direct.py`:**

```python
import requests
import json
from typing import List, Dict

def search_wikipedia_direct(query: str, max_docs: int = 3) -> List[Dict]:
    """Direct Wikipedia API search without LangChain"""
    try:
        # Search for pages
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_docs
        }
        
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        search_results = response.json().get('query', {}).get('search', [])
        
        documents = []
        for result in search_results:
            page_id = result['pageid']
            title = result['title']
            
            # Get full page content
            content_params = {
                "action": "query",
                "prop": "extracts",
                "pageids": page_id,
                "format": "json",
                "explaintext": True
            }
            
            content_response = requests.get(search_url, params=content_params, timeout=10)
            content_response.raise_for_status()
            pages = content_response.json().get('query', {}).get('pages', {})
            
            if str(page_id) in pages:
                content = pages[str(page_id)].get('extract', '')
                documents.append({
                    'title': title,
                    'content': content,
                    'source': f'https://en.wikipedia.org/wiki/{title.replace(" ", "_")}',
                    'metadata': {'title': title, 'source_type': 'wikipedia'}
                })
        
        return documents
        
    except Exception as e:
        print(f"Direct Wikipedia search failed: {e}")
        return []


def search_arxiv_direct(query: str, max_docs: int = 3) -> List[Dict]:
    """Direct ArXiv API search without LangChain"""
    try:
        import xml.etree.ElementTree as ET
        
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "max_results": max_docs
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        documents = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            summary = entry.find('atom:summary', ns)
            link = entry.find('atom:id', ns)
            
            if title is not None and summary is not None:
                documents.append({
                    'title': title.text.strip(),
                    'content': summary.text.strip(),
                    'source': link.text if link is not None else '',
                    'metadata': {'Title': title.text.strip(), 'source_type': 'arxiv'}
                })
        
        return documents
        
    except Exception as e:
        print(f"Direct ArXiv search failed: {e}")
        return []


# Test the functions
if __name__ == "__main__":
    print("Testing direct Wikipedia search...")
    wiki_docs = search_wikipedia_direct("Type 2 Diabetes")
    if wiki_docs:
        print(f"✅ Found {len(wiki_docs)} Wikipedia documents")
        print(f"   Sample: {wiki_docs[0]['title']}")
    else:
        print("❌ Wikipedia search failed")
    
    print("\nTesting direct ArXiv search...")
    arxiv_docs = search_arxiv_direct("diabetes treatment")
    if arxiv_docs:
        print(f"✅ Found {len(arxiv_docs)} ArXiv documents")
        print(f"   Sample: {arxiv_docs[0]['title'][:60]}...")
    else:
        print("❌ ArXiv search failed")
```

Then modify `enhanced_tools.py` to use these backup methods when loaders fail.

## Immediate Action Steps

1. **Run the diagnostic script** (see top of document)
2. **Check the output** and identify which test fails
3. **Apply the appropriate solution** from above
4. **Test with simple query:**
   ```bash
   python -c "from langchain_community.document_loaders import WikipediaLoader; print(WikipediaLoader('diabetes', load_max_docs=1).load())"
   ```

## Need More Help?

If none of these solutions work, please provide:
1. Output of `diagnose_wiki_arxiv.py`
2. Python version: `python --version`
3. Error messages from the application
4. Output of: `pip list | grep -E "(langchain|wikipedia|arxiv)"`
