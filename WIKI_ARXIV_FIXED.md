# Wikipedia & ArXiv Issue - RESOLVED ✅
**Date:** November 24, 2025

## Problem
Wikipedia and ArXiv searches stopped working suddenly after 2 weeks of normal operation.

## Root Cause
**PyMuPDF Version Conflict** - The ArXiv loader uses PyMuPDF (fitz) to process PDF papers. PyMuPDF was upgraded from version 1.23.26 to 1.26.3, which introduced a breaking change:
- Old version: `fitz.fitz.FileDataError`
- New version: `fitz.FileDataError` (removed the nested `fitz.fitz` attribute)

This caused the error: `AttributeError: module 'fitz' has no attribute 'fitz'`

## Solution Applied

### 1. **Downgraded PyMuPDF to stable version**
```bash
pip uninstall -y PyMuPDF
pip install PyMuPDF==1.23.26
```

### 2. **Updated requirements.txt**
Changed from:
```
PyMuPDF
```
To:
```
PyMuPDF==1.23.26
```

This prevents automatic upgrades that could break ArXiv functionality.

### 3. **Added User-Agent to Wikipedia API calls**
Fixed 403 errors in diagnostic script by adding proper User-Agent header (though this didn't affect the LangChain loader which already handles this).

## Test Results

### ✅ All Tests Passing:
- **Wikipedia_Search**: Working (1089 chars returned)
- **ArXiv_Search**: Working (1050 chars returned)
- **enhanced_wikipedia_search**: Working (with LLM summaries)
- **enhanced_arxiv_search**: Working (with LLM summaries)

### Test Output:
```
✅ Wikipedia_Search WORKING!
✅ ArXiv_Search WORKING!
✅ enhanced_wikipedia_search WORKING!
✅ enhanced_arxiv_search WORKING!
```

## Prevention
The `requirements.txt` file now pins PyMuPDF to version 1.23.26, preventing future automatic upgrades that could break compatibility.

## Files Modified
1. `/requirements.txt` - Pinned PyMuPDF==1.23.26
2. `/diagnose_wiki_arxiv.py` - Added User-Agent header for Wikipedia API

## How to Verify
Run the quick test:
```bash
cd /Users/bseetharaman/Desktop/Bala/2025/AI_Medical_App/Pulmonology_RAG_App
source ../.venv/bin/activate
python test_wiki_arxiv_quick.py
```

All tools should show "✅ WORKING!"

## Note for Future
If you need to upgrade PyMuPDF in the future, test ArXiv functionality carefully. The langchain-community package may need updates to support newer PyMuPDF versions.
