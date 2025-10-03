# Two-Store RAG Architecture with Lexical Gate

## Overview

This update transforms your Python medical app into a sophisticated two-store RAG (Retrieval-Augmented Generation) architecture with intelligent query routing. The system now maintains two separate vector databases and uses a TF-IDF lexical gate to determine the optimal knowledge source for each query.

## Architecture Components

### 1. Two Vector Stores (Chroma with LangChain)

#### `kb_local` - Internal Knowledge Base
- **Purpose**: Stores internal PDFs and URLs uploaded by users
- **Location**: `./vector_dbs/kb_local/`
- **Content**: User-uploaded medical documents, PDFs, and extracted web content
- **Priority**: High relevance for user-specific queries

#### `kb_external` - External Knowledge Base  
- **Purpose**: Stores Wikipedia and arXiv content
- **Location**: `./vector_dbs/kb_external/`
- **Content**: Medical Wikipedia articles, arXiv research papers
- **Priority**: Broad medical knowledge and research findings

### 2. TF-IDF Lexical Gate

The lexical gate uses TF-IDF (Term Frequency-Inverse Document Frequency) analysis to intelligently route queries:

- **Automation Summary**: Built from local knowledge base chunks
- **Threshold**: Configurable similarity threshold (default: 0.3)
- **Routing Logic**:
  - If query terms overlap significantly with internal index → Query `kb_local` first
  - If similarity score ≥ threshold → Query `kb_local` first, fallback to `kb_external`
  - Else → Query `kb_external` first, fallback to `kb_local`

### 3. Intelligent Query Routing

```
User Query → TF-IDF Gate → Route Decision
                ↓
    ┌─────────────────────────────────┐
    │ High Similarity (≥ threshold)   │
    │ Query kb_local first            │
    │ ↓ If weak response              │
    │ Fallback to kb_external         │
    └─────────────────────────────────┘
                ↓
    ┌─────────────────────────────────┐
    │ Low Similarity (< threshold)    │
    │ Query kb_external first         │
    │ ↓ If weak response              │
    │ Fallback to kb_local            │
    └─────────────────────────────────┘
```

## New Dependencies

The following packages have been added to `requirements.txt`:

```
# LangChain text processing
langchain-text-splitters

# Vector database
chromadb

# Document processing
pypdf
tiktoken

# Scikit-learn for TF-IDF gate
scikit-learn
```

## Key Files

### New Files Added

1. **`rag_architecture.py`** - Core RAG architecture implementation
   - `TFIDFLexicalGate` class
   - `TwoStoreRAGManager` class
   - Wikipedia and arXiv loaders integration

2. **`setup_external_kb.py`** - Utility script for external KB management
   - Initialize external knowledge base
   - Check knowledge base status
   - Test query routing functionality

### Modified Files

1. **`main.py`** - Updated with RAG integration
   - RAG manager initialization
   - Modified `/data` route with lexical gate routing
   - Integration with document upload process

2. **`requirements.txt`** - Added new dependencies

## Usage Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize External Knowledge Base

```bash
python setup_external_kb.py setup
```

This will populate the external KB with:
- 20 medical Wikipedia topics (3 articles each)
- 15 medical AI research queries from arXiv (2 papers each)

### 3. Check System Status

```bash
python setup_external_kb.py status
```

### 4. Test Query Routing

```bash
python setup_external_kb.py test
```

### 5. Run the Application

```bash
python main.py
```

## How It Works

### Document Upload Process

1. **User uploads PDF/URL** → Saved to session folder
2. **Click "Create Vector DB"** → Documents processed and added to:
   - Legacy vector DB (for backward compatibility)
   - `kb_local` (new RAG architecture)
   - TF-IDF lexical gate updated with new content

### Query Processing

1. **User submits query** → System checks if RAG architecture available
2. **TF-IDF Gate Analysis** → Calculates similarity with local documents
3. **Routing Decision** → Determines primary knowledge source
4. **Query Execution** → Queries primary source, fallback if response is weak
5. **Response Synthesis** → Combines results with citations and routing info

### Response Quality Assessment

The system evaluates response strength using:
- Response length (minimum 50 characters)
- Absence of weak indicators ("I don't know", "insufficient information", etc.)
- Content relevance to the query

## Backward Compatibility

The system maintains full backward compatibility:
- **Legacy routes** continue to work unchanged
- **Existing vector databases** remain functional
- **Fallback mechanism** uses original medical routing if RAG unavailable
- **Graceful degradation** if dependencies are missing

## Monitoring and Debugging

### Console Output

The system provides detailed logging:
- `🧠 Using Two-Store RAG Architecture with Lexical Gate`
- `🚪 Gate Decision: Local first (similarity: 0.456)`
- `📚 Adding documents to RAG manager's local knowledge base...`
- `✅ Successfully added 15 documents to kb_local`

### Response Metadata

Each response includes routing information:
```json
{
  "routing_details": {
    "method": "Two-Store RAG with Lexical Gate",
    "similarity_score": 0.456,
    "query_local_first": true,
    "sources_queried": ["kb_local"],
    "responses_count": 1
  }
}
```

## Configuration

### TF-IDF Gate Threshold

Adjust the similarity threshold in `rag_architecture.py`:
```python
lexical_gate = TFIDFLexicalGate(threshold=0.3)  # Default: 0.3
```

- **Lower threshold (0.1-0.2)**: More queries route to local KB first
- **Higher threshold (0.4-0.6)**: More queries route to external KB first

### External KB Content

Modify topics in `setup_external_kb.py`:
```python
medical_topics = [
    "your_custom_topic",
    "specialized_medical_field",
    # ... add your topics
]
```

## Troubleshooting

### RAG Architecture Not Available
If you see "⚠️ RAG Architecture not available - using legacy mode":
1. Check dependencies: `pip install scikit-learn chromadb langchain-text-splitters`
2. Verify imports in `rag_architecture.py`

### External KB Setup Issues
1. Check API keys in `.env` file
2. Verify internet connection for Wikipedia/arXiv access
3. Run `python setup_external_kb.py status` to diagnose

### Query Routing Problems
1. Check if lexical gate is fitted: Look for "🚪 Lexical Gate: Fitted" in status
2. Verify local documents exist in `kb_local`
3. Run test queries: `python setup_external_kb.py test`

## Performance Considerations

- **Initial Setup**: External KB initialization takes 5-10 minutes
- **Query Response**: Typically 2-5 seconds with routing overhead
- **Storage**: Each KB can grow to several GB with extensive content
- **Memory**: TF-IDF gate requires ~100MB RAM for moderate document sets

## Future Enhancements

- [ ] Hybrid search combining semantic and lexical similarity
- [ ] Dynamic threshold adjustment based on query performance
- [ ] Multi-language support for international medical content
- [ ] Integration with additional medical databases (PubMed, MEDLINE)
- [ ] Real-time knowledge base updates and synchronization