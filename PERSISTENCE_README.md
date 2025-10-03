# RAG Persistence System

## Overview

Your RAG (Retrieval-Augmented Generation) system now includes intelligent persistence to improve startup performance and user experience.

## Key Features

### 🚀 **Fast Startup**
- External knowledge base (Wikipedia + arXiv content) persists between app restarts
- No more waiting for Wikipedia/arXiv downloads on every startup
- Intelligent loading: only downloads content if external KB is empty

### 🔄 **Two-Store Architecture**
- **Local KB (`kb_local`)**: Your uploaded documents (PDFs, etc.)
- **External KB (`kb_external`)**: Medical research from Wikipedia & arXiv
- **Smart Routing**: TF-IDF lexical gate routes queries to the best knowledge source

### 📚 **User-Friendly Citations**
- Local KB citations → **"Adhoc Documents"**  
- External KB citations → **"Third Party Research"**
- Clear source attribution for better trust and transparency

## Current Status

- **Local KB**: 34 documents (your uploads)
- **External KB**: 13,567 documents (medical research)
- **Persistence**: ✅ Active - fast startups enabled

## Management Commands

### Check Status
```bash
python manage_external_kb.py status
```

### Force Rebuild External KB
```bash
python manage_external_kb.py rebuild
```

### Setup External KB (skip if exists)
```bash
python manage_external_kb.py setup
```

### Force Setup External KB
```bash
python manage_external_kb.py force
```

## How It Works

### Intelligent Query Routing
1. **TF-IDF Analysis**: Compares your query against both knowledge bases
2. **Similarity Threshold**: 0.3 threshold determines routing
3. **Smart Fallback**: Always provides relevant answers from best source

### Persistence Behavior
- **First Run**: Downloads medical content (takes a few minutes)
- **Subsequent Runs**: Loads from disk (starts immediately)
- **Content Updates**: Use management commands to refresh external knowledge

### Storage Location
```
vector_dbs/
├── kb_local/           # Your uploaded documents
├── kb_external/        # Medical research (persistent)
└── lexical_gate.pkl    # TF-IDF routing model
```

## Tips

1. **Fast Development**: External KB persists - no delays during development
2. **Content Management**: Use `manage_external_kb.py` to control external content
3. **Query Optimization**: Medical queries automatically use external research
4. **Local Documents**: Your uploads are always prioritized for specific content

## Technical Details

- **Vector Database**: Chroma with OpenAI embeddings
- **Lexical Gate**: TF-IDF with scikit-learn  
- **Persistence**: Automatic disk-based storage
- **Display Names**: User-friendly citation labels
- **Error Handling**: Graceful fallbacks and JSON-safe responses