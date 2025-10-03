# Migration Guide: Two-Store RAG Architecture

## Quick Start

### 1. Install New Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize External Knowledge Base
```bash
python setup_external_kb.py setup
```

### 3. Start the Application
```bash
python main.py
```

## What's Changed

### For Users
- **Improved Query Responses**: More accurate answers from dual knowledge sources
- **Intelligent Routing**: System automatically chooses the best knowledge source
- **Enhanced Citations**: Better source attribution and transparency
- **Same Interface**: No changes to the web interface

### For Developers
- **New Architecture**: Two-store RAG system with lexical gate
- **Backward Compatible**: All existing functionality preserved
- **Enhanced Logging**: Detailed routing and processing information
- **Extensible Design**: Easy to add new knowledge sources

## Testing Your Setup

### 1. Run Integration Test
```bash
python test_rag_integration.py
```

### 2. Check System Status
```bash
python setup_external_kb.py status
```

### 3. Test Query Routing
```bash
python setup_external_kb.py test
```

## Expected Behavior

### When RAG Architecture is Available
- Console shows: "🧠 Using Two-Store RAG Architecture with Lexical Gate"
- Query responses include routing information
- Better handling of medical queries with Wikipedia/arXiv content

### When RAG Architecture is Unavailable
- Console shows: "⚠️ RAG Architecture not available - using legacy mode"
- Falls back to original medical routing system
- All functionality remains available

## Troubleshooting

### If External KB Setup Fails
1. Check your `.env` file has correct API keys
2. Ensure internet connection for Wikipedia/arXiv access
3. Run with verbose output: `python -u setup_external_kb.py setup`

### If Integration Test Fails
1. Install missing dependencies: `pip install scikit-learn`
2. Check Python version compatibility (3.8+)
3. Verify LangChain installation: `pip install langchain-community`

### If Main App Has Issues
1. The app falls back to legacy mode automatically
2. Check console output for specific error messages
3. Restart the app after fixing dependencies

## Performance Notes

- **First Run**: External KB setup takes 5-10 minutes
- **Subsequent Runs**: Fast startup with cached knowledge bases
- **Query Time**: 2-5 seconds including intelligent routing
- **Storage**: ~500MB-1GB for full external knowledge base

## Support

If you encounter issues:
1. Check the console output for detailed error messages
2. Run the integration test to identify specific problems
3. Verify all dependencies are installed correctly
4. The system will fall back gracefully if components are unavailable