# Integrated Medical RAG System with Tool Routing

This implementation provides a comprehensive medical question-answering system with intelligent tool routing, guarded retrieval, and automatic fallback mechanisms.

## 🏗️ Architecture Overview

The system consists of four main components:

### 1. Self-Describing Tools (`tools.py`)
- **Wikipedia_Search**: For definitions, factual explanations, and general medical knowledge
- **ArXiv_Search**: For recent research papers and scientific studies  
- **Internal_VectorDB**: For uploaded PDFs and organizational knowledge base
- Each tool returns plain string output with automatic 1200-character limits
- Built-in fallback to Wikipedia when retrieval quality is low

### 2. Routing System (`prompts.py`)
- **ROUTING_SYSTEM_PROMPT**: Comprehensive instructions for LLM tool selection
- Clear "use" and "do not use" conditions for each tool
- Fallback behavior guidelines and source attribution requirements
- Confidence scoring and explanation templates

### 3. Query Router (`rag_architecture.py` - MedicalQueryRouter)
- **route_tools()**: Intelligent tool selection based on keyword heuristics
- Returns ranked list of 1-2 most appropriate tools
- Confidence scoring and reasoning explanation
- Context-aware decisions based on available content

### 4. Post-Retrieval Guard (`tools.py` + `rag_architecture.py`)
- **guarded_retrieve()**: Quality checking with similarity thresholds
- Automatic fallback to Wikipedia when retrieval fails
- Content relevance validation using main noun extraction
- Generic content filtering

## 🚀 Quick Start

### Basic Usage

```python
from integrated_rag import IntegratedMedicalRAG

# Initialize system
rag_system = IntegratedMedicalRAG(
    openai_api_key="your-api-key",
    base_vector_path="./vector_dbs"
)

# Query the system
result = rag_system.query("What is hypertension?")
print(f"Answer: {result['answer']}")
print(f"Primary Tool: {result['routing_info']['primary_tool']}")
print(f"Confidence: {result['routing_info']['confidence']}")
```

### Adding Documents

```python
from langchain.schema import Document

# Add to local knowledge base (user uploads)
local_docs = [
    Document(page_content="Your organization's protocols...", 
             metadata={'source': 'protocol.pdf'})
]
rag_system.add_documents_to_local(local_docs)

# Add to external knowledge base (Wikipedia/ArXiv content)
external_docs = [
    Document(page_content="Research findings...", 
             metadata={'source_type': 'arxiv', 'Title': 'Study Title'})
]
rag_system.add_documents_to_external(external_docs)
```

## 🧭 Routing Logic

### Tool Selection Heuristics

**ArXiv_Search** is selected when query contains:
- `latest`, `recent`, `research`, `study`, `paper`, `findings`
- `experiment`, `trial`, `breakthrough`, `cutting-edge`

**Internal_VectorDB** is selected when query contains:
- `uploaded`, `my file`, `my document`, `our protocol`
- `organization`, `internal`, `this document`

**Wikipedia_Search** is selected when query contains:
- `what is`, `define`, `explain`, `tell me about`
- `overview`, `basic`, `general`, `meaning`

### Fallback Mechanisms

1. **Low Similarity Fallback**: If Internal_VectorDB returns low relevance scores → Wikipedia
2. **No Content Fallback**: If no documents uploaded → Wikipedia  
3. **Generic Content Fallback**: If retrieved content is too generic → Wikipedia
4. **Error Fallback**: If any tool fails → Wikipedia

## 🛡️ Guarded Retrieval

The system includes post-retrieval quality checks:

- **Similarity Threshold**: Default 0.35 minimum relevance score
- **Content Relevance**: Ensures retrieved chunks contain main query terms
- **Quality Filtering**: Removes very short or generic responses
- **Automatic Fallback**: Triggers Wikipedia search when guards fail

```python
# Example of guarded retrieval in action
docs = guarded_retrieve(
    query="diabetes treatment", 
    retriever=vector_db_retriever,
    similarity_threshold=0.35
)

if docs is None:
    # Automatic fallback to Wikipedia
    response = Wikipedia_Search("diabetes treatment")
```

## 📊 System Status

Check system configuration and content:

```python
status = rag_system.get_system_status()
print(f"Local documents: {status['local_document_count']}")
print(f"External documents: {status['external_document_count']}")
print(f"Available tools: {status['available_tools']}")
```

## 🧪 Testing

Run comprehensive tests:

```bash
python test_integrated_system.py
```

The test suite covers:
- Query routing for different question types
- Tool function execution
- Guarded retrieval behavior
- Fallback mechanisms
- Prompt generation

## 📝 Example Queries and Expected Routing

| Query | Primary Tool | Reasoning |
|-------|-------------|-----------|
| "What is hypertension?" | Wikipedia_Search | Definition request |
| "Latest COVID-19 research" | ArXiv_Search | Research keyword |
| "My uploaded protocol says what?" | Internal_VectorDB | User document reference |
| "Recent studies on diabetes" | ArXiv_Search | Research + recent keywords |
| "Define myocardial infarction" | Wikipedia_Search | Definition keyword |

## ⚙️ Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Vector Database Paths
- Local KB: `./vector_dbs/kb_local/`
- External KB: `./vector_dbs/kb_external/`
- Lexical Gate: `./vector_dbs/lexical_gate.pkl`

### Customization Options

```python
# Adjust similarity threshold
guarded_retrieve(query, retriever, similarity_threshold=0.4)

# Modify routing keywords
router.arxiv_keywords.extend(['novel', 'innovative'])

# Change response length limits
_join_docs(docs, max_chars=1500)
```

## 🔧 Integration with Existing Systems

To integrate with your existing Flask app:

```python
from integrated_rag import IntegratedMedicalRAG

# In your Flask route
@app.route('/query', methods=['POST'])
def handle_query():
    question = request.json['question']
    session_id = request.json.get('session_id')
    
    result = rag_system.query(question, session_id)
    
    return jsonify({
        'answer': result['answer'],
        'source': result['routing_info']['primary_tool'],
        'confidence': result['routing_info']['confidence']
    })
```

## 🚨 Error Handling

The system includes comprehensive error handling:
- Tool execution failures → Automatic fallback
- Import errors → Graceful degradation  
- API failures → User-friendly error messages
- Missing content → Wikipedia fallback

## 📈 Performance Considerations

- **Tool Limitation**: Uses only top 1-2 tools per query for efficiency
- **Content Limits**: 1200-character response limits to prevent context overflow
- **Caching**: Vector databases persist between sessions
- **Fallback Speed**: Wikipedia fallback is fast and reliable

## 🔄 Future Enhancements

Potential improvements:
- Dynamic similarity thresholds based on query complexity
- User feedback integration for routing optimization
- Custom tool creation for specialized domains
- Multi-language support for international medical content

## 📞 Support

For implementation questions or issues:
1. Check the test suite results: `python test_integrated_system.py`
2. Review routing decisions in system logs
3. Verify vector database content with `get_system_status()`
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

---

**Note**: This system requires OpenAI API access and appropriate API keys. Ensure you have sufficient API quota for production usage.