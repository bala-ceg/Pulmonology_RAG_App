# Medical RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) system for medical information retrieval and analysis, featuring intelligent query routing, multi-source knowledge integration, and RLHF (Reinforcement Learning from Human Feedback) capabilities.

## Overview

This application provides an AI-powered medical assistant that combines multiple knowledge sources including internal documents, Wikipedia, arXiv research papers, PostgreSQL medical databases, and real-time web search. The system uses advanced RAG architecture with TF-IDF lexical gating and intelligent routing to provide accurate, context-aware medical information.

## Key Features

### Core Capabilities
- **Multi-Source Knowledge Integration**: Combines internal PDFs/URLs, Wikipedia, arXiv, PostgreSQL databases, and Tavily web search
- **Intelligent Query Routing**: Automatically routes queries to the most appropriate knowledge source
- **Two-Store RAG Architecture**: Separate knowledge bases for local (internal) and external sources
- **TF-IDF Lexical Gate**: Smart decision-making for query routing between knowledge bases
- **Session Management**: Maintains conversation context across multiple interactions
- **Patient Context Integration**: Tailors responses based on patient demographics and medical history

### Advanced Features
- **RLHF System**: Reinforcement Learning from Human Feedback for continuous improvement
- **Reward Model Training**: Uses sentence transformers and logistic regression for response ranking
- **Voice Diarization**: Separates doctor and patient voices in audio recordings
- **Audio Transcription**: Whisper-based transcription for medical conversations
- **Azure Blob Storage Integration**: Cloud storage for PDFs and audio files
- **PostgreSQL Database Tool**: Queries medical diagnosis data from structured databases
- **Real-time Web Search**: Tavily API integration for up-to-date medical information

### Document Processing
- **PDF Processing**: Extract text, tables, and images from medical PDFs
- **Web Content Extraction**: Scrape and process medical content from URLs
- **Metadata Management**: Track and organize document metadata
- **Vector Database**: Efficient semantic search using ChromaDB

## Architecture

### RAG Architecture Components

1. **kb_local**: Internal knowledge base (uploaded PDFs + URLs)
2. **kb_external**: External knowledge base (Wikipedia + arXiv)
3. **kb_organization**: Organization-specific knowledge bases (multi-discipline support)
4. **Session Vector DBs**: Per-conversation context storage

### Tool System

The application uses a tool-based architecture with the following tools:

- **Wikipedia_Search**: Medical definitions, factual explanations, and general medical knowledge
- **ArXiv_Search**: Latest research papers and scientific studies
- **Tavily_Search**: Real-time web search for current medical information
- **Internal_VectorDB**: Search through uploaded PDFs and URLs
- **PostgreSQL_Diagnosis_Search**: Query structured medical diagnosis data

### Query Flow

```
User Query → Query Router → Tool Selection → Knowledge Retrieval → Response Generation → RLHF Reranking → Final Answer
```

## Technology Stack

### Core Dependencies
- **Flask**: Web framework for API endpoints
- **LangChain**: RAG and agent orchestration
- **OpenAI**: GPT models for language understanding and generation
- **ChromaDB**: Vector database for semantic search
- **PostgreSQL**: Structured medical data storage

### Machine Learning
- **Scikit-learn**: TF-IDF vectorization and reward model
- **Sentence Transformers**: Embedding generation for RLHF
- **PyTorch**: Deep learning framework for audio processing

### Document Processing
- **PyMuPDF (fitz)**: PDF text extraction
- **pdfplumber**: PDF table extraction
- **BeautifulSoup4**: HTML parsing and web scraping
- **Selenium**: Dynamic web content extraction

### Audio Processing
- **OpenAI Whisper**: Speech-to-text transcription
- **pyannote.audio**: Speaker diarization
- **pydub**: Audio file manipulation
- **librosa**: Audio feature extraction

### Cloud Services
- **Azure Storage Blob**: Cloud file storage
- **Apify**: Web scraping API
- **Tavily**: Real-time web search API

## Installation

### Prerequisites
- Python 3.8 or higher
- PostgreSQL database
- OpenAI API key
- Azure Storage account (optional)
- Hugging Face token (for voice diarization)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Pulmonology_RAG_App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file with the following variables:

```env
# OpenAI Configuration
openai_api_key=your_openai_api_key
base_url=https://api.openai.com/v1
llm_model_name=gpt-4o-mini
embedding_model_name=text-embedding-ada-002

# PostgreSQL Configuration (RLHF Database)
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=pces_base
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# PostgreSQL Tool Configuration (Medical Diagnosis Database)
PG_TOOL_HOST=your_db_host
PG_TOOL_PORT=5432
PG_TOOL_NAME=pces_ehr_ccm
PG_TOOL_USER=your_db_user
PG_TOOL_PASSWORD=your_db_password

# Azure Storage (optional)
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_CONTAINER_NAME=your_container_name

# Hugging Face Token (for voice diarization)
HUGGINGFACE_TOKEN=your_huggingface_token

# Apify API Key (for web scraping)
apify_api_key=your_apify_key

# Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_key
```

4. Initialize the database:
```bash
python setup_local_db.py
```

5. Set up knowledge bases:
```bash
# Setup external knowledge base (Wikipedia + arXiv)
python setup_external_kb.py

# Setup organization-specific knowledge bases
python setup_organization_kb.py
```

## Usage

### Starting the Application

```bash
python main.py
```

The Flask server will start on `http://localhost:5000`

### API Endpoints

#### Chat Endpoints

**POST /chat**
- Send a medical query and receive an AI-generated response
- Request body:
```json
{
  "query": "What are the symptoms of pneumonia?",
  "session_id": "unique_session_id",
  "patient_context": "Optional patient information"
}
```

**POST /chat_agent**
- Use agent-based routing for complex queries
- Automatically selects appropriate tools and knowledge sources

**POST /chat_pdf**
- Generate PDF reports of chat conversations
- Includes formatted medical information with sources

#### Document Management

**POST /upload_pdf**
- Upload medical PDFs to the knowledge base
- Automatically processes and indexes content

**POST /upload_url**
- Add medical websites to the knowledge base
- Extracts and indexes web content

**GET /get_pdfs**
- List all uploaded PDFs with metadata

**GET /get_urls**
- List all indexed URLs with metadata

**DELETE /delete_pdf**
- Remove a PDF from the knowledge base

**DELETE /delete_url**
- Remove a URL from the knowledge base

#### Session Management

**GET /get_sessions**
- List all conversation sessions

**GET /get_messages/<session_id>**
- Retrieve messages from a specific session

**DELETE /delete_session/<session_id>**
- Delete a conversation session

#### Voice Processing

**POST /upload_audio**
- Upload audio files for transcription and diarization
- Separates doctor and patient voices
- Returns structured conversation transcript

#### RLHF Administration

**GET /admin_rlhf**
- Access RLHF admin interface
- View and rate AI responses
- Train reward models

**POST /train_reward_model**
- Train a new reward model using collected feedback
- Updates the model for improved response ranking

## RLHF System

The application includes a comprehensive RLHF system for continuous improvement:

### Components

1. **Interaction Logging**: All queries and responses are logged to the database
2. **Human Feedback**: Administrators can rate responses (1-5 stars)
3. **Reward Model**: Trained on feedback to predict response quality
4. **Reranking**: Uses trained model to rerank candidate responses

### Training the Reward Model

```bash
# Add diverse training samples
python add_diverse_samples.py

# Train the reward model
python train_reward_sbert.py

# Test the pipeline
python test_rlhf_pipeline.py
```

### Admin Interface

Access the RLHF admin interface at `http://localhost:5000/admin_rlhf` to:
- View unrated interactions
- Provide feedback on responses
- Train new reward models
- Monitor model performance

## Testing

The project includes comprehensive test suites:

```bash
# Test core RAG functionality
python tests/test_complete_integration.py

# Test RLHF pipeline
python test_rlhf_pipeline.py

# Test Wikipedia and arXiv connectivity
python test_wiki_arxiv_connectivity.py

# Test PostgreSQL integration
python tests/test_postgres_integration.py

# Test patient context handling
python tests/patient_context_test.py
```

## Database Schema

### RLHF Tables

**rlhf_interactions**
- Stores all query-response pairs
- Tracks user feedback and ratings
- Links to session and patient context

**rlhf_reward_model_training**
- Records reward model training history
- Stores model metadata and performance metrics

**p_diagnosis** (Medical Database)
- Contains structured diagnosis information
- Linked to hospital and encounter data

## Configuration

### Disciplines Configuration

Configure medical disciplines in `config/disciplines.json`:

```json
{
  "disciplines": [
    {
      "id": "family_medicine",
      "name": "Family Medicine",
      "description": "Comprehensive primary healthcare",
      "is_default": true,
      "kb_path": "Organization_KB/Family_Medicine",
      "vector_db_path": "vector_dbs/organization/family_medicine"
    }
  ],
  "selection_rules": {
    "min_selections": 1,
    "max_selections": 3,
    "default_discipline": "family_medicine"
  }
}
```

## Docker Support

Build and run using Docker:

```bash
# Build the image
docker build -t medical-rag-app .

# Run the container
docker run -p 5000:5000 --env-file .env medical-rag-app
```

## Project Structure

```
Pulmonology_RAG_App/
├── main.py                      # Flask application entry point
├── integrated_rag.py            # Integrated RAG system
├── rag_architecture.py          # Two-store RAG implementation
├── tools.py                     # Tool definitions for agent
├── prompts.py                   # System prompts and templates
├── postgres_tool.py             # PostgreSQL database tool
├── model_utils.py               # RLHF model utilities
├── rlhf_reranker.py            # Response reranking logic
├── train_reward_sbert.py       # Reward model training
├── voice_diarization.py        # Audio processing utilities
├── azure_storage.py            # Azure cloud storage
├── extract_pdf.py              # PDF processing utilities
├── extract_html.py             # Web scraping utilities
├── setup_local_db.py           # Database initialization
├── setup_external_kb.py        # External KB setup
├── setup_organization_kb.py    # Organization KB setup
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── .env                        # Environment variables
├── config/
│   └── disciplines.json        # Medical disciplines config
├── KB/
│   ├── PDF/                    # Uploaded PDFs
│   └── URL/                    # Cached web content
├── Organization_KB/            # Organization-specific docs
├── vector_dbs/                 # Vector databases
├── templates/
│   ├── index.html             # Chat interface
│   └── admin_rlhf.html        # RLHF admin interface
└── tests/                      # Test suites
```

## Troubleshooting

### Common Issues

**Wikipedia/arXiv Timeout**
- Check network connectivity
- Increase timeout settings in tool configurations
- See `WIKI_ARXIV_TIMEOUT_FIX.md` in Resources folder

**PostgreSQL Connection Issues**
- Verify database credentials in `.env`
- Ensure PostgreSQL is running
- Check network accessibility to database host
- See `POSTGRESQL_TOOL_GUIDE.md` in Resources folder

**OpenAI API Errors**
- Check API key validity
- Monitor quota limits
- See `OPENAI_QUOTA_GUIDE.md` in Resources folder

**Voice Diarization Fails**
- Ensure Hugging Face token is set
- Install pyannote.audio dependencies
- Accept required model licenses on Hugging Face

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- LangChain for RAG framework
- OpenAI for language models
- ChromaDB for vector storage
- pyannote.audio for speaker diarization
- Hugging Face for model hosting

## Support

For issues and questions:
- Create an issue on GitHub
- Check documentation in the Resources folder
- Review test files for usage examples

## Version History

- **v1.0.0**: Initial release with core RAG functionality
- **v1.1.0**: Added RLHF system and reward model training
- **v1.2.0**: Integrated voice diarization and audio processing
- **v1.3.0**: Added PostgreSQL database tool and patient context
- **v1.4.0**: Multi-discipline organization knowledge base support

## Roadmap

- [ ] Add support for more medical databases
- [ ] Implement multi-modal RAG (images, charts)
- [ ] Enhance RLHF with active learning
- [ ] Add support for multiple languages
- [ ] Implement advanced citation tracking
- [ ] Add real-time collaboration features
