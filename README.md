# Pulmonology RAG App

## Overview
Pulmonology RAG App is a Retrieval-Augmented Generation (RAG) chatbot designed to enhance patient care services by providing accurate and reliable responses based on medical documents and online resources. The application leverages OpenAI's language models, a Chroma vector store, and a Flask web interface for interaction.

## Features

### 🔹 Web-based Chat Interface
- User-friendly chat UI built with HTML, CSS, and JavaScript.
- Sidebar with options for New Chat, Chat History, and Settings.
- Editable user messages with options to refine inputs.

### 🔹 Retrieval-Augmented Generation (RAG) System
- Uses OpenAI's LLM for natural language understanding.
- Retrieves relevant information from indexed PDF and web data.
- Enhances AI-generated responses with source citations.

### 🔹 Voice-to-Text Support
- Integrates Whisper AI for audio transcription.
- Allows users to input queries via voice recording.
- Converts transcribed audio into text for further processing.

### 🔹 Document and URL Processing
- Extracts text content from PDFs and web pages.
- Splits content into manageable chunks for indexing.
- Stores indexed data in a Chroma vector database.

### 🔹 Query Handling
- Processes user input and retrieves contextually relevant responses.
- Provides citations from source documents to ensure accuracy.
- Limits conversation history to optimize response quality.

### 🔹 Chat Session Management
- Allows users to save and download chat history in JSON format.
- Enforces a conversation limit, prompting users to save sessions before continuing.

## Installation

### Prerequisites
- Python 3.8+
- Flask
- OpenAI API Key
- ChromaDB
- Whisper AI

### Steps
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/pulmonology-rag-app.git
   cd pulmonology-rag-app
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```sh
   export OPENAI_API_KEY='your_openai_api_key'
   export BASE_URL='your_openai_base_url'
   export LLM_MODEL_NAME='your_model_name'
   export EMBEDDING_MODEL_NAME='your_embedding_model'
   ```
4. Run the application:
   ```sh
   python Pulmonology_RAG_App/main.py
   ```
5. Access the web interface at:
   ```
   http://localhost:5000
   ```

## Usage
- Type a query or use voice input to interact with the chatbot.
- Retrieve AI-generated responses with citations from medical documents.
- Edit and refine messages before sending.
- Save chat history for future reference.

## Future Enhancements
- Integrating more document types (e.g., medical journals, databases).
- Improving UI with advanced chat features.
- Deploying as a cloud-based service.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Feel free to submit issues and pull requests to improve the app!
