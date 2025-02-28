# Pulmonology Q&A Bot

This repository contains a Flask-based web application that integrates a LangChain-powered Q&A system for pulmonology-related queries. The application leverages OpenAI's LLM and FAISS for information retrieval, providing expert-level responses to user questions.

## Features

- **Pulmonology Expertise**: Offers answers to questions specifically in the domain of pulmonology.
- **Interactive Web Interface**: Users can interact with the bot using a visually appealing HTML front-end built with Flowbite CSS.
- **Efficient Information Retrieval**: Uses FAISS vector store for fast and accurate document retrieval.
- **Dynamic and Scalable**: Supports chunk-based text processing and citation-based enhanced responses.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pulmonology-qna-bot.git
   cd pulmonology-qna-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in a `.env` file:
   ```env
   openai_api_key=YOUR_OPENAI_API_KEY
   base_url=YOUR_OPENAI_BASE_URL
   llm_model_name=YOUR_LLM_MODEL_NAME
   embedding_model_name=YOUR_EMBEDDING_MODEL_NAME
   ```

5. Prepare your metadata files:
   - Place your `pdf_metadata.json` and `url_metadata.json` files in the project root.

## Usage

1. Start the Flask application:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Type your pulmonology-related questions into the chat interface and receive expert responses.

## Project Structure

```plaintext
pulmonology-qna-bot/
├── static/         # Static assets (CSS, JS)
├── templates/      # HTML templates
├── app.py          # Main Flask application
├── requirements.txt # Python dependencies
├── pdf_metadata.json # PDF metadata input
├── url_metadata.json # URL metadata input
├── .env            # Environment variables
└── README.md       # Project documentation
```

## Technical Overview

### Front-End
- Built using HTML, CSS, and JavaScript.
- Flowbite CSS library for responsive and modern UI components.

### Back-End
- Flask for serving the web application.
- LangChain framework for constructing the Q&A pipeline.
- FAISS for vector-based document retrieval.

### Data Processing
- Supports metadata from both PDFs and URLs.
- Uses LangChain's `CharacterTextSplitter` for text chunking.

## Enhancements with Citations
The application enriches responses with citations from the underlying documents, helping users verify the source of information.

## Dependencies

- Python 3.10+
- Flask
- LangChain
- FAISS
- OpenAI API

Refer to `requirements.txt` for the complete list.

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- OpenAI for their robust APIs.
- Flowbite for the excellent UI components.
- LangChain for enabling advanced document processing.
