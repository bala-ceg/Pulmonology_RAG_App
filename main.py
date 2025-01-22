from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter,TextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import json
from typing import List

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

class ParagraphTextSplitter(TextSplitter):
    """Custom text splitter that splits documents by paragraphs."""

    def split_text(self, text: str) -> List[str]:
        """Split text into paragraphs using double newline as delimiter."""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


# Initialize LLM
llm = OpenAI(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model_name=os.getenv('llm_model_name')
)


# Step 1: Load and Process Metadata
def load_metadata(file_path: str) -> List[dict]:
    """Load JSON metadata from the given file path."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []


def process_pdf_metadata(pdf_metadata: List[dict], text_splitter: TextSplitter) -> List[Document]:
    """Chunk the text content from PDF metadata using the provided text splitter."""
    chunked_documents = []
    for pdf in pdf_metadata:
        file_name = pdf.get("file_name", "Unknown File")
        for page in pdf.get("text", []):
            if isinstance(page, dict) and "text" in page:
                page_content = page["text"]
                chunks = text_splitter.split_text(page_content)
                for chunk in chunks:
                    chunked_documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": file_name, "type": "pdf", "page": page.get("page", "Unknown Page")}
                        )
                    )
    return chunked_documents


def process_url_metadata(url_metadata: List[dict]) -> List[Document]:
    """Process URL metadata with paragraph-based splitting and chunking."""
    paragraph_splitter = ParagraphTextSplitter()
    character_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=128)

    documents = []
    for url_data in url_metadata:
        url = url_data.get("url", "Unknown URL")
        text_content = url_data.get("text", "")

        # Split text into paragraphs
        paragraphs = paragraph_splitter.split_text(text_content)

        # Further split paragraphs into smaller chunks
        for paragraph in paragraphs:
            chunks = character_splitter.split_text(paragraph)
            for chunk in chunks:
                documents.append(
                    Document(page_content=chunk, metadata={"source": url, "type": "url"})
                )
    return documents

# Load and process metadata
pdf_metadata = load_metadata("pdf_metadata.json")
url_metadata = load_metadata("url_metadata.json")


# Create embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model=os.getenv('embedding_model_name')
)

# Split documents and create FAISS vector store
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
pdf_documents = process_pdf_metadata(pdf_metadata, text_splitter)
url_documents = process_url_metadata(url_metadata)
all_documents = pdf_documents + url_documents
vector_store = Chroma.from_documents(all_documents, embedding=embeddings)


# Create retriever and QA chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Helper function to enhance response with citations
def enhance_with_citations(results):
    pdf_citations = []
    url_citations = []
    for doc in results:
        metadata = doc.metadata
        if metadata.get("type") == "pdf":
            pdf_citations.append(f"PDF: {metadata.get('source', 'Unknown PDF')} (Page {metadata.get('page', 'Unknown')})")
        elif metadata.get("type") == "url":
            url_citations.append(f"URL: {metadata.get('source', 'Unknown URL')}")
    citations = "\n".join(pdf_citations + url_citations)
    return citations

@app.route("/")
def index():
    return render_template("index.html")  

@app.route("/data", methods=["POST"])
def handle_query():
    user_input = request.json.get("data", "")
    if not user_input:
        return jsonify({"response": False, "message": "Please provide a valid input."})

    # Run the query through the QA chain
    try:
        response = qa_chain.invoke(user_input)
        search_results = retriever.invoke(user_input)
        citations = enhance_with_citations(search_results)
        message = f"{response['result']}\n\nCitations:\n{citations}"
        return jsonify({"response": True, "message": message})
    except Exception as e:
        return jsonify({"response": False, "message": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
