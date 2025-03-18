from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import whisper
import tempfile
import fitz  # PyMuPDF
import pdfplumber
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from typing import List



BASE_STORAGE_PATH = './KB/'
VECTOR_DB_PATH = './vector_dbs/'
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)



model = whisper.load_model("medium")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

persist_directory = "./vector_db"

# Initialize LLM
llm = OpenAI(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model_name=os.getenv('llm_model_name')
)


def get_timestamp():
    return datetime.now().strftime('%m%d%Y%H%M')


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

def process_pdf_metadata(pdf_metadata: list, text_splitter) -> list:
    """Chunk the text content from PDF metadata."""
    chunked_documents = []

    for doc in pdf_metadata:
        file_name = doc.get("file_name", "Unknown File")
        pages = doc.get("text", [])
        
        for page in pages:
            page_number = page.get("page", "Unknown Page")
            page_text = page.get("text", "").strip()

            # Skip empty pages
            if not page_text:
                continue

            # Split text and create chunks
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file_name, "type": "pdf", "page": page_number}
                    )
                )

    return chunked_documents

import json

def process_url_metadata(url_metadata: list, text_splitter) -> list:
    """Chunk the text content from URL metadata."""
    chunked_documents = []

    for entry in url_metadata:
        url = entry.get("url", "Unknown URL")
        text_content = entry.get("text", "").strip()
        date_info = entry.get("date", "Unknown Date")

        # Skip entries with empty text
        if not text_content:
            continue

        # Split the content and create document chunks
        chunks = text_splitter.split_text(text_content)
        for chunk in chunks:
            chunked_documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": url, "type": "url", "date": date_info}
                )
            )

    return chunked_documents


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
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096, 
        chunk_overlap=128,
        separators=["\n\n", "\n", ".", " "]
    )

pdf_documents = process_pdf_metadata(pdf_metadata, text_splitter)
url_documents = process_url_metadata(url_metadata,text_splitter)
all_documents = pdf_documents + url_documents


if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print(f"Persist directory '{persist_directory}' found. Skipping embedding.")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    print("Persist directory not found. Creating embeddings and initializing Chroma...")
    vector_store = Chroma.from_documents(all_documents, embedding=embeddings, persist_directory=persist_directory)
    


# Create retriever and QA chain
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Helper function to enhance response with citations
def enhance_with_citations(results):
    pdf_citations = set()  # Track unique PDF citations
    url_citations = set()  # Track unique URL citations

    for doc in results:
        metadata = doc.metadata
        if metadata.get("type") == "pdf":
            pdf_source = metadata.get('source', 'Unknown PDF')
            page_info = metadata.get('page', 'Unknown')
            pdf_citations.add(f"PDF: {pdf_source} (Page {page_info})")
        elif metadata.get("type") == "url":
            url_source = metadata.get('source', 'Unknown URL')
            url_citations.add(f"URL: {url_source}")

    # Join unique citations
    citations = "\n".join(pdf_citations.union(url_citations))
    return citations


@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)
        result = model.transcribe(temp.name)
    return jsonify({"text": result['text']})

@app.route("/plain_english", methods=["POST"])
def plain_english():
    user_text = request.json.get("text", "")
    if not user_text:
        return jsonify({"refined_text": "", "message": "No input provided."})

    try:
        prompt = f"Rewrite the following question in plain English for better clarity:\n\n{user_text}"
        refined_text = llm.invoke(prompt)  

        return jsonify({"refined_text": refined_text})
    except Exception as e:
        return jsonify({"refined_text": "", "message": f"Error: {str(e)}"})


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
    

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    user = request.form.get('user', 'guest')
    timestamp = get_timestamp()
    folder_path = os.path.join(BASE_STORAGE_PATH, "PDF", f"{user}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    
    files = request.files.getlist('files')
    print(f"Received {len(files)} PDF files for user {user} at {folder_path}")

    if not files or len(files) > 10:
        print("Error: No files uploaded or more than 10 PDFs.")
        return jsonify({"error": "You can upload up to 10 PDFs."}), 400
    
    for file in files:
        file_path = os.path.join(folder_path, file.filename)
        file.save(file_path)
        print(f"Saved PDF: {file_path}")
    
    return jsonify({"message": "PDFs uploaded successfully", "folder": folder_path})


@app.route('/upload_url', methods=['POST'])
def upload_url():
    user = request.form.get('user', 'guest')
    timestamp = get_timestamp()
    folder_path = os.path.join(BASE_STORAGE_PATH, "URL", f"{user}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    
    file = request.files.get('file')
    print(f"Received URL file for user {user} at {folder_path}")

    if not file:
        print("Error: No URL file uploaded.")
        return jsonify({"error": "No file uploaded."}), 400
    
    file_path = os.path.join(folder_path, file.filename)
    file.save(file_path)
    print(f"Saved URL file: {file_path}")
    
    return jsonify({"message": "URLs uploaded successfully", "folder": folder_path})


def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?\'":;\-\s]', '', text)
    text = text.strip()
    text = re.sub(r'\.{2,}', '.', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

def extract_text_from_pdf(pdf_file):
    text_content = []
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            clean_text = clean_extracted_text(page.get_text())
            text_content.append(clean_text)
    return text_content

def extract_text_from_url(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    
    service = Service(executable_path="/usr/local/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(url)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=" ")
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    finally:
        driver.quit()

@app.route('/create_vector_db', methods=['POST'])
def create_vector_db():
    global current_db
    user = request.form.get('user', 'guest')
    timestamp = get_timestamp()
    db_name = f"{user}_{timestamp}"
    persist_dir = os.path.join(VECTOR_DB_PATH, db_name)
    os.makedirs(persist_dir, exist_ok=True)
    
    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", f"{user}_{timestamp}")
    url_folder = os.path.join(BASE_STORAGE_PATH, "URL", f"{user}_{timestamp}")
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder)] if os.path.exists(pdf_folder) else []
    url_files = [os.path.join(url_folder, f) for f in os.listdir(url_folder)] if os.path.exists(url_folder) else []
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128)
    
    for pdf_file in pdf_files:
        text_content = extract_text_from_pdf(pdf_file)
        if not text_content:
            print(f"⚠️ Warning: No text extracted from {pdf_file}")
        for text in text_content:
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": pdf_file}))
    
    for url_file in url_files:
        with open(url_file, 'r') as file:
            urls = file.readlines()
            for url in urls:
                text = extract_text_from_url(url.strip())
                if not text:
                    print(f"⚠️ Warning: No text extracted from {url}")
                documents.append(Document(page_content=text, metadata={"source": "URL"}))
    
    if not documents:
        print("No valid text chunks found. Cannot create vector DB.")
        return jsonify({"error": "No valid documents found to create the database"}), 400
    
    vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)
    current_db = persist_dir  
    
    return jsonify({"message": "New Vector DB created", "db": db_name})


@app.route('/set_active_db', methods=['POST'])
def set_active_db():
    global current_db
    db_name = request.json.get('db_name')
    new_db_path = os.path.join(VECTOR_DB_PATH, db_name)
    if os.path.exists(new_db_path):
        current_db = new_db_path
        return jsonify({"message": "Active DB set successfully"})
    return jsonify({"error": "DB not found"}), 400



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
