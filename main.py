from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
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
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from typing import List
import os
import glob
import traceback
from apify_client import ApifyClient
import whisper, torch




BASE_STORAGE_PATH = './KB/'
VECTOR_DB_PATH = './vector_dbs/'
# os.makedirs(BASE_STORAGE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
last_created_folder = None 


VECTOR_DBS_FOLDER = "./vector_dbs"

def get_timestamp():
    """Generate timestamp in MMDDYYYYHHMM format."""
    return time.strftime("%m%d%Y%H%M")

def get_latest_vector_db():
    """Finds the latest vector database in the vector_dbs folder."""
    vector_dbs = glob.glob(os.path.join(VECTOR_DBS_FOLDER, "*"))  # Adjust if needed

    if not vector_dbs:
        print("No existing vector DB found. A new one will be created.")
        return None

    latest_db = max(vector_dbs, key=os.path.getmtime)
    print(f"Using latest vector DB: {latest_db}")
    return latest_db

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

persist_directory = "./vector_db"


# Initialize LLM
llm = ChatOpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("base_url"),  # https://api.openai.com/v1
    model_name=os.getenv("llm_model_name")  # gpt-3.5-turbo
)

client = ApifyClient(os.getenv("apify_api_key"))  # Initialize Apify client

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


# if os.path.exists(persist_directory) and os.listdir(persist_directory):
#     print(f"Persist directory '{persist_directory}' found. Skipping embedding.")
#     vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# else:
#     print("Persist directory not found. Creating embeddings and initializing Chroma...")
#     vector_store = Chroma.from_documents(all_documents, embedding=embeddings, persist_directory=persist_directory)
    


# # Create retriever and QA chain
# retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
# qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Helper function to enhance response with citations
def enhance_with_citations(results):
    pdf_citations = set()  # Track unique PDF citations
    url_citations = set()  # Track unique URL citations

    for doc in results:
        metadata = getattr(doc, "metadata", {})  # Ensure metadata exists
        doc_type = metadata.get("type")  # Check for 'type'

        if doc_type == "pdf":
            pdf_source = metadata.get("source", "Unknown PDF")
            page_info = metadata.get("page", "Unknown Page")
            pdf_citations.add(f"PDF: {pdf_source} (Page {page_info})")

        elif doc_type == "url":
            url_source = metadata.get("source", "Unknown URL")
            url_citations.add(f"URL: {url_source}")

    # Combine citations
    return "\n".join(pdf_citations.union(url_citations)) or "No citations available"

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

# def extract_text_from_url(url):
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("start-maximized")
#     chrome_options.add_argument("--disable-blink-features=AutomationControlled")
#     chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
    
#     service = Service(executable_path="/usr/local/bin/chromedriver")
#     driver = webdriver.Chrome(service=service, options=chrome_options)
    
#     try:
#         driver.get(url)
#         html_content = driver.page_source
#         soup = BeautifulSoup(html_content, 'html.parser')
#         for script_or_style in soup(["script", "style"]):
#             script_or_style.decompose()
#         text = soup.get_text(separator=" ")
#         text = re.sub(r"[^\x00-\x7F]+", " ", text)
#         text = re.sub(r"\s+", " ", text).strip()
#         return text
#     finally:
#         driver.quit()



def extract_text_from_url(url):
    # Define run input with Playwright crawler and filtering
    run_input = {
        "startUrls": [{"url": url}],
        "useSitemaps": False,
        "respectRobotsTxtFile": True,
        "crawlerType": "playwright:adaptive",
        "includeUrlGlobs": [],
        "excludeUrlGlobs": [],
        "initialCookies": [],
        "proxyConfiguration": {"useApifyProxy": True},
        "keepElementsCssSelector": "",
        "removeElementsCssSelector": """nav, footer, script, style, noscript, svg, img[src^='data:'],
        [role=\"alert\"],
        [role=\"banner\"],
        [role=\"dialog\"],
        [role=\"alertdialog\"],
        [role=\"region\"][aria-label*=\"skip\" i],
        [aria-modal=\"true\"]""",
        "clickElementsCssSelector": "[aria-expanded=\"false\"]",
            }

    # Run the Apify actor
    run = client.actor("apify/website-content-crawler").call(run_input=run_input)
    
    # Collect and clean text content
    full_text = ""
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        page_text = item.get("text", "")
        page_text = re.sub(r"[^\x00-\x7F]+", " ", page_text)  # Remove non-ASCII
        page_text = re.sub(r"\s+", " ", page_text).strip()    # Normalize whitespace
        full_text += page_text + "\n"
    print(full_text)
    return full_text.strip()



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
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            refined_text = response.content.strip()
        else:
            refined_text = str(response).strip()

        return jsonify({"refined_text": refined_text})
    except Exception as e:
        return jsonify({"refined_text": "", "message": f"Error: {str(e)}"})



@app.route("/data", methods=["POST"])
def handle_query():
    user_input = request.json.get("data", "")
    if not user_input:
        return jsonify({"response": False, "message": "Please provide a valid input."})

    try:
        # Dynamically load the latest vector DB
        latest_vector_db = get_latest_vector_db() or "./vector_db"
        
        # Reinitialize vector store and retriever
        vector_store = Chroma(persist_directory=latest_vector_db, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Run the query through the QA chain
        response = qa_chain.invoke(user_input)
        print(response)
        search_results = retriever.invoke(user_input)
        print(search_results)
        citations = enhance_with_citations(search_results)

        message = f"{response['result']}\n\nCitations:\n{citations}"     
        return jsonify({"response": True, "message": message})

    except Exception as e:
        return jsonify({"response": False, "message": f"Error: {str(e)}"})


def initialize_session(user="guest"):
    """Initializes a new session folder when the page is refreshed."""
    global last_created_folder
    timestamp = get_timestamp()
    last_created_folder = f"{user}_{timestamp}"  # Format: user_MMDDYYYYHHMM

    # Ensure required directories exist
    os.makedirs(os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder), exist_ok=True)
    os.makedirs(os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder), exist_ok=True)

    print(f"📂 New session folder created: {last_created_folder}")
    return last_created_folder


@app.route('/', methods=['GET'])
def index():
    """Refresh page to create a new session folder."""
    user = request.args.get('user', 'guest')
    initialize_session(user)
    return render_template("index.html")


def count_files_in_folder(folder):
    """Returns the number of files in a given folder."""
    if os.path.exists(folder):
        return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    return 0

def can_upload_more_files(new_files_count):
    """Check if the total PDFs and URLs in the session folder exceed the limit (10)."""
    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
    url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
    
    pdf_count = count_files_in_folder(pdf_folder)
    url_count = count_files_in_folder(url_folder)
    
    return (pdf_count + url_count + new_files_count) <= 10  # Limit is 10 files total

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload multiple PDF files to the session folder."""
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist('file')  # Get multiple files
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No valid files uploaded"}), 400

    if not can_upload_more_files(len(files)):
        return jsonify({"message": "Cannot process: Maximum of 10 PDFs exceeded."})

    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
    os.makedirs(pdf_folder, exist_ok=True)

    saved_files = []
    
    try:
        for file in files:
            file_path = os.path.join(pdf_folder, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

        return jsonify({"message": "PDFs uploaded successfully", "files": saved_files})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/upload_url', methods=['POST'])
def upload_url():
    """Upload a text file containing URLs."""
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400
    


    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file uploaded"}), 400

        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
        os.makedirs(url_folder, exist_ok=True)
        file_path = os.path.join(url_folder, "urls.txt")
        file.save(file_path)

        # Read URLs
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]

        if not urls:
            return jsonify({"error": "Uploaded file contains no valid URLs"}), 400
        if len(urls) > 3:
            return jsonify({"message": "Cannot process: Maximum of 3 URLs exceeded."})

        return jsonify({"message": "URLs uploaded successfully", "file": file_path, "url_count": len(urls)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route('/create_vector_db', methods=['POST'])
def create_vector_db():
    """Parse PDFs and URLs, then create a Chroma Vector Database."""
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    try:
        persist_dir = os.path.join(VECTOR_DB_PATH, last_created_folder)
        os.makedirs(persist_dir, exist_ok=True)

        pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)

        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')] if os.path.exists(pdf_folder) else []
        url_files = [os.path.join(url_folder, f) for f in os.listdir(url_folder)] if os.path.exists(url_folder) else []

        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128)

        for pdf_file in pdf_files:
                text_content = extract_text_from_pdf(pdf_file)
                if not text_content:
                    print(f"No text extracted from {pdf_file}")
                    continue

                if isinstance(text_content, str):  # Convert single string to a list
                    text_content = [text_content]
                print(text_content)
                for text in text_content:
                    chunks = text_splitter.split_text(text)
                    for page_num, chunk in enumerate(chunks, start=1):  # `enumerate` starts from 1
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": pdf_file,  # File path
                                "type": "pdf",        # Ensure type is set
                                "page": page_num      # Correct page number
                            }
                        ))


        #  Process URLs
        for url_file in url_files:
            with open(url_file, 'r', encoding='utf-8') as file:
                urls = [line.strip() for line in file.readlines() if line.strip()]

            for url in urls:
                try:
                    text = extract_text_from_url(url)
                    if not text:
                        print(f" No text extracted from {url}")
                        continue

                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": url,
                                "type": "url"
                            }
                        ))


                except Exception as e:
                    print(f"Error extracting text from {url}: {str(e)}")

        if not documents:
            return jsonify({"error": "No valid documents found to create the database"}), 400
        
        print(f"{len(documents)} documents prepared for vectorization")

        # Convert to Chroma format
        vector_store = Chroma.from_documents(
            documents,  # Use the list of Document objects directly
            embedding=embeddings,
            persist_directory=persist_dir
        )

        return jsonify({"message": "New Vector DB created", "db": last_created_folder})

    except Exception as e:
        print(f" Error in /create_vector_db: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500





if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000, use_reloader=False)
