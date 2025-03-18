from flask import Flask, request, jsonify, render_template
import os
import json
import shutil
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)

BASE_STORAGE_PATH = "./KB"
VECTOR_DB_PATH = "./vector_db"
current_db = None  # Holds the latest active DB

# Ensure storage directories exist
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Load OpenAI embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model=os.getenv('embedding_model_name')
)

# Helper function to get timestamp
def get_timestamp():
    return datetime.now().strftime("%m%d%Y%H%M")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    user = request.form.get('user', 'guest')
    timestamp = get_timestamp()
    folder_path = os.path.join(BASE_STORAGE_PATH, "PDF", f"{user}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    
    files = request.files.getlist('files')
    if not files or len(files) > 10:
        return jsonify({"error": "You can upload up to 10 PDFs."}), 400
    
    for file in files:
        file.save(os.path.join(folder_path, file.filename))
    
    return jsonify({"message": "PDFs uploaded successfully", "folder": folder_path})

@app.route('/upload_url', methods=['POST'])
def upload_url():
    user = request.form.get('user', 'guest')
    timestamp = get_timestamp()
    folder_path = os.path.join(BASE_STORAGE_PATH, "URL", f"{user}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)
    
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded."}), 400
    
    file.save(os.path.join(folder_path, file.filename))
    
    return jsonify({"message": "URLs file uploaded successfully", "folder": folder_path})

@app.route('/create_vector_db', methods=['POST'])
def create_vector_db():
    global current_db
    user = request.form.get('user', 'guest')
    timestamp = get_timestamp()
    db_name = f"{user}_{timestamp}"
    persist_dir = os.path.join(VECTOR_DB_PATH, db_name)
    
    os.makedirs(persist_dir, exist_ok=True)
    
    # Load PDFs and URLs
    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", f"{user}_{timestamp}")
    url_folder = os.path.join(BASE_STORAGE_PATH, "URL", f"{user}_{timestamp}")
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder)] if os.path.exists(pdf_folder) else []
    url_files = [os.path.join(url_folder, f) for f in os.listdir(url_folder)] if os.path.exists(url_folder) else []
    
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128)
    
    for pdf_file in pdf_files:
        with open(pdf_file, 'r') as file:
            text = file.read()
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": pdf_file}))
    
    for url_file in url_files:
        with open(url_file, 'r') as file:
            urls = file.readlines()
            for url in urls:
                documents.append(Document(page_content=url.strip(), metadata={"source": "URL"}))
    
    vector_store = Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)
    current_db = persist_dir  # Update active DB
    
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
