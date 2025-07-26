from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile
import traceback
import whisper
import torch

from config import BASE_STORAGE_PATH, VECTOR_DB_PATH
from helpers import (
    get_timestamp,
    get_latest_vector_db,
    load_metadata,
    process_pdf_metadata,
    process_url_metadata,
    enhance_with_citations,
    can_upload_more_files,
)
from document_utils import extract_text_from_pdf, extract_text_from_url

load_dotenv()

app = Flask(__name__)

# Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

# Initialize language models
llm = ChatOpenAI(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("base_url"),
    model_name=os.getenv("llm_model_name"),
)
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("openai_api_key"),
    base_url=os.getenv("base_url"),
    model=os.getenv("embedding_model_name"),
)

last_created_folder = None


# --------- Metadata Loading ---------
pdf_metadata = load_metadata("pdf_metadata.json")
url_metadata = load_metadata("url_metadata.json")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,
    chunk_overlap=128,
    separators=["\n\n", "\n", ".", " "],
)

pdf_documents = process_pdf_metadata(pdf_metadata, text_splitter)
url_documents = process_url_metadata(url_metadata, text_splitter)
all_documents = pdf_documents + url_documents


# --------- Routes ---------
@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.save(temp.name)
        result = model.transcribe(temp.name)
    return jsonify({"text": result["text"]})


@app.route("/plain_english", methods=["POST"])
def plain_english():
    user_text = request.json.get("text", "")
    if not user_text:
        return jsonify({"refined_text": "", "message": "No input provided."})
    try:
        prompt = f"Rewrite the following question in plain English for better clarity:\n\n{user_text}"
        response = llm.invoke(prompt)
        refined_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
        return jsonify({"refined_text": refined_text})
    except Exception as e:
        return jsonify({"refined_text": "", "message": f"Error: {str(e)}"})


@app.route("/data", methods=["POST"])
def handle_query():
    user_input = request.json.get("data", "")
    if not user_input:
        return jsonify({"response": False, "message": "Please provide a valid input."})
    try:
        latest_vector_db = get_latest_vector_db() or "./vector_db"
        vector_store = Chroma(persist_directory=latest_vector_db, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.invoke(user_input)
        search_results = retriever.invoke(user_input)
        citations = enhance_with_citations(search_results)
        message = f"{response['result']}\n\nCitations:\n{citations}"
        return jsonify({"response": True, "message": message})
    except Exception as e:
        return jsonify({"response": False, "message": f"Error: {str(e)}"})


# --------- Session Management ---------
def initialize_session(user: str = "guest"):
    global last_created_folder
    timestamp = get_timestamp()
    last_created_folder = f"{user}_{timestamp}"
    os.makedirs(os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder), exist_ok=True)
    os.makedirs(os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder), exist_ok=True)
    return last_created_folder


@app.route("/", methods=["GET"])
def index():
    user = request.args.get("user", "guest")
    initialize_session(user)
    return render_template("index.html")


@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No valid files uploaded"}), 400

    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
    url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
    if not can_upload_more_files(pdf_folder, url_folder, len(files)):
        return jsonify({"message": "Cannot process: Maximum of 10 PDFs exceeded."})

    os.makedirs(pdf_folder, exist_ok=True)
    saved_files = []
    try:
        for file in files:
            file_path = os.path.join(pdf_folder, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)
        return jsonify({"message": "PDFs uploaded successfully", "files": saved_files})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/upload_url", methods=["POST"])
def upload_url():
    global last_created_folder
    if not last_created_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file uploaded"}), 400
        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
        os.makedirs(url_folder, exist_ok=True)
        file_path = os.path.join(url_folder, "urls.txt")
        file.save(file_path)
        with open(file_path, "r") as f:
            urls = [line.strip() for line in f.readlines() if line.strip()]
        if not urls:
            return jsonify({"error": "Uploaded file contains no valid URLs"}), 400
        if len(urls) > 3:
            return jsonify({"message": "Cannot process: Maximum of 3 URLs exceeded."})
        return jsonify({"message": "URLs uploaded successfully", "file": file_path, "url_count": len(urls)})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/create_vector_db", methods=["POST"])
def create_vector_db():
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
                continue
            if isinstance(text_content, str):
                text_content = [text_content]
            for text in text_content:
                chunks = text_splitter.split_text(text)
                for page_num, chunk in enumerate(chunks, start=1):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": pdf_file, "type": "pdf", "page": page_num},
                        )
                    )

        for url_file in url_files:
            with open(url_file, "r", encoding="utf-8") as file:
                urls = [line.strip() for line in file.readlines() if line.strip()]
            for url in urls:
                try:
                    text = extract_text_from_url(url)
                    if not text:
                        continue
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        documents.append(
                            Document(page_content=chunk, metadata={"source": url, "type": "url"})
                        )
                except Exception as e:
                    print(f"Error extracting text from {url}: {str(e)}")

        if not documents:
            return jsonify({"error": "No valid documents found to create the database"}), 400

        Chroma.from_documents(documents, embedding=embeddings, persist_directory=persist_dir)
        return jsonify({"message": "New Vector DB created", "db": last_created_folder})
    except Exception as e:
        print(f"Error in /create_vector_db: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000, use_reloader=False)
