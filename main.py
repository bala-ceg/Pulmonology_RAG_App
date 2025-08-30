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
import io
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
import psycopg
import asyncio
import threading

# Import new utilities
try:
    import importlib
    import azure_storage
    importlib.reload(azure_storage)  # Force reload the module
    from azure_storage import get_storage_manager
    AZURE_AVAILABLE = True
except ImportError:
    print("Azure storage not available. Install with: pip install azure-storage-blob")
    AZURE_AVAILABLE = False

try:
    from voice_diarization import get_diarization_processor
    DIARIZATION_AVAILABLE = True
except ImportError:
    print("Voice diarization not available. Install dependencies: pip install pyannote.audio torch")
    DIARIZATION_AVAILABLE = False
from psycopg import sql




BASE_STORAGE_PATH = './KB/'
VECTOR_DB_PATH = './vector_dbs/'
ORGANIZATION_KB_PATH = './Organization_KB/'
ORGANIZATION_VECTOR_DB_PATH = './vector_dbs/organization/'

# Create required directories
os.makedirs(BASE_STORAGE_PATH, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(ORGANIZATION_KB_PATH, exist_ok=True)
os.makedirs(ORGANIZATION_VECTOR_DB_PATH, exist_ok=True)

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

# Database configuration
db_config = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

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

# Load disciplines configuration
def load_disciplines_config():
    """Load disciplines configuration from JSON file."""
    try:
        with open("config/disciplines.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Warning: disciplines.json not found. Using default configuration.")
        return {
            "disciplines": [
                {
                    "id": "family_medicine",
                    "name": "Family Medicine", 
                    "description": "Comprehensive primary healthcare",
                    "is_default": True,
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

# Load configuration
disciplines_config = load_disciplines_config()

class MedicalQueryRouter:
    """Intelligent router that determines which medical disciplines are relevant for a query."""
    
    def __init__(self, llm, disciplines_config):
        self.llm = llm
        self.disciplines = disciplines_config.get("disciplines", [])
        self.discipline_keywords = self._build_keyword_map()
    
    def _build_keyword_map(self):
        """Build a map of keywords for each discipline."""
        keyword_map = {}
        
        # Medical specialty keywords
        specialty_keywords = {
            "family_medicine": [
                "primary care", "general practice", "family doctor", "annual checkup", "preventive care",
                "common cold", "flu", "hypertension", "diabetes", "vaccination", "routine care",
                "wellness exam", "physical exam", "blood pressure", "cholesterol", "general health"
            ],
            "cardiology": [
                "heart", "cardiac", "cardiovascular", "chest pain", "heart attack", "myocardial infarction",
                "heart failure", "arrhythmia", "atrial fibrillation", "coronary", "angina", "pacemaker",
                "cardiologist", "EKG", "ECG", "echocardiogram", "blood pressure", "hypertension",
                "heart rate", "cardiac arrest", "valve", "aorta", "coronary artery"
            ],
            "neurology": [
                "brain", "neurological", "nervous system", "stroke", "seizure", "epilepsy", "migraine",
                "headache", "Parkinson's", "Alzheimer's", "dementia", "multiple sclerosis", "MS",
                "neurologist", "MRI brain", "CT brain", "memory loss", "confusion", "dizziness",
                "numbness", "tingling", "weakness", "paralysis", "spinal cord", "nerve"
            ],
            "doctors_files": [
                "my files", "my documents", "uploaded", "document", "file", "PDF", "article",
                "my upload", "personal documents", "doctor's files", "my records", "uploaded content",
                "session files", "my PDFs", "document I uploaded", "file I shared", "my data"
            ]
        }
        
        return specialty_keywords
    
    def _has_session_files(self):
        """Check if the current session has uploaded files."""
        global last_created_folder
        if not last_created_folder:
            return False
        
        # Check for PDFs in session
        pdf_path = os.path.join(BASE_STORAGE_PATH, "PDF", last_created_folder)
        url_path = os.path.join(BASE_STORAGE_PATH, "URL", last_created_folder)
        
        pdf_files = []
        url_files = []
        
        if os.path.exists(pdf_path):
            pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
        
        if os.path.exists(url_path):
            url_files = [f for f in os.listdir(url_path) if f.endswith('.txt')]
        
        return len(pdf_files) > 0 or len(url_files) > 0
    
    def analyze_query(self, query):
        """Analyze query and determine relevant disciplines using AI + keywords."""
        query_lower = query.lower()
        
        # First, use keyword matching for quick routing
        relevant_disciplines = []
        confidence_scores = {}
        
        for discipline_id, keywords in self.discipline_keywords.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
            if keyword_matches > 0:
                confidence = min(keyword_matches / len(keywords) * 100, 95)  # Cap at 95%
                relevant_disciplines.append(discipline_id)
                confidence_scores[discipline_id] = confidence
        
        # Special handling for doctors_files - include if user has uploaded files and query might be relevant
        has_files = self._has_session_files()
        if has_files and "doctors_files" not in relevant_disciplines:
            # Check if query could be asking about user's files (more lenient keywords)
            user_file_keywords = ["my", "document", "file", "upload", "PDF", "article", "personal", "doctor", "record"]
            if any(keyword in query_lower for keyword in user_file_keywords):
                relevant_disciplines.append("doctors_files")
                confidence_scores["doctors_files"] = 85  # High confidence for user file queries
        
        # If no keyword matches, use AI to analyze
        if not relevant_disciplines:
            relevant_disciplines = self._ai_analyze_query(query)
            for discipline in relevant_disciplines:
                confidence_scores[discipline] = 70  # Default AI confidence
            
            # Add doctors_files to AI analysis if user has files
            if has_files and "doctors_files" not in relevant_disciplines:
                relevant_disciplines.append("doctors_files")
                confidence_scores["doctors_files"] = 75
        
        # Ensure we have at least one discipline (default to family medicine)
        if not relevant_disciplines:
            relevant_disciplines = ["family_medicine"]
            confidence_scores["family_medicine"] = 60
        
        # Sort by confidence
        relevant_disciplines.sort(key=lambda d: confidence_scores.get(d, 0), reverse=True)
        
        return {
            "disciplines": relevant_disciplines[:2],  # Limit to top 3
            "confidence_scores": confidence_scores,
            "routing_method": "hybrid" if len(relevant_disciplines) > 0 else "default"
        }
    
    def _ai_analyze_query(self, query):
        """Use AI to analyze query when keyword matching fails."""
        try:
            discipline_names = [d["name"] for d in self.disciplines]
            
            prompt = f"""
            Analyze this medical query and determine which medical specialties are most relevant:
            
            Query: "{query}"
            
            Available specialties: {', '.join(discipline_names)}
            
            Guidelines:
            - If the query mentions "my files", "my documents", "uploaded", or refers to user's personal documents, include "Doctor's Files"
            - If the query is general or could apply to multiple specialties, include Family Medicine
            - If unclear, default to Family Medicine
            - Consider that "Doctor's Files" contains user-uploaded PDFs and documents
            
            Return only the specialty names that are relevant, separated by commas.
            Response format: Specialty1, Specialty2 (max 3)
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse AI response and map to discipline IDs
            ai_specialties = [s.strip() for s in content.split(',')]
            relevant_disciplines = []
            
            for specialty in ai_specialties:
                for discipline in self.disciplines:
                    if discipline["name"].lower() in specialty.lower():
                        relevant_disciplines.append(discipline["id"])
                        break
            
            return relevant_disciplines
            
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return ["family_medicine"]  # Fallback

# Initialize the router
medical_router = MedicalQueryRouter(llm, disciplines_config)

def get_available_disciplines():
    """Return list of available disciplines for UI dropdown."""
    return disciplines_config.get("disciplines", [])

def validate_discipline_selection(selected_disciplines):
    """Validate user's discipline selection against rules."""
    rules = disciplines_config.get("selection_rules", {})
    min_sel = rules.get("min_selections", 1)
    max_sel = rules.get("max_selections", 3)
    
    if len(selected_disciplines) < min_sel:
        return False, f"Please select at least {min_sel} discipline(s)"
    if len(selected_disciplines) > max_sel:
        return False, f"Please select no more than {max_sel} discipline(s)"
    
    # Validate discipline IDs exist
    valid_ids = [d["id"] for d in disciplines_config.get("disciplines", [])]
    invalid_ids = [d for d in selected_disciplines if d not in valid_ids]
    if invalid_ids:
        return False, f"Invalid discipline(s): {', '.join(invalid_ids)}"
    
    return True, "Valid selection"

def get_discipline_vector_db_path(discipline_id):
    """Get vector database path for a specific discipline."""
    for discipline in disciplines_config.get("disciplines", []):
        if discipline["id"] == discipline_id:
            return discipline.get("vector_db_path", "")
    return None

def create_organization_vector_db(discipline_id, documents):
    """Create or update organization vector database for a specific discipline."""
    vector_db_path = get_discipline_vector_db_path(discipline_id)
    if not vector_db_path:
        raise ValueError(f"Unknown discipline: {discipline_id}")
    
    persist_dir = os.path.join(".", vector_db_path)
    os.makedirs(persist_dir, exist_ok=True)
    
    # Create or update the vector store
    vector_store = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vector_store

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
    org_citations = set()  # Track unique Organization KB citations

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
            
        elif doc_type == "organization_pdf":
            org_source = metadata.get("source", "Unknown Document")
            discipline = metadata.get("discipline", "Unknown Discipline")
            page_info = metadata.get("page", "Unknown Page")
            org_citations.add(f"Organization KB - {discipline.replace('_', ' ').title()}: {org_source} (Page {page_info})")

    # Combine citations
    all_citations = pdf_citations.union(url_citations).union(org_citations)
    return "\n".join(all_citations) or "No citations available"

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

@app.route('/transcribe_patient_notes', methods=['POST'])
def transcribe_patient_notes():
    """
    Transcribe patient audio recording and generate medical summary.
    Expects: audio file, doctor_name, patient_name
    Returns: transcribed text, summary, and conclusion
    """
    try:
        # Get form data
        audio_file = request.files.get('audio')
        doctor_name = request.form.get('doctor_name', '')
        patient_name = request.form.get('patient_name', '')
        
        print(f"Processing patient recording for: {patient_name} by Dr. {doctor_name}")
        
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400
        
        if not doctor_name or not patient_name:
            return jsonify({"error": "Doctor name and patient name are required"}), 400
        
        # Transcribe audio using Whisper
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
                temp_path = temp.name
                audio_file.save(temp_path)
                print(f"Audio file saved to: {temp_path}")
                
                result = model.transcribe(temp_path)
                transcribed_text = result['text']
                print(f"Transcription completed. Length: {len(transcribed_text)} characters")
                
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return jsonify({"error": f"Transcription failed: {str(e)}"}), 500
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Generate medical summary using OpenAI
        summary_prompt = f"""
        As a medical AI assistant, please analyze the following patient consultation transcript and provide a professional medical summary.
        
        Patient: {patient_name}
        Doctor: {doctor_name}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Transcript:
        {transcribed_text}
        
        Please provide:
        1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions
        2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned
        
        Format your response exactly as:
        SUMMARY:
        [Provide a clear, professional summary of the medical consultation]
        
        CONCLUSION:
        [Provide conclusions, recommendations, and any follow-up actions mentioned]
        """
        
        try:
            # Get AI response
            print("Generating medical summary...")
            ai_response = llm.invoke(summary_prompt)
            if hasattr(ai_response, 'content'):
                ai_content = ai_response.content.strip()
            else:
                ai_content = str(ai_response).strip()
            
            print(f"AI summary generated. Length: {len(ai_content)} characters")
            
            # Parse summary and conclusion from AI response
            summary_parts = ai_content.split("CONCLUSION:")
            if len(summary_parts) == 2:
                summary = summary_parts[0].replace("SUMMARY:", "").strip()
                conclusion = summary_parts[1].strip()
            else:
                # Fallback parsing method
                lines = ai_content.split('\n')
                summary_lines = []
                conclusion_lines = []
                in_conclusion = False
                
                for line in lines:
                    if 'CONCLUSION' in line.upper():
                        in_conclusion = True
                        continue
                    elif 'SUMMARY' in line.upper():
                        in_conclusion = False
                        continue
                    
                    if in_conclusion:
                        conclusion_lines.append(line)
                    else:
                        summary_lines.append(line)
                
                summary = '\n'.join(summary_lines).strip()
                conclusion = '\n'.join(conclusion_lines).strip()
                
                # Final fallback
                if not summary and not conclusion:
                    summary = ai_content[:len(ai_content)//2]
                    conclusion = ai_content[len(ai_content)//2:]
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return jsonify({"error": f"Summary generation failed: {str(e)}"}), 500
        
        response_data = {
            "success": True,
            "transcribed_text": transcribed_text,
            "summary": summary,
            "conclusion": conclusion,
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Successfully processed patient recording for {patient_name}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Unexpected error in transcribe_patient_notes: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """
    Generate medical summary and conclusion from transcription text.
    Expects: transcription text, doctor_name, patient_name
    Returns: summary and conclusion
    """
    try:
        data = request.get_json()
        transcription = data.get('transcription', '').strip()
        doctor_name = data.get('doctor_name', '').strip()
        patient_name = data.get('patient_name', '').strip()
        
        if not transcription:
            return jsonify({"success": False, "error": "No transcription text provided"}), 400
        
        print(f"Generating summary for patient: {patient_name} by Dr. {doctor_name}")
        
        # Generate medical summary using OpenAI
        summary_prompt = f"""
        As a medical AI assistant, please analyze the following patient consultation transcript and provide a professional medical summary.
        
        Patient: {patient_name if patient_name else 'Not specified'}
        Doctor: {doctor_name if doctor_name else 'Not specified'}
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Transcript:
        {transcription}
        
        Please provide:
        1. A concise clinical summary highlighting key medical information, symptoms, findings, and discussions
        2. Professional conclusions with recommendations, follow-up actions, or treatment plans mentioned
        
        If the transcript does not contain relevant medical information, please provide appropriate default responses indicating the lack of medical content.
        
        Format your response exactly as:
        SUMMARY:
        [Provide a clear, professional summary of the medical consultation]
        
        CONCLUSION:
        [Provide conclusions, recommendations, and any follow-up actions mentioned]
        """
        
        try:
            # Get AI response
            print("Generating medical summary from transcription...")
            ai_response = llm.invoke(summary_prompt)
            if hasattr(ai_response, 'content'):
                ai_content = ai_response.content.strip()
            else:
                ai_content = str(ai_response).strip()
            
            print(f"AI summary generated. Length: {len(ai_content)} characters")
            
            # Parse summary and conclusion from AI response
            summary_parts = ai_content.split("CONCLUSION:")
            if len(summary_parts) == 2:
                summary = summary_parts[0].replace("SUMMARY:", "").strip()
                conclusion = summary_parts[1].strip()
            else:
                # Fallback parsing method
                lines = ai_content.split('\n')
                summary_lines = []
                conclusion_lines = []
                in_conclusion = False
                
                for line in lines:
                    if 'CONCLUSION' in line.upper():
                        in_conclusion = True
                        continue
                    elif 'SUMMARY' in line.upper():
                        in_conclusion = False
                        continue
                    
                    if in_conclusion:
                        conclusion_lines.append(line)
                    else:
                        summary_lines.append(line)
                
                summary = '\n'.join(summary_lines).strip()
                conclusion = '\n'.join(conclusion_lines).strip()
                
                # Final fallback with appropriate default responses
                if not summary and not conclusion:
                    summary = "The consultation transcript provided does not contain any relevant medical information, symptoms, findings, or discussions related to a patient's health."
                    conclusion = "As there is no pertinent information available in the transcript, no medical conclusions, recommendations, or follow-up actions can be provided. It is recommended to ensure accurate and detailed documentation of patient consultations for proper medical assessment and care."
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return jsonify({"success": False, "error": f"Summary generation failed: {str(e)}"}), 500
        
        return jsonify({
            "success": True,
            "summary": summary,
            "conclusion": conclusion
        })
        
    except Exception as e:
        print(f"Unexpected error in generate_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"An unexpected error occurred: {str(e)}"}), 500

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



def clean_response_text(text):
    """Remove emojis and clean up response text while preserving line breaks."""
    import re
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub('', text)
    
    # Clean up extra spaces but preserve line breaks
    # First, handle multiple spaces within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Then clean up excessive line breaks (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove trailing spaces at the end of lines
    text = re.sub(r' +\n', '\n', text)
    
    return text.strip()

@app.route("/data", methods=["POST"])
def handle_query():
    user_input = request.json.get("data", "")
    if not user_input:
        return jsonify({"response": False, "message": "Please provide a valid input."})

    try:
        # 🧠 INTELLIGENT ROUTING: Analyze query to determine relevant disciplines
        routing_result = medical_router.analyze_query(user_input)
        relevant_disciplines = routing_result["disciplines"]
        confidence_scores = routing_result["confidence_scores"]
        
        print(f"🧠 Query: '{user_input}'")
        print(f"🎯 Routed to disciplines: {relevant_disciplines}")
        print(f"📊 Confidence scores: {confidence_scores}")
        
        # Collect responses from multiple sources
        all_responses = []
        all_citations = []
        
        # 1. Query Organization KB (discipline-specific) - Skip session-based disciplines
        for discipline_id in relevant_disciplines:
            # Skip session-based disciplines for Organization KB
            discipline_config = next((d for d in disciplines_config.get("disciplines", []) if d["id"] == discipline_id), None)
            if discipline_config and discipline_config.get("is_session_based", False):
                continue
                
            try:
                vector_db_path = get_discipline_vector_db_path(discipline_id)
                if vector_db_path and os.path.exists(vector_db_path):
                    print(f"🏥 Querying Organization KB: {discipline_id}")
                    
                    vector_store = Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
                    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                    
                    org_response = qa_chain.invoke(user_input)
                    search_results = retriever.invoke(user_input)
                    
                    if org_response['result'].strip():
                        all_responses.append({
                            "source": f"Organization KB - {discipline_id.replace('_', ' ').title()}",
                            "content": org_response['result'],
                            "confidence": confidence_scores.get(discipline_id, 70)
                        })
                        
                        org_citations = enhance_with_citations(search_results)
                        if org_citations != "No citations available":
                            clean_citations = clean_response_text(org_citations)
                            # Split individual citations and format each one
                            citation_lines = [line.strip() for line in clean_citations.split('\n') if line.strip()]
                            for citation_line in citation_lines:
                                all_citations.append(f"**{discipline_id.replace('_', ' ').title()}: {citation_line}**")
                            
            except Exception as e:
                print(f"Error querying {discipline_id}: {e}")
        
        # 2. Query Adhoc KB (user-uploaded content) - Higher priority if doctors_files is selected
        doctors_files_selected = "doctors_files" in relevant_disciplines
        try:
            latest_vector_db = get_latest_vector_db()
            if latest_vector_db and os.path.exists(latest_vector_db):
                if doctors_files_selected:
                    print("📄 Querying Doctor's Files (prioritized)")
                else:
                    print("📄 Querying Adhoc KB (user uploads)")
                
                vector_store = Chroma(persist_directory=latest_vector_db, embedding_function=embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                
                adhoc_response = qa_chain.invoke(user_input)
                search_results = retriever.invoke(user_input)
                
                if adhoc_response['result'].strip():
                    source_name = "Doctor's Files" if doctors_files_selected else "User Uploaded Documents"
                    confidence = confidence_scores.get("doctors_files", 85) if doctors_files_selected else 85
                    
                    all_responses.append({
                        "source": source_name,
                        "content": adhoc_response['result'],
                        "confidence": confidence
                    })
                    
                    adhoc_citations = enhance_with_citations(search_results)
                    if adhoc_citations != "No citations available":
                        clean_citations = clean_response_text(adhoc_citations)
                        # Split individual citations and format each one
                        citation_lines = [line.strip() for line in clean_citations.split('\n') if line.strip()]
                        citation_prefix = "Doctor's Files" if doctors_files_selected else "User Documents"
                        for citation_line in citation_lines:
                            all_citations.append(f"**{citation_prefix}: {citation_line}**")
                        
        except Exception as e:
            print(f"Error querying adhoc KB: {e}")
        
        # 3. Synthesize final response
        if all_responses:
            # Sort responses by confidence
            all_responses.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Create comprehensive response - just the content without headers
            final_response = ""
            
            for i, resp in enumerate(all_responses[:2], 1):  # Limit to top 2 responses
                if i > 1:
                    final_response += "\n\n"  # Add spacing between multiple responses
                final_response += clean_response_text(resp['content'])
            
            # Add citations in bold
            if all_citations:
                final_response += "\n\n**Citations:**\n"
                for citation in all_citations:
                    final_response += f"{citation}\n"
            
            # Add routing information on a new line
            routing_info = f"\n**Query Routing:** Analyzed and routed to {', '.join([d.replace('_', ' ').title() for d in relevant_disciplines])}"
            final_response += routing_info
            
            # Don't apply clean_response_text to the final response as it breaks formatting
            
            # Return response with routing details
            return jsonify({
                "response": True, 
                "message": final_response,
                "routing_details": {
                    "disciplines": [d.replace('_', ' ').title() for d in relevant_disciplines],
                    "confidence_scores": confidence_scores,
                    "sources": f"{len(all_responses)} sources found",
                    "method": routing_result.get("routing_method", "hybrid")
                }
            })
        else:
            # Fallback to general response
            fallback_response = f"""
            I understand you're asking about: "{user_input}"
            
            However, I couldn't find specific information in the available medical knowledge bases for the disciplines I identified: {', '.join([d.replace('_', ' ').title() for d in relevant_disciplines])}.
            
            This could be because:
            1. The Organization KB doesn't have information on this specific topic yet
            2. No user documents have been uploaded that relate to this query
            3. The query might need to be more specific
            
            **Query was routed to:** {', '.join([d.replace('_', ' ').title() for d in relevant_disciplines])}
            
            Consider uploading relevant medical documents or rephrasing your question for better results.
            """
            
            return jsonify({
                "response": True, 
                "message": clean_response_text(fallback_response),
                "routing_details": {
                    "disciplines": [d.replace('_', ' ').title() for d in relevant_disciplines],
                    "confidence_scores": confidence_scores,
                    "sources": "No relevant sources found",
                    "method": routing_result.get("routing_method", "hybrid")
                }
            })

    except Exception as e:
        print(f"Error in handle_query: {e}")
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

@app.route('/api/disciplines', methods=['GET'])
def get_disciplines():
    """Return available disciplines for UI dropdown."""
    try:
        disciplines = get_available_disciplines()
        return jsonify({
            "success": True,
            "disciplines": disciplines,
            "selection_rules": disciplines_config.get("selection_rules", {})
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/validate_disciplines', methods=['POST'])
def validate_disciplines():
    """Validate selected disciplines."""
    try:
        selected = request.json.get("selected_disciplines", [])
        is_valid, message = validate_discipline_selection(selected)
        return jsonify({
            "success": True,
            "is_valid": is_valid,
            "message": message
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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

@app.route('/upload_organization_kb', methods=['POST'])
def upload_organization_kb():
    """Upload documents to Organization KB for specific disciplines."""
    try:
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        discipline_id = request.form.get('discipline_id')
        if not discipline_id:
            return jsonify({"error": "No discipline specified"}), 400
        
        # Validate discipline
        discipline_path = None
        for discipline in disciplines_config.get("disciplines", []):
            if discipline["id"] == discipline_id:
                discipline_path = discipline.get("kb_path")
                break
        
        if not discipline_path:
            return jsonify({"error": f"Invalid discipline: {discipline_id}"}), 400
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No valid files uploaded"}), 400
        
        # Create discipline directory
        discipline_dir = os.path.join(".", discipline_path)
        os.makedirs(discipline_dir, exist_ok=True)
        
        saved_files = []
        documents = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=128)
        
        for file in files:
            if file.filename.endswith('.pdf'):
                file_path = os.path.join(discipline_dir, file.filename)
                file.save(file_path)
                saved_files.append(file.filename)
                
                # Extract text and create documents
                text_content = extract_text_from_pdf(file_path)
                if text_content:
                    if isinstance(text_content, str):
                        text_content = [text_content]
                    
                    for page_num, text in enumerate(text_content, start=1):
                        chunks = text_splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(Document(
                                page_content=chunk,
                                metadata={
                                    "source": file.filename,
                                    "type": "organization_pdf",
                                    "discipline": discipline_id,
                                    "page": page_num
                                }
                            ))
        
        # Create/update vector database for this discipline
        if documents:
            create_organization_vector_db(discipline_id, documents)
            
        return jsonify({
            "message": f"Successfully uploaded {len(saved_files)} files to {discipline_id}",
            "files": saved_files,
            "documents_created": len(documents)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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


@app.route("/generate_patient_pdf", methods=["POST"])
def generate_patient_pdf():
    """Generate PDF for patient notes with patient problem capture and Azure upload"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.colors import black
        import io
        import os
        
        data = request.json
        
        # Extract data
        doctor_name = data.get('doctorName', '')
        patient_name = data.get('patientName', '')
        patient_id = data.get('patientId', '')
        date_time = data.get('dateTime', '')
        transcription = data.get('transcription', '')
        summary = data.get('summary', '')
        conclusion = data.get('conclusion', '')
        
        # NEW: Extract patient problem (required for first PDF only)
        patient_problem = data.get('patientProblem', '')
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, 
                               topMargin=inch, bottomMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica',
            textColor='blue'
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        session_style = ParagraphStyle(
            'SessionStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            spaceAfter=5
        )
        
        # Build PDF content
        story = []
        
        # Add logo at the top right with patient-specific title
        logo_path = os.path.join(os.path.dirname(__file__), 'ClientLogo101.png')
        patient_display_name = patient_name if patient_name else "Patient Name"
        main_title = f"Patient – {patient_display_name} – Recording Notes"
        
        if os.path.exists(logo_path):
            try:
                # Create a table to position title on left and logo on right
                logo_img = Image(logo_path, width=1.5*inch, height=0.9*inch)
                title_paragraph = Paragraph(main_title, title_style)
                
                # Create table with title and logo
                header_data = [[title_paragraph, logo_img]]
                header_table = Table(header_data, colWidths=[4.5*inch, 2*inch])
                header_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 15))
            except Exception as e:
                print(f"Warning: Could not add logo to PDF: {e}")
                # Fallback to just title
                story.append(Paragraph(main_title, title_style))
                story.append(Spacer(1, 15))
        else:
            # Fallback to just title if logo doesn't exist
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
        
        # Add "Patient Recording Notes" subtitle
        story.append(Paragraph("Patient Recording Notes", subtitle_style))
        story.append(Spacer(1, 10))
        
        # Patient and Doctor Information
        patient_id_display = patient_id if patient_id else "N/A"
        patient_info = f"Patient Name: {patient_display_name} – Patient ID: {patient_id_display}"
        doctor_info = f"Doctor's Name: {doctor_name if doctor_name else 'Unknown Doctor'}"
        
        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 5))
        story.append(Paragraph(doctor_info, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Session Information
        story.append(Paragraph("Session Information", header_style))
        story.append(Paragraph("Transcription Engine: Whisper; Summary Engine: OpenAI", session_style))
        
        # Format date
        display_date = date_time if date_time else datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        story.append(Paragraph(f"Date: {display_date}", session_style))
        story.append(Spacer(1, 15))
        
        # NEW: Patient Problem (if provided)
        if patient_problem:
            story.append(Paragraph("Patient Problem", header_style))
            story.append(Paragraph(patient_problem, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Original Transcription Text
        story.append(Paragraph("Original Transcription Text", header_style))
        transcription_text = transcription if transcription else "No transcription available"
        story.append(Paragraph(transcription_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Medical Summary
        story.append(Paragraph("Medical Summary", header_style))
        summary_text = summary if summary else "No summary available"
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Conclusion & Recommendations
        story.append(Paragraph("Conclusion & Recommendations", header_style))
        conclusion_text = conclusion if conclusion else "No conclusion available"
        story.append(Paragraph(conclusion_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
        
        # NEW: Upload to Azure if available
        azure_url = None
        if AZURE_AVAILABLE:
            try:
                # Prepare metadata
                metadata = {
                    'doctor_name': doctor_name,
                    'patient_name': patient_name,
                    'patient_id': patient_id,
                    'date_time': date_time,
                    'patient_problem': patient_problem,
                    'pdf_type': 'patient_notes',
                    'generated_at': datetime.now().isoformat()
                }
                
                # Upload to patient summary container
                azure_url = upload_pdf_to_azure(pdf_content, filename, "patient_summary", metadata)
                
            except Exception as e:
                print(f"Azure upload failed: {e}")
        
        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Add Azure URL to response headers if available
        if azure_url:
            response.headers['X-Azure-URL'] = azure_url
        
        return response
        
    except ImportError:
        # Fallback if reportlab is not installed
        return jsonify({"error": "PDF generation not available. Please install reportlab."}), 500
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route("/generate_chat_pdf", methods=["POST"])
def generate_chat_pdf():
    """Generate PDF for chat conversation with specified format"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.colors import black
        import io
        import os
        
        data = request.json
        
        # Extract data
        doctor_name = data.get('doctorName', 'Dr. Name')
        patient_name = data.get('patientName', '')  # NEW: Include patient name
        messages = data.get('messages', [])
        json_data = data.get('jsonData', '')
        
        # Generate timestamp in YYYY MM DD HH MM format
        now = datetime.now()
        formatted_date = now.strftime("%Y %m %d %H %M")
        
        # Create PDF filename - include patient name if provided
        timestamp = now.strftime("%Y%m%d%H%M")
        if patient_name:
            filename = f"{doctor_name.upper().replace(' ', '')}-{patient_name.upper().replace(' ', '')}-{timestamp}.pdf"
        else:
            filename = f"{doctor_name.upper().replace(' ', '')}-{timestamp}.pdf"
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, 
                               topMargin=inch, bottomMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        header_style = ParagraphStyle(
            'HeaderStyle',
            parent=styles['Normal'],
            fontSize=12,
            fontName='Helvetica-Bold',
            spaceAfter=12,
            alignment=TA_LEFT
        )
        
        section_header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            spaceAfter=6,
            spaceBefore=12,
            alignment=TA_CENTER
        )
        
        content_style = ParagraphStyle(
            'ContentStyle',
            parent=styles['Normal'],
            fontSize=10,
            fontName='Helvetica',
            spaceAfter=8,
            alignment=TA_LEFT
        )
        
        # Build PDF content
        story = []
        
        # Add logo at the top
        logo_path = os.path.join(os.path.dirname(__file__), 'ClientLogo101.png')
        if os.path.exists(logo_path):
            try:
                logo = Image(logo_path, width=2*inch, height=1.2*inch)
                logo.hAlign = 'CENTER'
                story.append(logo)
                story.append(Spacer(1, 20))
            except Exception as e:
                print(f"Warning: Could not add logo to PDF: {e}")
        
        # Header section
        story.append(Paragraph(f"Doctor Name: {doctor_name}", header_style))
        story.append(Paragraph(f"Date: {formatted_date}", header_style))
        story.append(Spacer(1, 20))
        
        # JSON Data section (if provided)
        if json_data:
            story.append(Paragraph("JSON Data:", section_header_style))
            story.append(Paragraph(json_data, content_style))
            story.append(Spacer(1, 20))
        
        # Chat conversation
        for i, message in enumerate(messages):
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                story.append(Paragraph("****** Doctor Input *****", section_header_style))
                story.append(Paragraph(content, content_style))
                story.append(Spacer(1, 12))
            elif role == 'ai':
                story.append(Paragraph("****** System Output *****", section_header_style))
                story.append(Paragraph(content, content_style))
                story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # NEW: Upload to Azure if available
        azure_url = None
        if AZURE_AVAILABLE:
            try:
                # Prepare metadata
                metadata = {
                    'doctor_name': doctor_name,
                    'pdf_type': 'research_chat',
                    'generated_at': datetime.now().isoformat(),
                    'message_count': len(messages),
                    'has_json_data': bool(json_data)
                }
                
                # Upload to research container
                azure_url = upload_pdf_to_azure(pdf_content, filename, "research", metadata)
                
            except Exception as e:
                print(f"Azure upload failed: {e}")
        
        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Add Azure URL to response headers if available
        if azure_url:
            response.headers['X-Azure-URL'] = azure_url
        
        return response
        
    except ImportError:
        # Fallback if reportlab is not installed
        return jsonify({"error": "PDF generation not available. Please install reportlab."}), 500
    except Exception as e:
        print(f"Error generating chat PDF: {e}")
        return jsonify({"error": f"Failed to generate PDF: {str(e)}"}), 500


@app.route("/generate_conversation_pdf", methods=["POST"])
def generate_conversation_pdf():
    """Generate PDF for conversation segments with voice diarization results"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.colors import black, green, blue
        import io
        import os
        
        data = request.json
        
        # Extract data
        doctor_name = data.get('doctorName', 'Doctor')
        patient_name = data.get('patientName', 'Patient')
        date_time = data.get('dateTime', '')
        segments = data.get('segments', [])
        full_transcript = data.get('fullTranscript', '')
        processing_info = data.get('processingInfo', {})
        is_duplicate = data.get('isDuplicate', False)
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch, leftMargin=inch, 
                               topMargin=inch, bottomMargin=inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=TA_LEFT,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            alignment=TA_CENTER,
            fontName='Helvetica',
            textColor=blue
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        segment_style = ParagraphStyle(
            'SegmentStyle',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            fontName='Helvetica'
        )
        
        doctor_segment_style = ParagraphStyle(
            'DoctorSegmentStyle',
            parent=segment_style,
            leftIndent=20,
            textColor=green
        )
        
        patient_segment_style = ParagraphStyle(
            'PatientSegmentStyle',
            parent=segment_style,
            leftIndent=20,
            textColor=blue
        )
        
        # Build PDF content
        story = []
        
        # Add logo and title
        logo_path = os.path.join(os.path.dirname(__file__), 'ClientLogo101.png')
        duplicate_text = " (Duplicate)" if is_duplicate else ""
        main_title = f"Doctor-Patient Conversation Analysis{duplicate_text}"
        
        if os.path.exists(logo_path):
            try:
                logo_img = Image(logo_path, width=1.5*inch, height=0.9*inch)
                title_paragraph = Paragraph(main_title, title_style)
                
                header_data = [[title_paragraph, logo_img]]
                header_table = Table(header_data, colWidths=[4.5*inch, 2*inch])
                header_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                story.append(header_table)
                story.append(Spacer(1, 15))
            except Exception as e:
                print(f"Warning: Could not add logo to PDF: {e}")
                story.append(Paragraph(main_title, title_style))
                story.append(Spacer(1, 15))
        else:
            story.append(Paragraph(main_title, title_style))
            story.append(Spacer(1, 15))
        
        # Add subtitle
        story.append(Paragraph("Voice Diarization & Transcription Results", subtitle_style))
        story.append(Spacer(1, 10))
        
        # Participant Information
        doctor_info = f"Doctor: {doctor_name}"
        patient_info = f"Patient: {patient_name}"
        
        story.append(Paragraph(doctor_info, styles['Normal']))
        story.append(Spacer(1, 5))
        story.append(Paragraph(patient_info, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Processing Information
        story.append(Paragraph("Processing Information", header_style))
        engine_info = processing_info.get('engine', 'OpenAI Voice Diarization + Whisper')
        total_segments = processing_info.get('totalSegments', len(segments))
        
        story.append(Paragraph(f"Engine: {engine_info}", styles['Normal']))
        story.append(Paragraph(f"Total Segments: {total_segments}", styles['Normal']))
        
        # Format date
        display_date = date_time if date_time else datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        story.append(Paragraph(f"Processing Date: {display_date}", styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Conversation Segments
        if segments and len(segments) > 0:
            story.append(Paragraph("Conversation Segments", header_style))
            
            for i, segment in enumerate(segments):
                role = segment.get('role', 'Unknown')
                text = segment.get('text', '')
                start_time = segment.get('start', '')
                end_time = segment.get('end', '')
                confidence = segment.get('confidence', 0)
                
                # Create segment header
                timing_info = f"[{start_time} - {end_time}]" if start_time and end_time else ""
                confidence_info = f"(Confidence: {int(confidence * 100)}%)" if confidence > 0 else ""
                header_text = f"{role} {timing_info} {confidence_info}"
                
                # Choose style based on role
                if role.lower() == 'doctor':
                    story.append(Paragraph(f"<b>{header_text}</b>", doctor_segment_style))
                    story.append(Paragraph(text, doctor_segment_style))
                else:  # Patient
                    story.append(Paragraph(f"<b>{header_text}</b>", patient_segment_style))
                    story.append(Paragraph(text, patient_segment_style))
                
                story.append(Spacer(1, 8))
            
            story.append(Spacer(1, 15))
        
        # Full Transcript
        story.append(Paragraph("Complete Transcript", header_style))
        transcript_text = full_transcript if full_transcript else "No transcript available"
        story.append(Paragraph(transcript_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        pdf_content = buffer.getvalue()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        duplicate_suffix = "-DUPLICATE" if is_duplicate else ""
        filename = f"CONVERSATION-{doctor_name.upper().replace(' ', '')}{duplicate_suffix}-{timestamp}.pdf"
        
        # Upload to Azure if available
        azure_url = None
        if AZURE_AVAILABLE:
            try:
                metadata = {
                    'doctor_name': doctor_name,
                    'patient_name': patient_name,
                    'date_time': date_time,
                    'pdf_type': 'conversation_segments',
                    'is_duplicate': is_duplicate,
                    'total_segments': len(segments),
                    'generated_at': datetime.now().isoformat()
                }
                
                azure_url = upload_pdf_to_azure(pdf_content, filename, "conversation", metadata)
                
            except Exception as e:
                print(f"Azure upload failed: {e}")
        
        # Return PDF as response
        from flask import make_response
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        if azure_url:
            response.headers['X-Azure-URL'] = azure_url
        
        return response
        
    except ImportError:
        return jsonify({"error": "PDF generation not available. Please install reportlab."}), 500
    except Exception as e:
        print(f"Error generating conversation PDF: {e}")
        return jsonify({"error": f"Failed to generate conversation PDF: {str(e)}"}), 500


@app.route('/search_doctors', methods=['GET'])
def search_doctors():
    """
    Search for doctors by first_name and last_name from pces_users table.
    Returns matching doctors based on partial input.
    """
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify([])
        
        # Connect to database
        with psycopg.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                # Search by first_name, last_name, or combined name
                search_query = """
                SELECT DISTINCT first_name, last_name 
                FROM pces_users 
                WHERE LOWER(first_name) LIKE %s 
                   OR LOWER(last_name) LIKE %s 
                   OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                ORDER BY first_name, last_name
                LIMIT 10
                """
                
                search_pattern = f"%{query}%"
                cursor.execute(search_query, (search_pattern, search_pattern, search_pattern))
                results = cursor.fetchall()
                
                # Format results
                doctors = []
                for row in results:
                    if row[0] and row[1]:  # Ensure both names exist
                        full_name = f"{row[0]} {row[1]}"
                        doctors.append({
                            "first_name": row[0],
                            "last_name": row[1],
                            "full_name": full_name
                        })
                
                return jsonify(doctors)
                
    except Exception as e:
        print(f"Error searching doctors: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/search_patients', methods=['GET'])
def search_patients():
    """
    Search for patients by first_name and last_name from patient table.
    Returns matching patients based on partial input.
    """
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify([])
        
        # Connect to database
        with psycopg.connect(**db_config) as conn:
            with conn.cursor() as cursor:
                # Search by first_name, last_name, or combined name
                search_query = """
                SELECT DISTINCT patient_id, first_name, last_name 
                FROM patient 
                WHERE LOWER(first_name) LIKE %s 
                   OR LOWER(last_name) LIKE %s 
                   OR LOWER(CONCAT(first_name, ' ', last_name)) LIKE %s
                ORDER BY first_name, last_name
                LIMIT 10
                """
                
                search_pattern = f"%{query}%"
                cursor.execute(search_query, (search_pattern, search_pattern, search_pattern))
                results = cursor.fetchall()
                
                # Format results
                patients = []
                for row in results:
                    if row[1] and row[2]:  # Ensure both names exist
                        full_name = f"{row[1]} {row[2]}"
                        patients.append({
                            "patient_id": row[0],
                            "first_name": row[1],
                            "last_name": row[2],
                            "full_name": full_name
                        })
                
                return jsonify(patients)
                
    except Exception as e:
        print(f"Error searching patients: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe_doctor_patient_conversation", methods=["POST"])
def transcribe_doctor_patient_conversation():
    """Handle doctor-patient conversation recording with voice diarization"""
    if not DIARIZATION_AVAILABLE:
        return jsonify({"error": "Voice diarization not available. Please install required dependencies."}), 500
    
    try:
        audio_file = request.files['audio']
        doctor_name = request.form.get('doctor_name', 'Unknown Doctor')
        patient_name = request.form.get('patient_name', 'Unknown Patient')
        
        # Save uploaded audio to temporary file with proper format
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        # Convert audio to proper format for pyannote
        try:
            import librosa
            import soundfile as sf
            
            # Load audio and convert to proper format
            audio_data, sample_rate = librosa.load(temp_audio_path, sr=16000, mono=True)
            
            # Create a new properly formatted WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_audio:
                sf.write(converted_audio.name, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                converted_audio_path = converted_audio.name
            
            # Clean up original temp file
            os.unlink(temp_audio_path)
            temp_audio_path = converted_audio_path
            
        except ImportError:
            # If librosa/soundfile not available, try with pydub
            try:
                from pydub import AudioSegment
                import io
                
                # Load the audio file
                audio_segment = AudioSegment.from_file(temp_audio_path)
                
                # Convert to proper format: 16kHz, mono, 16-bit PCM
                audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                
                # Export as proper WAV
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_audio:
                    audio_segment.export(converted_audio.name, format="wav")
                    converted_audio_path = converted_audio.name
                
                # Clean up original temp file
                os.unlink(temp_audio_path)
                temp_audio_path = converted_audio_path
                
            except ImportError:
                # If no audio processing libraries available, log warning and continue
                print("Warning: No audio processing libraries available. Using original audio format.")
        
        try:
            # Process conversation with voice diarization (OpenAI-only mode)
            diarization_processor = get_diarization_processor()
            
            # Run async function in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            print("🔄 Using OpenAI-only processing (bypassing pyannote)")
            result = loop.run_until_complete(
                diarization_processor.process_doctor_patient_conversation_openai_only(temp_audio_path)
            )
            
            if result.get("error"):
                return jsonify({"success": False, "error": result["error"]}), 500
            
            # Prepare conversation data
            conversation_data = {
                "doctor_name": doctor_name,
                "patient_name": patient_name,
                "session_date": datetime.now().isoformat(),
                "duration": result.get("total_duration", "Unknown"),
                "total_segments": result.get("total_segments", 0),
                "doctor_segments": result.get("doctor_segments", 0),
                "patient_segments": result.get("patient_segments", 0),
                "speakers_detected": result.get("speakers_detected", 0),
                "transcript": result.get("transcript", []),
                "raw_transcript": result.get("raw_transcript", ""),
                "role_mapping": result.get("role_mapping", {})
            }
            
            return jsonify({
                "success": True,
                "conversation_data": conversation_data,
                "message": "Doctor-patient conversation processed successfully"
            })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error processing doctor-patient conversation: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Update existing PDF generation functions to include Azure uploads
def upload_pdf_to_azure(pdf_content, filename, pdf_type, metadata=None):
    """Helper function to upload PDFs to Azure"""
    if not AZURE_AVAILABLE:
        print("Azure storage not available")
        return None
    
    try:
        storage_manager = get_storage_manager()
        
        if pdf_type == "research":
            return storage_manager.upload_research_pdf(pdf_content, filename, metadata.get('patient_problem'), metadata)
        elif pdf_type == "patient_summary":
            # Extract patient data from metadata for the patient_data parameter
            patient_data = {
                'patient_name': metadata.get('patient_name'),
                'patient_id': metadata.get('patient_id'),
                'doctor_name': metadata.get('doctor_name'),
                'session_date': metadata.get('date_time')
            }
            return storage_manager.upload_patient_summary_pdf(pdf_content, filename, patient_data, metadata)
        elif pdf_type == "conversation":
            # Extract conversation data from metadata for the conversation_data parameter
            conversation_data = {
                'doctor_name': metadata.get('doctor_name'),
                'patient_name': metadata.get('patient_name'),
                'duration': metadata.get('duration', 'Unknown'),
                'session_date': metadata.get('date_time')
            }
            return storage_manager.upload_conversation_pdf(pdf_content, filename, conversation_data, metadata)
        
    except Exception as e:
        print(f"Azure upload failed: {e}")
        return None


@app.route('/check_azure_files', methods=['GET'])
def check_azure_files():
    """Check what files have been uploaded to Azure"""
    if not AZURE_AVAILABLE:
        return jsonify({"error": "Azure storage not available"}), 500
    
    try:
        storage_manager = get_storage_manager()
        
        # Get file type filter from query params
        file_type = request.args.get('type', 'all')  # 'research', 'patient_summary', 'conversation', or 'all'
        
        files_info = {}
        
        if file_type in ['research', 'all']:
            research_files = storage_manager.list_files_in_container("contoso", "pces/documents/research/")
            files_info['research_files'] = research_files
        
        if file_type in ['patient_summary', 'all']:
            patient_files = storage_manager.list_files_in_container("contoso", "pces/documents/doc-patient-summary/")
            files_info['patient_summary_files'] = patient_files
        
        if file_type in ['conversation', 'all']:
            conversation_files = storage_manager.list_files_in_container("contoso", "pces/documents/conversation/")
            files_info['conversation_files'] = conversation_files
        
        # Count totals
        total_files = sum(len(files) for files in files_info.values())
        
        return jsonify({
            "success": True,
            "total_files": total_files,
            "files": files_info,
            "message": f"Found {total_files} files in Azure storage"
        })
        
    except Exception as e:
        print(f"Error checking Azure files: {e}")
        return jsonify({"error": f"Failed to check Azure files: {str(e)}"}), 500


@app.route('/check_azure_file/<path:filename>', methods=['GET'])
def check_azure_file(filename):
    """Check if a specific file exists in Azure"""
    if not AZURE_AVAILABLE:
        return jsonify({"error": "Azure storage not available"}), 500
    
    try:
        storage_manager = get_storage_manager()
        container_name = request.args.get('container', 'contoso')
        file_path = request.args.get('path', f'pces/documents/research/{filename}')
        
        # Check if file exists
        exists = storage_manager.check_file_exists(container_name, file_path)
        
        if exists:
            # Get file metadata
            file_info = storage_manager.get_file_metadata(container_name, file_path)
            return jsonify({
                "success": True,
                "exists": True,
                "file_info": file_info
            })
        else:
            return jsonify({
                "success": True,
                "exists": False,
                "message": f"File {filename} not found in Azure"
            })
        
    except Exception as e:
        print(f"Error checking Azure file: {e}")
        return jsonify({"error": f"Failed to check Azure file: {str(e)}"}), 500


@app.route('/azure_storage_info', methods=['GET'])
def azure_storage_info():
    """Get Azure storage configuration and status"""
    try:
        info = {
            "azure_available": AZURE_AVAILABLE,
            "connection_configured": bool(os.getenv('AZURE_STORAGE_CONNECTION_STRING')),
            "containers": {
                "contoso": {
                    "research_path": "pces/documents/research/",
                    "patient_summary_path": "pces/documents/doc-patient-summary/",
                    "conversations_path": "pces/documents/conversation/"
                }
            }
        }
        
        if AZURE_AVAILABLE and info["connection_configured"]:
            try:
                storage_manager = get_storage_manager()
                # Test connection by listing containers
                containers = []
                for container in storage_manager.blob_service_client.list_containers():
                    containers.append(container.name)
                info["available_containers"] = containers
                info["connection_status"] = "Connected"
            except Exception as e:
                info["connection_status"] = f"Connection failed: {str(e)}"
        else:
            info["connection_status"] = "Not configured"
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({"error": f"Failed to get Azure info: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3000, use_reloader=False)
