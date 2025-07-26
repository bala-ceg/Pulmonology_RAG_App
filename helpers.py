import os
import glob
import time
import json
import re
from typing import List
from langchain.schema import Document
from .config import BASE_STORAGE_PATH, VECTOR_DBS_FOLDER


def get_timestamp() -> str:
    """Generate timestamp in MMDDYYYYHHMM format."""
    return time.strftime("%m%d%Y%H%M")


def get_latest_vector_db() -> str | None:
    """Return the most recently modified vector DB path."""
    vector_dbs = glob.glob(os.path.join(VECTOR_DBS_FOLDER, "*"))
    if not vector_dbs:
        print("No existing vector DB found. A new one will be created.")
        return None
    latest_db = max(vector_dbs, key=os.path.getmtime)
    print(f"Using latest vector DB: {latest_db}")
    return latest_db


def load_metadata(file_path: str) -> List[dict]:
    """Load JSON metadata from a file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
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
            if not page_text:
                continue
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                chunked_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": file_name, "type": "pdf", "page": page_number},
                    )
                )
    return chunked_documents


def process_url_metadata(url_metadata: list, text_splitter) -> list:
    """Chunk the text content from URL metadata."""
    chunked_documents = []
    for entry in url_metadata:
        url = entry.get("url", "Unknown URL")
        text_content = entry.get("text", "").strip()
        date_info = entry.get("date", "Unknown Date")
        if not text_content:
            continue
        chunks = text_splitter.split_text(text_content)
        for chunk in chunks:
            chunked_documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": url, "type": "url", "date": date_info},
                )
            )
    return chunked_documents


def clean_extracted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?'\":;\-\s]", "", text)
    text = text.strip()
    text = re.sub(r"\.{2,}", ".", text)
    return text.encode("ascii", "ignore").decode("utf-8")


def count_files_in_folder(folder: str) -> int:
    """Return the number of files in a folder."""
    if os.path.exists(folder):
        return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
    return 0


def can_upload_more_files(pdf_folder: str, url_folder: str, new_files_count: int) -> bool:
    """Check if the upload exceeds the allowed limit."""
    pdf_count = count_files_in_folder(pdf_folder)
    url_count = count_files_in_folder(url_folder)
    return (pdf_count + url_count + new_files_count) <= 10


def enhance_with_citations(results):
    """Return formatted citation information from retrieved documents."""
    pdf_citations = set()
    url_citations = set()
    for doc in results:
        metadata = getattr(doc, "metadata", {})
        doc_type = metadata.get("type")
        if doc_type == "pdf":
            pdf_source = metadata.get("source", "Unknown PDF")
            page_info = metadata.get("page", "Unknown Page")
            pdf_citations.add(f"PDF: {pdf_source} (Page {page_info})")
        elif doc_type == "url":
            url_source = metadata.get("source", "Unknown URL")
            url_citations.add(f"URL: {url_source}")
    return "\n".join(pdf_citations.union(url_citations)) or "No citations available"
