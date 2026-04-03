"""
Documents Blueprint

Routes:
  POST /upload_pdf              — upload PDFs to the session folder
  POST /upload_url              — upload a text file of URLs to the session folder
  POST /upload_organization_kb  — upload PDFs to the Organization KB
  POST /create_vector_db        — build a Chroma vector DB from session files

Shared resources accessed via current_app.config:
  EMBEDDINGS          — OpenAIEmbeddings instance
  TEXT_SPLITTER       — RecursiveCharacterTextSplitter instance
  APIFY_CLIENT        — ApifyClient instance
  LAST_SESSION_FOLDER — current session folder name (set by disciplines_bp)
  RAG_MANAGER         — TwoStoreRAGManager (optional)
  DISCIPLINES_CONFIG  — loaded disciplines JSON
"""

from __future__ import annotations

import os
import re
import traceback

import fitz  # PyMuPDF
from flask import Blueprint, current_app, jsonify, request
from langchain_core.documents import Document

from config import Config
from utils.error_handlers import get_logger, handle_route_errors

logger = get_logger(__name__)

documents_bp = Blueprint("documents_bp", __name__)

# ---------------------------------------------------------------------------
# Path constants (mirror main.py)
# ---------------------------------------------------------------------------
BASE_STORAGE_PATH = Config.KB_PATH
VECTOR_DB_PATH = Config.VECTOR_DB_PATH


# ---------------------------------------------------------------------------
# Shared resource accessors
# ---------------------------------------------------------------------------

def _get_embeddings():
    return current_app.config.get("EMBEDDINGS")


def _get_text_splitter():
    return current_app.config.get("TEXT_SPLITTER")


def _get_apify_client():
    return current_app.config.get("APIFY_CLIENT")


def _get_rag_manager():
    return current_app.config.get("RAG_MANAGER")


def _get_session_folder() -> str | None:
    return current_app.config.get("LAST_SESSION_FOLDER")


def _get_disciplines_config() -> dict:
    return current_app.config.get("DISCIPLINES_CONFIG") or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_files_in_folder(folder: str) -> int:
    """Return the number of files in a given folder."""
    if os.path.exists(folder):
        return len(
            [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        )
    return 0


def _can_upload_more_files(session_folder: str, new_files_count: int) -> bool:
    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", session_folder)
    url_folder = os.path.join(BASE_STORAGE_PATH, "URL", session_folder)
    pdf_count = count_files_in_folder(pdf_folder)
    url_count = count_files_in_folder(url_folder)
    limit = Config.MAX_FILES_PER_SESSION
    return (pdf_count + url_count + new_files_count) <= limit


def _clean_extracted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?'\":;\-\s]", "", text)
    text = text.strip()
    text = re.sub(r"\.{2,}", ".", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text


def extract_text_from_pdf(pdf_file: str) -> list[str]:
    """Extract text content from a PDF file, one entry per page."""
    text_content: list[str] = []
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            clean_text = _clean_extracted_text(page.get_text())
            text_content.append(clean_text)
    return text_content


def extract_text_from_url(url: str) -> str:
    """Extract text from a URL using the Apify website-content-crawler actor."""
    apify_client = _get_apify_client()
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
        "removeElementsCssSelector": (
            'nav, footer, script, style, noscript, svg, img[src^=\'data:\'],\n'
            '        [role="alert"],\n'
            '        [role="banner"],\n'
            '        [role="dialog"],\n'
            '        [role="alertdialog"],\n'
            '        [role="region"][aria-label*="skip" i],\n'
            '        [aria-modal="true"]'
        ),
        "clickElementsCssSelector": '[aria-expanded="false"]',
    }

    run = apify_client.actor("apify/website-content-crawler").call(run_input=run_input)

    full_text = ""
    for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
        page_text = item.get("text", "")
        page_text = re.sub(r"[^\x00-\x7F]+", " ", page_text)
        page_text = re.sub(r"\s+", " ", page_text).strip()
        full_text += page_text + "\n"
    logger.debug("Extracted URL text (length: %d)", len(full_text))
    return full_text.strip()


def _get_discipline_vector_db_path(discipline_id: str) -> str | None:
    cfg = _get_disciplines_config()
    for discipline in cfg.get("disciplines", []):
        if discipline["id"] == discipline_id:
            return discipline.get("vector_db_path", "")
    return None


def _create_organization_vector_db(discipline_id: str, documents: list) -> object:
    from langchain_chroma import Chroma

    vector_db_path = _get_discipline_vector_db_path(discipline_id)
    if not vector_db_path:
        raise ValueError(f"Unknown discipline: {discipline_id}")

    persist_dir = os.path.join(".", vector_db_path)
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = _get_embeddings()
    return Chroma.from_documents(
        documents, embedding=embeddings, persist_directory=persist_dir
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@documents_bp.route("/upload_pdf", methods=["POST"])
@handle_route_errors
def upload_pdf():
    """Upload multiple PDF files to the session folder."""
    session_folder = _get_session_folder()
    if not session_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    files = request.files.getlist("file")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No valid files uploaded"}), 400

    if not _can_upload_more_files(session_folder, len(files)):
        return jsonify(
            {"message": f"Cannot process: Maximum of {Config.MAX_FILES_PER_SESSION} PDFs exceeded."}
        )

    pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", session_folder)
    os.makedirs(pdf_folder, exist_ok=True)

    saved_files: list[str] = []
    try:
        for file in files:
            file_path = os.path.join(pdf_folder, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)

        return jsonify({"message": "PDFs uploaded successfully", "files": saved_files})

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/upload_url", methods=["POST"])
@handle_route_errors
def upload_url():
    """Upload a text file containing URLs."""
    session_folder = _get_session_folder()
    if not session_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file uploaded"}), 400

        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", session_folder)
        os.makedirs(url_folder, exist_ok=True)
        file_path = os.path.join(url_folder, "urls.txt")
        file.save(file_path)

        with open(file_path, "r") as fh:
            urls = [line.strip() for line in fh.readlines() if line.strip()]

        if not urls:
            return jsonify({"error": "Uploaded file contains no valid URLs"}), 400

        max_urls = Config.MAX_URLS_PER_SESSION
        if len(urls) > max_urls:
            return jsonify(
                {"message": f"Cannot process: Maximum of {max_urls} URLs exceeded."}
            )

        return jsonify(
            {
                "message": "URLs uploaded successfully",
                "file": file_path,
                "url_count": len(urls),
            }
        )

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@documents_bp.route("/upload_organization_kb", methods=["POST"])
@handle_route_errors
def upload_organization_kb():
    """Upload documents to Organization KB for specific disciplines."""
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        discipline_id = request.form.get("discipline_id")
        if not discipline_id:
            return jsonify({"error": "No discipline specified"}), 400

        cfg = _get_disciplines_config()
        discipline_path = None
        for discipline in cfg.get("disciplines", []):
            if discipline["id"] == discipline_id:
                discipline_path = discipline.get("kb_path")
                break

        if not discipline_path:
            return jsonify({"error": f"Invalid discipline: {discipline_id}"}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No valid files uploaded"}), 400

        discipline_dir = os.path.join(".", discipline_path)
        os.makedirs(discipline_dir, exist_ok=True)

        from langchain_text_splitters import RecursiveCharacterTextSplitter

        local_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP
        )

        saved_files: list[str] = []
        documents: list[Document] = []

        for file in files:
            if file.filename.endswith(".pdf"):
                file_path = os.path.join(discipline_dir, file.filename)
                file.save(file_path)
                saved_files.append(file.filename)

                text_content = extract_text_from_pdf(file_path)
                if text_content:
                    if isinstance(text_content, str):
                        text_content = [text_content]

                    for page_num, text in enumerate(text_content, start=1):
                        chunks = local_splitter.split_text(text)
                        for chunk in chunks:
                            documents.append(
                                Document(
                                    page_content=chunk,
                                    metadata={
                                        "source": file.filename,
                                        "type": "organization_pdf",
                                        "discipline": discipline_id,
                                        "page": page_num,
                                    },
                                )
                            )

        if documents:
            _create_organization_vector_db(discipline_id, documents)

        return jsonify(
            {
                "message": f"Successfully uploaded {len(saved_files)} files to {discipline_id}",
                "files": saved_files,
                "documents_created": len(documents),
            }
        )

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {exc}"}), 500


@documents_bp.route("/create_vector_db", methods=["POST"])
@handle_route_errors
def create_vector_db():
    """Parse PDFs and URLs, then create a Chroma Vector Database."""
    session_folder = _get_session_folder()
    if not session_folder:
        return jsonify({"error": "No active session. Refresh the page first."}), 400

    try:
        from langchain_chroma import Chroma

        persist_dir = os.path.join(VECTOR_DB_PATH, session_folder)
        os.makedirs(persist_dir, exist_ok=True)

        pdf_folder = os.path.join(BASE_STORAGE_PATH, "PDF", session_folder)
        url_folder = os.path.join(BASE_STORAGE_PATH, "URL", session_folder)

        pdf_files = (
            [
                os.path.join(pdf_folder, f)
                for f in os.listdir(pdf_folder)
                if f.endswith(".pdf")
            ]
            if os.path.exists(pdf_folder)
            else []
        )
        url_files = (
            [os.path.join(url_folder, f) for f in os.listdir(url_folder)]
            if os.path.exists(url_folder)
            else []
        )

        documents: list[Document] = []
        text_splitter = _get_text_splitter()

        for pdf_file in pdf_files:
            text_content = extract_text_from_pdf(pdf_file)
            if not text_content:
                logger.warning("No text extracted from %s", pdf_file)
                continue

            if isinstance(text_content, str):
                text_content = [text_content]

            for text in text_content:
                chunks = text_splitter.split_text(text)
                for page_num, chunk in enumerate(chunks, start=1):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": pdf_file,
                                "type": "pdf",
                                "page": page_num,
                            },
                        )
                    )

        for url_file in url_files:
            with open(url_file, "r", encoding="utf-8") as fh:
                urls = [line.strip() for line in fh.readlines() if line.strip()]

            for url in urls:
                try:
                    text = extract_text_from_url(url)
                    if not text:
                        logger.warning("No text extracted from %s", url)
                        continue
                    chunks = text_splitter.split_text(text)
                    for chunk in chunks:
                        documents.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": url, "type": "url"},
                            )
                        )
                except Exception as exc:
                    logger.error("Error extracting text from %s: %s", url, exc)

        if not documents:
            return (
                jsonify({"error": "No valid documents found to create the database"}),
                400,
            )

        logger.info("%d documents prepared for vectorization", len(documents))

        embeddings = _get_embeddings()
        Chroma.from_documents(
            documents, embedding=embeddings, persist_directory=persist_dir
        )

        rag_manager = _get_rag_manager()
        if rag_manager is not None:
            try:
                logger.info("Adding documents to RAG manager's local knowledge base...")
                rag_manager.add_documents_to_local(documents)
                logger.info(
                    "Successfully added %d documents to kb_local and updated lexical gate",
                    len(documents),
                )
            except Exception as exc:
                logger.warning("Error adding documents to RAG manager: %s", exc)

        return jsonify(
            {"message": "New Vector DB created", "db": session_folder}
        )

    except Exception as exc:
        logger.error("Error in /create_vector_db: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
