from flask import Flask, request, jsonify, render_template
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize LLM
llm = OpenAI(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model_name=os.getenv('llm_model_name')
)

# Helper function to load metadata
def load_metadata(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Helper function to process PDF metadata
def process_pdf_metadata(pdf_metadata):
    documents = []
    for pdf in pdf_metadata:
        file_name = pdf.get("file_name", "Unknown File")
        for page in pdf.get("text", []):
            if isinstance(page, dict) and "text" in page:
                page_content = page["text"]
                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={"source": file_name, "type": "pdf", "page": page.get("page", "Unknown Page")}
                    )
                )
    return documents

# Helper function to process URL metadata
def process_url_metadata(url_metadata):
    documents = []
    for url_data in url_metadata:
        url = url_data.get("url", "Unknown URL")
        text_content = url_data.get("text", "")
        documents.append(
            Document(
                page_content=text_content,
                metadata={"source": url, "type": "url"}
            )
        )
    return documents

# Load and process metadata
pdf_metadata = load_metadata("pdf_metadata.json")
url_metadata = load_metadata("url_metadata.json")
pdf_documents = process_pdf_metadata(pdf_metadata)
url_documents = process_url_metadata(url_metadata)
all_documents = pdf_documents + url_documents

# Create embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv('openai_api_key'),
    base_url=os.getenv('base_url'),
    model=os.getenv('embedding_model_name')
)

# Split documents and create FAISS vector store
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
split_docs = text_splitter.split_documents(all_documents)
vector_store = FAISS.from_documents(split_docs, embeddings)

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
