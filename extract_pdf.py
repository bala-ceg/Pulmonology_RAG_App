import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import os
import json
from pathlib import Path

def extract_text_from_pdf(pdf_file):
    """
    Extracts the text content from a PDF.
    """
    text_content = []
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text_content.append({"page": page_num + 1, "text": page.get_text()})
    return text_content

def extract_tables_from_pdf(pdf_file):
    """
    Extracts tables from a PDF and returns them as pandas DataFrames.
    """
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            for table in page_tables:
                table_df = pd.DataFrame(table)  
                tables.append({"page": page_num, "table": table_df})
    return tables

def extract_images_from_pdf(pdf_file, output_folder="extracted_images"):
    """
    Extracts images from a PDF and saves them to the specified folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    images = []
    pdf = fitz.open(pdf_file)
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        for img_index, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_folder, f"{Path(pdf_file).stem}_page_{page_num+1}_img_{img_index}.{image_ext}")
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            images.append({"page": page_num + 1, "image_path": image_path})
    return images

def process_pdf(pdf_file, output_folder="extracted_images"):
    """
    Extracts text, tables, and images from a PDF and returns metadata.
    """
    metadata = {"file_name": pdf_file, "text": [], "tables": [], "images": []}

    text_content = extract_text_from_pdf(pdf_file)
    metadata["text"] = text_content

    tables = extract_tables_from_pdf(pdf_file)
    metadata["tables"] = [{"page": t["page"], "table_data": t["table"].to_dict()} for t in tables]


    images = extract_images_from_pdf(pdf_file, output_folder)
    metadata["images"] = images

    return metadata

def process_pdfs_in_folder(folder_path, output_folder="extracted_images", metadata_file="metadata.json"):
    """
    Processes all PDFs in a folder and extracts metadata.
    """
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
    all_metadata = []

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        metadata = process_pdf(pdf_file, output_folder)
        all_metadata.append(metadata)


    with open(metadata_file, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"Metadata saved to {metadata_file}.")
    return all_metadata


if __name__ == "__main__":
    folder_path = "/Users/bseetharaman/Desktop/Bala/assignments/ai_medical_app/KB/PDF"  
    output_folder = "output_folder"       
    metadata_file = "metadata.json"         

    metadata = process_pdfs_in_folder(folder_path, output_folder, metadata_file)
    print("Extraction completed.")
