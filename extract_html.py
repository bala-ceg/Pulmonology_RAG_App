import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from pathlib import Path
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def extract_text_from_url(url):
    """
    Extracts the textual content from a URL (HTML page).
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


    service = Service(executable_path="/usr/local/bin/chromedriver")  # Update with your chromedriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        driver.get(url)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        text_content = soup.get_text(separator="\n", strip=True)
        return text_content
    finally:
        driver.quit()

def extract_tables_from_url(url):
    """
    Extracts tables from a URL and returns them as pandas DataFrames.
    """

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


    service = Service(executable_path="/usr/local/bin/chromedriver")  # Update with your chromedriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    tables = []
    # Find all tables in the HTML
    html_tables = soup.find_all('table')
    for table in html_tables:
        rows = table.find_all('tr')
        table_data = []
        for row in rows:
            cols = row.find_all(['td', 'th'])
            cols = [ele.text.strip() for ele in cols]
            table_data.append(cols)
        table_df = pd.DataFrame(table_data)
        tables.append(table_df)
    
    return tables

def extract_images_from_url(url, output_folder="extracted_images"):
    """
    Extracts image URLs from a webpage and downloads them.
    """
    os.makedirs(output_folder, exist_ok=True)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")


    service = Service(executable_path="/usr/local/bin/chromedriver")  # Update with your chromedriver path
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    images = []
    img_tags = soup.find_all('img')
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            img_url = img_url if img_url.startswith('http') else url + img_url
            img_name = Path(img_url).name
            img_path = os.path.join(output_folder, img_name)
            img_data = requests.get(img_url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)
            images.append({"image_url": img_url, "image_path": img_path})
    
    return images

def process_url(url, output_folder="extracted_images"):
    """
    Processes a URL to extract text, tables, and images, and returns metadata.
    """
    metadata = {"url": url, "text": "", "tables": []}

    # Extract text
    text_content = extract_text_from_url(url)
    metadata["text"] = text_content

    # Extract tables
    tables = extract_tables_from_url(url)
    metadata["tables"] = [table.to_dict() for table in tables]

    # # # Extract images
    # images = extract_images_from_url(url, output_folder)
    # metadata["images"] = images

    return metadata

def process_urls_from_txt(txt_file, output_folder="extracted_images", metadata_file="metadata.json"):
    """
    Processes all URLs listed in a .txt file and extracts metadata.
    """
    with open(txt_file, 'r') as f:
        urls = f.readlines()

    all_metadata = []

    for url in urls:
        url = url.strip()
        print(f"Processing URL: {url}...")
        metadata = process_url(url, output_folder)
        all_metadata.append(metadata)

    # Save metadata to JSON for later use
    with open(metadata_file, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"Metadata saved to {metadata_file}.")
    return all_metadata

# Example Usage
if __name__ == "__main__":
    txt_file = "/Users/bseetharaman/Desktop/Bala/assignments/ai_medical_app/KB/URL/Pulmonology URLs.txt"               # Replace with the path to your .txt file containing URLs
    output_folder = "output_folder"     # Folder to save images
    metadata_file = "url_metadata.json" # File to save metadata

    metadata = process_urls_from_txt(txt_file, output_folder, metadata_file)
    print("Extraction completed.")
