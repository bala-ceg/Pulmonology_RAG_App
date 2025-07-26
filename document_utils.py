import os
import re
import fitz
import pdfplumber
from bs4 import BeautifulSoup
from apify_client import ApifyClient
from .helpers import clean_extracted_text

client = ApifyClient(os.getenv("apify_api_key"))


def extract_text_from_pdf(pdf_file: str):
    text_content = []
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            clean_text = clean_extracted_text(page.get_text())
            text_content.append(clean_text)
    return text_content


def extract_text_from_url(url: str):
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

    run = client.actor("apify/website-content-crawler").call(run_input=run_input)

    full_text = ""
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        page_text = item.get("text", "")
        page_text = re.sub(r"[^\x00-\x7F]+", " ", page_text)
        page_text = re.sub(r"\s+", " ", page_text).strip()
        full_text += page_text + "\n"
    return full_text.strip()
