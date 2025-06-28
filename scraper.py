import os
import requests
import fitz  # PyMuPDF
from urllib.parse import urlparse
import streamlit as st
from config import PDF_DIR, TEXT_DIR

def is_valid_url(url):
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_pdf(url, save_dir=PDF_DIR):
    """Download PDF from URL with error handling"""
    try:
        if not is_valid_url(url):
            raise ValueError("Invalid URL format")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Download with timeout and headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        
        # Validate content type
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            # Still try to process, might be a direct PDF link
            pass
        
        # Generate filename
        filename = url.split("/")[-1].split('?')[0]
        if not filename or not filename.endswith('.pdf'):
            filename = f"downloaded_{hash(url) % 10000}.pdf"
        
        filepath = os.path.join(save_dir, filename)
        
        # Download file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify it's a valid PDF
        try:
            doc = fitz.open(filepath)
            doc.close()
        except:
            os.remove(filepath)
            raise ValueError("Downloaded file is not a valid PDF")
        
        return filepath
        
    except requests.RequestException as e:
        raise Exception(f"Failed to download PDF: Network error - {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(filepath):
    """Extract text from PDF with error handling"""
    try:
        doc = fitz.open(filepath)
        text_parts = []
        
        for page_num, page in enumerate(doc):
            try:
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
            except Exception as e:
                st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        
        doc.close()
        
        if not text_parts:
            raise ValueError("No text could be extracted from the PDF")
        
        return "\n\n".join(text_parts)
        
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def process_pdf_url(url):
    """Process PDF from URL - download and extract text"""
    try:
        # Download PDF
        pdf_path = download_pdf(url)
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Save text
        os.makedirs(TEXT_DIR, exist_ok=True)
        text_filename = os.path.basename(pdf_path) + ".txt"
        text_path = os.path.join(TEXT_DIR, text_filename)
        
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return text_path, len(text.split())
        
    except Exception as e:
        raise Exception(f"Failed to process PDF from URL: {str(e)}")