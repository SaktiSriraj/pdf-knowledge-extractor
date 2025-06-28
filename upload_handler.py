import os
import fitz
import streamlit as st
from config import PDF_DIR, TEXT_DIR

def handle_uploaded_pdf(uploaded_file):
    """Handle uploaded PDF file with error handling"""
    try:
        # Validate file
        if uploaded_file is None:
            raise ValueError("No file uploaded")
        
        if not uploaded_file.name.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")
        
        # Create directories
        os.makedirs(PDF_DIR, exist_ok=True)
        os.makedirs(TEXT_DIR, exist_ok=True)
        
        # Save uploaded file
        pdf_path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Verify it's a valid PDF
        try:
            doc = fitz.open(pdf_path)
            doc.close()
        except:
            os.remove(pdf_path)
            raise ValueError("Uploaded file is not a valid PDF")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Save text
        text_filename = uploaded_file.name + ".txt"
        text_path = os.path.join(TEXT_DIR, text_filename)
        
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        return text_path, len(text.split())
        
    except Exception as e:
        raise Exception(f"Failed to process uploaded PDF: {str(e)}")

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