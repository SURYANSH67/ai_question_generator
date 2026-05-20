import os
import re
import pdfplumber
from pypdf import PdfReader
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(file_path_or_bytes) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file with page numbers and metadata.
    Supports either a file path or a bytes-like object / UploadedFile.
    """
    pages_data = []
    
    # Try using pdfplumber first for better formatting and table preservation
    try:
        if hasattr(file_path_or_bytes, "read"):
            # It's a file-like object (e.g. Streamlit UploadedFile)
            # Reset seek position just in case
            file_path_or_bytes.seek(0)
            pdf = pdfplumber.open(file_path_or_bytes)
        else:
            pdf = pdfplumber.open(file_path_or_bytes)
            
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(layout=False) or ""
            text = clean_text(text)
            if text.strip():
                pages_data.append({
                    "page_number": i + 1,
                    "text": text,
                    "character_count": len(text),
                    "word_count": len(text.split())
                })
        pdf.close()
    except Exception as e:
        # Fallback to PyPDF if pdfplumber fails
        print(f"pdfplumber failed: {e}. Falling back to PyPDF.")
        try:
            if hasattr(file_path_or_bytes, "seek"):
                file_path_or_bytes.seek(0)
            reader = PdfReader(file_path_or_bytes)
            pages_data = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = clean_text(text)
                if text.strip():
                    pages_data.append({
                        "page_number": i + 1,
                        "text": text,
                        "character_count": len(text),
                        "word_count": len(text.split())
                    })
        except Exception as fallback_e:
            raise ValueError(f"Could not parse PDF: {fallback_e}")
            
    return pages_data

def clean_text(text: str) -> str:
    """
    Clean and preprocess extracted text.
    - Resolves double spaces.
    - Cleans up weird page header/footer fragments or redundant hyphens.
    """
    if not text:
        return ""
    # Remove weird hyphens split at lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Replace multiple spaces/newlines with single space/newline
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text

def chunk_text(pages_data: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split extracted page data into chunks while retaining page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_id = 0
    
    for page in pages_data:
        page_text = page["text"]
        page_num = page["page_number"]
        
        # Split the text on the page
        page_chunks = splitter.split_text(page_text)
        
        for text_chunk in page_chunks:
            if text_chunk.strip():
                chunks.append({
                    "chunk_id": chunk_id,
                    "page_number": page_num,
                    "text": text_chunk,
                    "character_count": len(text_chunk),
                    "word_count": len(text_chunk.split())
                })
                chunk_id += 1
                
    return chunks

def extract_topics_with_llm(text_sample: str, openai_api_key: str, base_url: Optional[str] = None, model_name: str = "gpt-4o-mini") -> List[str]:
    """
    Extract key chapters, topics, or concepts from a sample of the text using the configured provider.
    Useful for auto-populating choices for teachers.
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=openai_api_key, base_url=base_url)
    prompt = f"""
    Analyze the following educational content excerpt and extract a list of 5 to 10 key subjects/chapters/topics/concepts discussed.
    Provide the output as a simple comma-separated list of topics, with no numbering, introduction, or formatting.
    
    Content:
    {text_sample[:8000]}
    
    Topics:
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an educational assistant designed to extract topic names from text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        topics_str = response.choices[0].message.content.strip()
        topics = [t.strip() for t in topics_str.split(",") if t.strip()]
        return topics
    except Exception as e:
        print(f"Error extracting topics with {model_name}: {e}")
        # Return fallback topics
        return ["General Content", "Key Concepts", "Overview"]
