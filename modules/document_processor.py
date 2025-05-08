import os
import glob
from typing import List, Tuple, Dict, Any
from PyPDF2 import PdfReader  # Use PyPDF2 instead of PyMuPDF
from docx import Document
import yaml
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import re

class DocumentProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        # Initialize NLTK - only download punkt, not punkt_tab
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt')
        
        # Set environment variable to disable tokenizers parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.documents_dir = self.config['paths']['documents_dir']
        self.chunk_size = self.config['retrieval']['chunk_size']

    def read_pdf(self, file_path: str) -> str:
        """Read PDF file using PyPDF2 instead of PyMuPDF"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""

    def read_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def read_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def read_csv(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            text_chunks = []
            
            # Add file name as context
            text_chunks.append(f"Document: {os.path.basename(file_path)}")
            
            # Add headers as a description
            headers = df.columns.tolist()
            text_chunks.append(f"Fields: {', '.join(headers)}")
            
            # Process each row
            for idx, row in df.iterrows():
                row_text = []
                for col in headers:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        # Clean and format the value
                        value = str(row[col]).replace('\n', ' ').strip()
                        row_text.append(f"{col}: {value}")
                if row_text:
                    text_chunks.append(" | ".join(row_text))
            
            result = "\n\n".join(text_chunks)
            print(f"Successfully processed CSV file {file_path} with {len(text_chunks)} entries")
            return result
        except Exception as e:
            print(f"Error processing CSV file {file_path}: {str(e)}")
            raise

    def load_documents(self) -> List[Tuple[str, str]]:
        documents = []
        abs_doc_dir = os.path.abspath(self.documents_dir)
        print(f"Searching for documents in: {abs_doc_dir}")
        
        if not os.path.exists(abs_doc_dir):
            print(f"Creating documents directory: {abs_doc_dir}")
            os.makedirs(abs_doc_dir, exist_ok=True)
            return documents
            
        for file_path in glob.glob(os.path.join(abs_doc_dir, '*.*')):
            print(f"Processing file: {file_path}")
            try:
                ext = os.path.splitext(file_path)[1].lower()
                text = None
                
                if ext == '.pdf':
                    text = self.read_pdf(file_path)
                elif ext == '.docx':
                    text = self.read_docx(file_path)
                elif ext == '.txt':
                    text = self.read_txt(file_path)
                elif ext == '.csv':
                    text = self.read_csv(file_path)
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue

                if text and text.strip():
                    chunks = self.chunk_text(text)
                    if chunks:
                        documents.extend([(file_path, chunk) for chunk in chunks])
                        print(f"Successfully processed {file_path} into {len(chunks)} chunks")
                    else:
                        print(f"No valid chunks extracted from {file_path}")
                else:
                    print(f"No text content extracted from {file_path}")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        print(f"Total documents processed: {len(documents)}")
        return documents

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces with sentence-level splitting and overlap
        """
        if not text or not text.strip():
            return []
        
        try:
            # Split into sentences using only punkt (not punkt_tab)
            try:
                # Simple sentence splitting as fallback
                sentences = []
                # First try with NLTK's sent_tokenize
                try:
                    sentences = sent_tokenize(text)
                except Exception as e:
                    print(f"NLTK tokenization failed: {e}, using fallback")
                    # Simple fallback: split by periods, question marks, and exclamation points
                    for s in re.split(r'(?<=[.!?])\s+', text):
                        if s.strip():
                            sentences.append(s.strip())
            
                if not sentences:
                    # If still no sentences, just split by newlines and limit length
                    sentences = [s.strip() for s in text.split('\n') if s.strip()]
                    if not sentences:
                        # Last resort: just chunk by character count
                        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            except Exception as e:
                print(f"Sentence splitting failed: {e}, using simple chunking")
                # Simple chunking as last resort
                return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            
            chunks = []
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_size = len(sentence)
                
                if current_size + sentence_size > self.chunk_size:
                    if current_chunk:
                        # Add the current chunk to the list
                        chunks.append(" ".join(current_chunk))
                        # Start a new chunk with the current sentence
                        current_chunk = [sentence]
                        current_size = sentence_size
                    else:
                        # Handle very long sentences by splitting them
                        chunks.append(sentence[:self.chunk_size])
                        if len(sentence) > self.chunk_size:
                            current_chunk = [sentence[self.chunk_size:]]
                            current_size = len(current_chunk[0])
                        else:
                            current_chunk = []
                            current_size = 0
                else:
                    current_chunk.append(sentence)
                    current_size += sentence_size
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Post-process chunks
            processed_chunks = []
            for chunk in chunks:
                # Clean up whitespace
                chunk = ' '.join(chunk.split())
                # Only keep chunks above minimum size
                if len(chunk) >= 50:  # Reduced minimum size
                    processed_chunks.append(chunk)
            
            return processed_chunks if processed_chunks else [text[:self.chunk_size]]
            
        except Exception as e:
            print(f"Error during text chunking: {str(e)}")
            # Fallback to simple chunking
            print("Falling back to simple chunking")
            # Split text into chunks of chunk_size
            chunks = []
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():  # Only keep non-empty chunks
                    chunks.append(chunk)
            return chunks if chunks else [text[:self.chunk_size]]

    def extract_metadata(self, file_path: str, text: str) -> Dict[str, Any]:
        """Extract metadata from documents for better context"""
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_type': os.path.splitext(file_path)[1],
            'char_count': len(text),
            'word_count': len(text.split()),
            'creation_date': os.path.getctime(file_path),
            'modification_date': os.path.getmtime(file_path),
        }
        return metadata


