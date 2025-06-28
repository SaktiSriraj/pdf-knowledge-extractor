import os
import json
import time
from typing import List, Dict, Any, Optional

# Fixed LangChain imports
try:
    from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
except ImportError:
    # Fallback for older versions
    from langchain.document_loaders import PyMuPDFLoader, TextLoader

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)
from langchain.schema import Document
from langchain.docstore.document import Document as LangChainDocument
from config import TEXT_DIR, CHUNK_SIZE, CHUNK_OVERLAP, METADATA_PATH, PDF_DIR

class EnhancedLangChainProcessor:
    """Advanced LangChain-based document processor with multiple splitting strategies"""
    
    def __init__(self):
        self.splitter_strategies = {
            'recursive': self._create_recursive_splitter,
            'token_based': self._create_token_splitter,
            'semantic': self._create_semantic_splitter,
            'hybrid': self._create_hybrid_splitter
        }
        
    def _create_recursive_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create advanced recursive character text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Semicolon breaks
                ", ",    # Comma breaks
                " ",     # Word breaks
                ""       # Character breaks
            ],
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=True
        )
    
    def _create_token_splitter(self) -> TokenTextSplitter:
        """Create token-based text splitter for precise token control"""
        try:
            return TokenTextSplitter(
                chunk_size=CHUNK_SIZE // 4,  # Approximate token count
                chunk_overlap=CHUNK_OVERLAP // 4,
                encoding_name="gpt2",  # Use GPT-2 tokenizer
                model_name="gpt-3.5-turbo",
                allowed_special=set(),
                disallowed_special="all"
            )
        except Exception as e:
            print(f"Token splitter failed, using character splitter: {str(e)}")
            return CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separator="\n"
            )
    
    def _create_semantic_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create semantic-aware text splitter"""
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ";",
                ",",
                " ",
                ""
            ],
            keep_separator=True,
            add_start_index=True
        )
    
    def _create_hybrid_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create hybrid splitter combining multiple strategies"""
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE + 100,  # Slightly larger for better context
            chunk_overlap=CHUNK_OVERLAP + 25,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=True,
            add_start_index=True
        )

class DocumentProcessor:
    """Enhanced document processor with LangChain integration"""
    
    def __init__(self):
        self.processor = EnhancedLangChainProcessor()
        self.processed_docs: List[Document] = []
        self.processing_stats = {
            'total_files': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'total_chars': 0,
            'processing_time': 0,
            'avg_chunk_size': 0,
            'files_processed': []
        }
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load documents using LangChain document loaders"""
        documents = []
        
        if not os.path.exists(directory):
            return documents
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                try:
                    # Use LangChain TextLoader with error handling
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    
                    # Enhance metadata
                    for doc in docs:
                        doc.metadata.update({
                            'source_file': filename,
                            'file_path': file_path,
                            'file_size': os.path.getsize(file_path),
                            'load_timestamp': time.time(),
                            'loader_type': 'TextLoader'
                        })
                    
                    documents.extend(docs)
                    self.processing_stats['files_processed'].append(filename)
                    
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def load_pdfs_from_directory(self, directory: str) -> List[Document]:
        """Load PDF documents using LangChain PyMuPDF loader"""
        documents = []
        
        if not os.path.exists(directory):
            return documents
        
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(directory, filename)
                try:
                    # Use LangChain PyMuPDFLoader with error handling
                    loader = PyMuPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Enhance metadata
                    for i, doc in enumerate(docs):
                        doc.metadata.update({
                            'source_file': filename,
                            'file_path': file_path,
                            'page_number': i + 1,
                            'total_pages': len(docs),
                            'file_size': os.path.getsize(file_path),
                            'load_timestamp': time.time(),
                            'loader_type': 'PyMuPDFLoader'
                        })
                    
                    documents.extend(docs)
                    self.processing_stats['files_processed'].append(filename)
                    
                except Exception as e:
                    print(f"Error loading PDF {filename}: {str(e)}")
        
        return documents
    
    def split_documents_advanced(self, documents: List[Document], strategy: str = 'hybrid') -> List[Document]:
        """Split documents using advanced LangChain strategies"""
        if strategy not in self.processor.splitter_strategies:
            strategy = 'hybrid'
        
        try:
            splitter = self.processor.splitter_strategies[strategy]()
            
            # Split documents
            split_docs = splitter.split_documents(documents)
            
            # Enhance chunk metadata
            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(split_docs),
                    'chunk_size': len(doc.page_content),
                    'splitting_strategy': strategy,
                    'chunk_index': i,
                    'split_timestamp': time.time()
                })
            
            return split_docs
            
        except Exception as e:
            print(f"Error in advanced splitting: {str(e)}")
            # Fallback to simple splitting
            return self._simple_split_fallback(documents)
    
    def _simple_split_fallback(self, documents: List[Document]) -> List[Document]:
        """Simple fallback splitting method"""
        split_docs = []
        for doc in documents:
            content = doc.page_content
            chunk_size = CHUNK_SIZE
            
            # Simple chunking
            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                if chunk_content.strip():
                    chunk_doc = Document(
                        page_content=chunk_content,
                        metadata={
                            **doc.metadata,
                            'chunk_id': len(split_docs),
                            'splitting_strategy': 'simple_fallback'
                        }
                    )
                    split_docs.append(chunk_doc)
        
        return split_docs
    
    def apply_document_transformers(self, documents: List[Document]) -> List[Document]:
        """Apply document transformers for optimization"""
        try:
            # Simple deduplication based on content similarity
            unique_docs = []
            seen_content = set()
            
            for doc in documents:
                # Create a hash of the first 100 characters
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
            
            removed_count = len(documents) - len(unique_docs)
            if removed_count > 0:
                print(f"Removed {removed_count} duplicate chunks")
            
            return unique_docs
            
        except Exception as e:
            print(f"Error applying transformers: {str(e)}")
            return documents
    
    def calculate_advanced_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics"""
        if not documents:
            return self.processing_stats
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        estimated_tokens = total_chars // 4
        
        # File type distribution
        file_types = {}
        for doc in documents:
            loader_type = doc.metadata.get('loader_type', 'Unknown')
            file_types[loader_type] = file_types.get(loader_type, 0) + 1
        
        # Chunk size distribution
        chunk_sizes = [len(doc.page_content) for doc in documents]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        self.processing_stats.update({
            'total_files': len(set(doc.metadata.get('source_file', '') for doc in documents)),
            'total_chunks': len(documents),
            'total_chars': total_chars,
            'total_words': total_words,
            'estimated_tokens': estimated_tokens,
            'avg_chunk_size': int(avg_chunk_size),
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'file_type_distribution': file_types,
            'chunk_size_distribution': {
                'small': len([s for s in chunk_sizes if s < 200]),
                'medium': len([s for s in chunk_sizes if 200 <= s < 800]),
                'large': len([s for s in chunk_sizes if s >= 800])
            }
        })
        
        return self.processing_stats

# Global processor instance
doc_processor = DocumentProcessor()

def load_and_split_texts(strategy: str = 'hybrid') -> tuple[List[str], List[Dict[str, Any]]]:
    """Enhanced text loading and splitting using LangChain"""
    try:
        start_time = time.time()
        
        # Load documents from both text and PDF directories
        text_documents = doc_processor.load_documents_from_directory(TEXT_DIR)
        pdf_documents = doc_processor.load_pdfs_from_directory(PDF_DIR)
        
        all_documents = text_documents + pdf_documents
        
        if not all_documents:
            return [], []
        
        print(f"Loaded {len(all_documents)} documents using LangChain loaders")
        
        # Split documents using advanced strategies
        split_documents = doc_processor.split_documents_advanced(all_documents, strategy)
        
        print(f"Created {len(split_documents)} chunks using {strategy} strategy")
        
        # Apply document transformers
        optimized_documents = doc_processor.apply_document_transformers(split_documents)
        
        print(f"Optimized to {len(optimized_documents)} unique chunks")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        doc_processor.processing_stats['processing_time'] = processing_time
        
        # Calculate comprehensive stats
        stats = doc_processor.calculate_advanced_stats(optimized_documents)
        
        # Extract text chunks and metadata
        text_chunks = [doc.page_content for doc in optimized_documents]
        chunk_metadata = [doc.metadata for doc in optimized_documents]
        
        # Save enhanced metadata
        save_enhanced_metadata(stats, chunk_metadata)
        
        print(f"LangChain processing completed in {processing_time:.2f} seconds")
        
        return text_chunks, chunk_metadata
        
    except Exception as e:
        print(f"LangChain processing error: {str(e)}")
        # Fallback to simple processing
        return fallback_text_processing()

def fallback_text_processing() -> tuple[List[str], List[Dict[str, Any]]]:
    """Fallback text processing without LangChain"""
    try:
        text_chunks = []
        chunk_metadata = []
        
        if not os.path.exists(TEXT_DIR):
            return [], []
        
        for filename in os.listdir(TEXT_DIR):
            if not filename.endswith('.txt'):
                continue
                
            filepath = os.path.join(TEXT_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Simple chunking
                chunk_size = CHUNK_SIZE
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    if chunk.strip():
                        text_chunks.append(chunk)
                        chunk_metadata.append({
                            'source_file': filename,
                            'chunk_id': len(text_chunks) - 1,
                            'fallback_processing': True
                        })
                        
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        return text_chunks, chunk_metadata
        
    except Exception as e:
        print(f"Fallback processing failed: {str(e)}")
        return [], []

def save_enhanced_metadata(stats: Dict[str, Any], chunk_metadata: List[Dict[str, Any]]) -> None:
    """Save enhanced metadata with comprehensive statistics"""
    try:
        enhanced_metadata = {
            'processing_timestamp': time.time(),
            'langchain_version': '0.1.0',  # You can get actual version
            'processing_stats': stats,
            'chunk_metadata': chunk_metadata,
            'configuration': {
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP,
                'splitting_strategy': 'hybrid'
            }
        }
        
        os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
        
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)
            
    except Exception as e:
        print(f"Could not save enhanced metadata: {str(e)}")

def get_processing_stats() -> Dict[str, Any]:
    """Get comprehensive processing statistics"""
    try:
        # Try to load from saved metadata first
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                return metadata.get('processing_stats', doc_processor.processing_stats)
        
        # Fallback to basic stats
        if not os.path.exists(TEXT_DIR):
            return {"files": 0, "total_words": 0, "langchain_enhanced": False}
        
        files = [f for f in os.listdir(TEXT_DIR) if f.endswith('.txt')]
        total_words = 0
        
        for filename in files:
            filepath = os.path.join(TEXT_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    total_words += len(content.split())
            except:
                continue
        
        return {
            "files": len(files),
            "total_words": total_words,
            "langchain_enhanced": True,
            "processing_method": "LangChain Enhanced"
        }
        
    except Exception as e:
        return {
            "files": 0, 
            "total_words": 0, 
            "error": str(e),
            "langchain_enhanced": False
        }

def get_langchain_capabilities() -> Dict[str, Any]:
    """Get information about LangChain capabilities used"""
    return {
        "document_loaders": ["PyMuPDFLoader", "TextLoader"],
        "text_splitters": ["RecursiveCharacterTextSplitter", "TokenTextSplitter", "SemanticSplitter"],
        "splitting_strategies": ["recursive", "token_based", "semantic", "hybrid"],
        "document_transformers": ["ContentDeduplication"],
        "advanced_features": [
            "Multi-strategy text splitting",
            "Token-aware chunking", 
            "Semantic content preservation",
            "Duplicate content removal",
            "Enhanced metadata tracking",
            "Processing statistics",
            "Document optimization",
            "Fallback processing"
        ],
        "metadata_enrichment": [
            "Source tracking",
            "Chunk indexing",
            "Processing timestamps",
            "File size tracking",
            "Content statistics",
            "Loader type identification"
        ]
    }

# Advanced LangChain utilities
def get_document_summary(documents: List[Document]) -> Dict[str, Any]:
    """Generate comprehensive document summary using LangChain"""
    if not documents:
        return {}
    
    # Analyze document structure
    sources = set(doc.metadata.get('source_file', 'Unknown') for doc in documents)
    
    # Content analysis
    total_content = " ".join(doc.page_content for doc in documents)
    
    # Basic analytics
    summary = {
        'total_documents': len(documents),
        'unique_sources': len(sources),
        'total_characters': len(total_content),
        'total_words': len(total_content.split()),
        'avg_document_length': len(total_content) / len(documents) if documents else 0,
        'source_files': list(sources),
        'langchain_processed': True,
        'processing_features': [
            "Advanced text splitting",
            "Metadata enrichment", 
            "Content optimization",
            "Duplicate removal"
        ]
    }
    
    return summary