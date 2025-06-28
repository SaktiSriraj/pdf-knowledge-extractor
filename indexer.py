import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Fixed LangChain imports for current version (no more warnings)
try:
    from langchain_community.vectorstores import FAISS as LangChainFAISS
except ImportError:
    from langchain.vectorstores import FAISS as LangChainFAISS

# Use the new langchain-huggingface package (no deprecation warnings)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_EMBEDDINGS_AVAILABLE = True
    print("✅ Using langchain-huggingface (latest)")
except ImportError:
    try:
        # Fallback to older versions with warning suppression
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        LANGCHAIN_EMBEDDINGS_AVAILABLE = True
        print("⚠️ Using older LangChain embeddings (consider upgrading)")
    except ImportError:
        try:
            import warnings
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
            
            from langchain.embeddings import HuggingFaceEmbeddings
            LANGCHAIN_EMBEDDINGS_AVAILABLE = True
            print("⚠️ Using legacy LangChain embeddings")
        except ImportError:
            LANGCHAIN_EMBEDDINGS_AVAILABLE = False
            print("ℹ️ LangChain embeddings not available, using sentence-transformers directly")

from langchain.schema import Document
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME

class OptimizedLangChainIndexer:
    """Performance-optimized indexer with latest LangChain compatibility"""
    
    def __init__(self):
        self.embeddings = None
        self.sentence_transformer = None
        self.vector_store = None
        self._model_loaded = False
        
    def _load_models_once(self):
        """Load models once and cache them"""
        if self._model_loaded:
            return
            
        try:
            print("🔄 Loading embedding model (one-time setup)...")
            start_time = time.time()
            
            # Load sentence transformer directly (faster and more reliable)
            self.sentence_transformer = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                device='cpu'
            )
            
            # Try to prepare LangChain embeddings with new package
            if LANGCHAIN_EMBEDDINGS_AVAILABLE:
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name=EMBEDDING_MODEL_NAME,
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': False}  # Faster
                    )
                    print("✅ LangChain embeddings initialized (no warnings)")
                except Exception as e:
                    print(f"⚠️ LangChain embeddings failed: {str(e)}")
                    self.embeddings = None
            else:
                print("ℹ️ Using sentence-transformers directly (recommended for performance)")
                self.embeddings = None
            
            load_time = time.time() - start_time
            print(f"✅ Models loaded in {load_time:.2f} seconds")
            self._model_loaded = True
            
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}")
    
    def create_vectorstore_fast(self, chunks: List[str], metadata: List[Dict] = None) -> tuple:
        """Fast vector store creation with optimizations"""
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            
            self._load_models_once()
            
            print(f"🚀 Creating optimized index for {len(chunks)} chunks...")
            total_start = time.time()
            
            # Generate embeddings efficiently
            embed_start = time.time()
            embeddings = self._generate_embeddings_fast(chunks)
            embed_time = time.time() - embed_start
            print(f"⚡ Generated embeddings in {embed_time:.2f} seconds")
            
            # Create FAISS index
            index_start = time.time()
            dimension = embeddings.shape[1]
            
            # Use faster index type for small datasets
            if len(chunks) < 1000:
                index = faiss.IndexFlatIP(dimension)  # Inner product (faster)
                # Normalize embeddings for IP similarity
                faiss.normalize_L2(embeddings)
            else:
                index = faiss.IndexFlatL2(dimension)  # L2 for larger datasets
            
            index.add(embeddings)
            index_time = time.time() - index_start
            print(f"⚡ Built FAISS index in {index_time:.2f} seconds")
            
            # Save efficiently
            save_start = time.time()
            self._save_index_fast(index, chunks, metadata)
            save_time = time.time() - save_start
            print(f"⚡ Saved index in {save_time:.2f} seconds")
            
            total_time = time.time() - total_start
            print(f"🎉 Total indexing time: {total_time:.2f} seconds")
            
            return index, chunks
            
        except Exception as e:
            raise Exception(f"Fast vector store creation failed: {str(e)}")
    
    def _generate_embeddings_fast(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings with performance optimizations"""
        try:
            # Use sentence transformer directly (faster than LangChain wrapper)
            chunk_count = len(chunks)
            
            if chunk_count <= 50:
                # Small batch - process all at once
                print(f"   🔥 Processing {chunk_count} chunks in single batch")
                embeddings = self.sentence_transformer.encode(
                    chunks,
                    batch_size=min(32, chunk_count),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False  # Skip normalization for speed
                )
            else:
                # Large batch - process in chunks with progress
                print(f"   📊 Processing {chunk_count} chunks in batches")
                batch_size = min(64, chunk_count)
                all_embeddings = []
                
                for i in range(0, chunk_count, batch_size):
                    batch = chunks[i:i + batch_size]
                    batch_embeddings = self.sentence_transformer.encode(
                        batch,
                        batch_size=min(32, len(batch)),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=False
                    )
                    all_embeddings.append(batch_embeddings)
                    
                    # Show progress every few batches
                    if i % (batch_size * 2) == 0:
                        progress = min((i + batch_size) / chunk_count * 100, 100)
                        print(f"      📈 Embedding progress: {progress:.0f}%")
                
                embeddings = np.vstack(all_embeddings)
            
            return embeddings.astype('float32')
            
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def _save_index_fast(self, index, chunks: List[str], metadata: List[Dict] = None):
        """Save index files efficiently"""
        try:
            # Create directory
            index_dir = os.path.dirname(FAISS_INDEX_PATH)
            os.makedirs(index_dir, exist_ok=True)
            
            # Save traditional FAISS (fast)
            traditional_path = FAISS_INDEX_PATH + "_traditional"
            faiss.write_index(index, traditional_path)
            
            # Save chunks and metadata (parallel)
            chunks_path = FAISS_INDEX_PATH + "_chunks.pkl"
            metadata_path = FAISS_INDEX_PATH + "_metadata.pkl"
            
            def save_chunks():
                with open(chunks_path, "wb") as f:
                    pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            def save_metadata():
                if metadata:
                    with open(metadata_path, "wb") as f:
                        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(save_chunks)]
                if metadata:
                    futures.append(executor.submit(save_metadata))
                
                # Wait for completion
                for future in futures:
                    future.result()
            
            # Try to save LangChain format (optional, in background) - only for small datasets
            if self.embeddings and len(chunks) < 300:  # Reduced threshold for faster processing
                self._save_langchain_format_async(chunks, metadata)
            else:
                print("ℹ️ Skipping LangChain format for large dataset (performance optimization)")
            
        except Exception as e:
            raise Exception(f"Index saving failed: {str(e)}")
    
    def _save_langchain_format_async(self, chunks: List[str], metadata: List[Dict] = None):
        """Save LangChain format in background (optional)"""
        def save_langchain():
            try:
                if not self.embeddings:
                    return
                
                print("🔄 Saving LangChain format in background...")
                
                # Convert to documents
                documents = []
                for i, chunk in enumerate(chunks):
                    doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                    doc_metadata.update({'chunk_index': i})
                    
                    doc = Document(page_content=chunk, metadata=doc_metadata)
                    documents.append(doc)
                
                # Create and save vector store
                vector_store = LangChainFAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                
                index_dir = os.path.dirname(FAISS_INDEX_PATH)
                vector_store.save_local(index_dir)
                self.vector_store = vector_store  # Cache it
                print("✅ LangChain format saved successfully")
                
            except Exception as e:
                print(f"⚠️ Background LangChain save failed (non-critical): {str(e)}")
        
        # Run in background thread
        thread = threading.Thread(target=save_langchain, daemon=True)
        thread.start()
    
    def load_vectorstore(self, index_path: str):
        """Load LangChain vector store from disk"""
        try:
            index_dir = os.path.dirname(index_path)
            
            if not os.path.exists(index_dir):
                raise FileNotFoundError(f"Vector store directory not found: {index_dir}")
            
            # Check for required files
            required_files = [
                os.path.join(index_dir, "index.faiss"),
                os.path.join(index_dir, "index.pkl")
            ]
            
            if not all(os.path.exists(f) for f in required_files):
                raise FileNotFoundError("LangChain vector store files not found")
            
            if not self.embeddings:
                self._load_models_once()
            
            if not self.embeddings:
                raise Exception("Embeddings not initialized")
            
            # Load using LangChain's built-in load method
            vector_store = LangChainFAISS.load_local(
                index_dir,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.vector_store = vector_store
            print("✅ LangChain vector store loaded successfully!")
            
            return vector_store
            
        except Exception as e:
            raise Exception(f"Failed to load vector store: {str(e)}")

# Global optimized indexer
langchain_indexer = OptimizedLangChainIndexer()

def build_faiss_index(chunks: List[str], chunk_metadata: Optional[List[Dict]] = None):
    """Optimized FAISS index building"""
    try:
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        print(f"🏗️ Building optimized index for {len(chunks)} chunks...")
        
        # Use optimized indexer
        index, processed_chunks = langchain_indexer.create_vectorstore_fast(chunks, chunk_metadata)
        
        return index, processed_chunks
        
    except Exception as e:
        raise Exception(f"Failed to build optimized index: {str(e)}")

def load_faiss_index():
    """Fast index loading with caching"""
    try:
        # Check cache first
        if hasattr(load_faiss_index, '_cached_result'):
            cached_time = getattr(load_faiss_index, '_cached_time', 0)
            if time.time() - cached_time < 300:  # 5-minute cache
                print("⚡ Using cached index")
                return load_faiss_index._cached_result
        
        load_start = time.time()
        
        # Load files
        traditional_path = FAISS_INDEX_PATH + "_traditional"
        chunks_path = FAISS_INDEX_PATH + "_chunks.pkl"
        metadata_path = FAISS_INDEX_PATH + "_metadata.pkl"
        
        if not os.path.exists(traditional_path):
            raise FileNotFoundError("No FAISS index found. Please process some PDFs first.")
        
        # Load in parallel for speed
        def load_index():
            return faiss.read_index(traditional_path)
        
        def load_chunks():
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    return pickle.load(f)
            return []
        
        def load_metadata():
            if os.path.exists(metadata_path):
                with open(metadata_path, "rb") as f:
                    return pickle.load(f)
            return None
        
        # Load in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            index_future = executor.submit(load_index)
            chunks_future = executor.submit(load_chunks)
            metadata_future = executor.submit(load_metadata)
            
            index = index_future.result()
            chunks = chunks_future.result()
            metadata = metadata_future.result()
        
        load_time = time.time() - load_start
        print(f"⚡ Loaded index in {load_time:.2f} seconds ({len(chunks)} chunks)")
        
        # Cache result
        result = (index, chunks, metadata)
        load_faiss_index._cached_result = result
        load_faiss_index._cached_time = time.time()
        
        return result
        
    except Exception as e:
        raise Exception(f"Failed to load index: {str(e)}")

def initialize_embedding_model():
    """Fast model initialization"""
    try:
        langchain_indexer._load_models_once()
        return langchain_indexer.sentence_transformer
    except Exception as e:
        raise Exception(f"Failed to initialize model: {str(e)}")

def get_index_info():
    """Fast index info retrieval"""
    try:
        traditional_path = FAISS_INDEX_PATH + "_traditional"
        chunks_path = FAISS_INDEX_PATH + "_chunks.pkl"
        
        info = {
            "exists": False,
            "chunk_count": 0,
            "traditional_backup": False,
            "langchain_enhanced": False,
            "file_size_mb": 0
        }
        
        # Check traditional backup
        if os.path.exists(traditional_path) and os.path.exists(chunks_path):
            info["exists"] = True
            info["traditional_backup"] = True
            info["file_size_mb"] = round(os.path.getsize(traditional_path) / 1024 / 1024, 2)
            
            # Quick chunk count
            try:
                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                    info["chunk_count"] = len(chunks)
            except:
                info["chunk_count"] = 0
        
        # Check LangChain format
        index_dir = os.path.dirname(FAISS_INDEX_PATH)
        langchain_files = [
            os.path.join(index_dir, "index.faiss"),
            os.path.join(index_dir, "index.pkl")
        ]
        
        if all(os.path.exists(f) for f in langchain_files):
            info["langchain_enhanced"] = True
        
        return info
        
    except Exception as e:
        return {"exists": False, "error": str(e)}

# LangChain compatibility functions
def similarity_search_with_score(query: str, k: int = 5) -> List[tuple]:
    """Perform similarity search with scores using LangChain"""
    try:
        if langchain_indexer.vector_store is None:
            langchain_indexer.load_vectorstore(FAISS_INDEX_PATH)
        
        if langchain_indexer.vector_store is None:
            raise Exception("LangChain vector store not available")
        
        # Perform similarity search with scores
        docs_with_scores = langchain_indexer.vector_store.similarity_search_with_score(query, k=k)
        
        return docs_with_scores
        
    except Exception as e:
        raise Exception(f"LangChain similarity search with scores failed: {str(e)}")

def similarity_search_with_langchain(query: str, k: int = 5) -> List[Document]:
    """Perform similarity search using LangChain vector store"""
    try:
        if langchain_indexer.vector_store is None:
            langchain_indexer.load_vectorstore(FAISS_INDEX_PATH)
        
        if langchain_indexer.vector_store is None:
            raise Exception("LangChain vector store not available")
        
        # Perform similarity search
        docs = langchain_indexer.vector_store.similarity_search(query, k=k)
        
        return docs
        
    except Exception as e:
        raise Exception(f"LangChain similarity search failed: {str(e)}")