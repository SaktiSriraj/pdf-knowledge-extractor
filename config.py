import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Data directories
PDF_DIR = str(DATA_DIR / "pdfs")
TEXT_DIR = str(DATA_DIR / "texts")
FAISS_INDEX_PATH = str(DATA_DIR / "faiss_index")
METADATA_PATH = str(DATA_DIR / "metadata.json")

# LangChain Configuration
LANGCHAIN_VECTORSTORE_PATH = str(DATA_DIR / "langchain_vectorstore")
LANGCHAIN_CACHE_DIR = str(DATA_DIR / "langchain_cache")

# Embedding model (local)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LangChain Text Splitting Configuration
LANGCHAIN_CHUNK_SIZE = int(os.getenv("LANGCHAIN_CHUNK_SIZE", "600"))
LANGCHAIN_CHUNK_OVERLAP = int(os.getenv("LANGCHAIN_CHUNK_OVERLAP", "100"))
LANGCHAIN_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Advanced LangChain Features
LANGCHAIN_ENABLE_CACHING = os.getenv("LANGCHAIN_ENABLE_CACHING", "true").lower() == "true"
LANGCHAIN_VERBOSE = os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true"
LANGCHAIN_DEBUG = os.getenv("LANGCHAIN_DEBUG", "false").lower() == "true"

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8501")
SITE_NAME = os.getenv("SITE_NAME", "Enhanced RAG PDF Chat")

# Processing parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
DEVICE = "cpu"

# LangChain Model Configuration
LANGCHAIN_MODEL_TEMPERATURE = float(os.getenv("LANGCHAIN_MODEL_TEMPERATURE", "0.7"))
LANGCHAIN_MAX_TOKENS = int(os.getenv("LANGCHAIN_MAX_TOKENS", "2000"))

# Debug: Print configuration status
if OPENROUTER_API_KEY:
   print(f"✅ OpenRouter API key loaded: {OPENROUTER_API_KEY[:8]}...")
else:
   print("⚠️ OpenRouter API key not found in environment variables")

if LANGCHAIN_VERBOSE:
   print("🔧 LangChain verbose mode enabled")
   print(f"📊 LangChain chunk size: {LANGCHAIN_CHUNK_SIZE}")
   print(f"🔄 LangChain chunk overlap: {LANGCHAIN_CHUNK_OVERLAP}")

# Create directories
for directory in [PDF_DIR, TEXT_DIR, str(DATA_DIR), LANGCHAIN_VECTORSTORE_PATH, LANGCHAIN_CACHE_DIR]:
   os.makedirs(directory, exist_ok=True)

# LangChain Environment Variables
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # Disable tracing for privacy
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""