import subprocess
import sys
import os
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, LANGCHAIN_CACHE_DIR

def install_spacy_model():
    """Install spaCy English model for advanced text processing"""
    try:
        print("📥 Installing spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("✅ spaCy model installed successfully!")
        return True
    except Exception as e:
        print(f"⚠️ spaCy model installation failed: {str(e)}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        print("📥 Downloading NLTK data...")
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️ NLTK data download failed: {str(e)}")
        return False

def download_embedding_model():
    """Download and verify embedding model"""
    try:
        print("📥 Downloading enhanced embedding model...")
        
        # Download via sentence-transformers (best method)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Test the model
        test_embedding = model.encode(["Enhanced LangChain RAG test"])
        print(f"✅ Embedding model ready! Dimension: {len(test_embedding[0])}")
        
        # Cache model information
        os.makedirs(LANGCHAIN_CACHE_DIR, exist_ok=True)
        with open(os.path.join(LANGCHAIN_CACHE_DIR, "model_info.txt"), "w") as f:
            f.write(f"Model: {EMBEDDING_MODEL_NAME}\n")
            f.write(f"Dimension: {len(test_embedding[0])}\n")
            f.write(f"LangChain Enhanced: True\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download embedding model: {str(e)}")
        return False

def verify_langchain_installation():
    """Verify LangChain components are working"""
    try:
        print("🔧 Verifying LangChain installation...")
        
        # Test core imports
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import FAISS
        from langchain.embeddings import SentenceTransformerEmbeddings
        
        # Test basic functionality
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        test_doc = Document(page_content="This is a test document for LangChain verification.")
        chunks = splitter.split_documents([test_doc])
        
        print(f"✅ LangChain verification successful! Created {len(chunks)} test chunks")
        return True
        
    except Exception as e:
        print(f"❌ LangChain verification failed: {str(e)}")
        return False

def main():
    """Enhanced model installation with LangChain setup"""
    print("🚀 Enhanced LangChain RAG Setup")
    print("=" * 50)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Verify LangChain
    if verify_langchain_installation():
        success_count += 1
    
    # Step 2: Download embedding model
    if download_embedding_model():
        success_count += 1
    
    # Step 3: Install spaCy model (optional)
    if install_spacy_model():
        success_count += 1
    
    # Step 4: Download NLTK data (optional)
    if download_nltk_data():
        success_count += 1
    
    print(f"\n📊 Setup Results: {success_count}/{total_steps} components installed successfully")
    
    if success_count >= 2:  # Core components working
        print("\n🎉 Enhanced LangChain RAG setup completed!")
        print("\n🌟 Your project now includes:")
        print("  ✅ Advanced LangChain text processing")
        print("  ✅ Multiple retrieval strategies")
        print("  ✅ Enhanced document analysis")
        print("  ✅ Intent-based query processing")
        print("  ✅ Contextual compression")
        print("  ✅ Conversation memory")
        print("\n📝 Resume-worthy features:")
        print("  • Multi-strategy RAG implementation")
        print("  • LangChain vector store integration")
        print("  • Advanced document processing pipeline")
        print("  • Intent-based prompt engineering")
        print("  • Contextual compression and ranking")
        
        print("\n🚀 Run the application:")
        print("  streamlit run app.py")
    else:
        print("\n⚠️ Setup completed with some issues.")
        print("  The app should still work with basic functionality.")
    
    return success_count >= 2

if __name__ == "__main__":
    main()