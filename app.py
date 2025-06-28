import streamlit as st
import os
import time
import threading
from scraper import process_pdf_url
from upload_handler import handle_uploaded_pdf
from preprocess import load_and_split_texts, get_processing_stats
from indexer import build_faiss_index, load_faiss_index, get_index_info
from rag_pipeline import run_rag_pipeline
from model_loader import llm_manager, test_openrouter_connection
from config import OPENROUTER_API_KEY, OPENROUTER_MODEL

# Page configuration
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling and compact chat
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stTitle {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .chat-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
    }
    
    .processing-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
    
    /* Hide Streamlit notifications and blue boxes */
    .stAlert, .stNotification, .stInfo, div[data-testid="stNotificationContentInfo"] {
        display: none !important;
    }
    
    .stContainer:empty {
        display: none !important;
    }
    
    /* File uploader customization */
    .stFileUploader {
        background: transparent !important;
    }
    
    .stFileUploader > div {
        background: rgba(255,255,255,0.1) !important;
        border: 2px dashed rgba(255,255,255,0.3) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    .stFileUploader > div > div {
        background: transparent !important;
        color: white !important;
    }
    
    .stFileUploader button {
        background: rgba(255,255,255,0.2) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
    }
    
    /* Loading animations */
    .loading-dots {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 20px;
    }
    
    .loading-dots div {
        position: absolute;
        top: 8px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #fff;
        animation: loading-dots 1.2s linear infinite;
    }
    
    .loading-dots div:nth-child(1) { left: 8px; animation-delay: 0s; }
    .loading-dots div:nth-child(2) { left: 32px; animation-delay: -0.4s; }
    .loading-dots div:nth-child(3) { left: 56px; animation-delay: -0.8s; }
    
    @keyframes loading-dots {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    .spinner {
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 50%;
        border-top: 3px solid white;
        width: 20px;
        height: 20px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 10px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .progress-text {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .step-indicator {
        background: rgba(255,255,255,0.2);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid white;
    }
    
    .completed-step {
        background: rgba(40, 167, 69, 0.3);
        border-left-color: #28a745;
    }
    
    .current-step {
        background: rgba(255, 193, 7, 0.3);
        border-left-color: #ffc107;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Compact chat styles */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem 0;
    }
    
    .source-box {
        background: #fff3e0; 
        padding: 0.6rem; 
        border-radius: 8px; 
        margin: 0.3rem 0;
        border-left: 3px solid #ff9800;
        font-size: 0.85rem;
    }
    
    /* Responsive chat bubbles */
    @media (max-width: 768px) {
        .chat-bubble {
            max-width: 90% !important;
            margin-left: 5% !important;
            margin-right: 5% !important;
        }
    }
    
    /* Smooth animations for new messages */
    .new-message {
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'index_ready' not in st.session_state:
        st.session_state.index_ready = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_tested' not in st.session_state:
        st.session_state.api_tested = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0

class ProgressTracker:
    """Real-time progress tracking class"""
    def __init__(self, progress_bar, status_container, steps_container):
        self.progress_bar = progress_bar
        self.status_container = status_container
        self.steps_container = steps_container
        self.current_step = 0
        self.total_steps = 6
        self.step_names = [
            "📥 Processing Documents",
            "📄 Extracting Text Content", 
            "✂️ Splitting into Sections",
            "🧠 Generating Embeddings",
            "🔍 Building Search Index",
            "✅ Finalizing Setup"
        ]
        
    def update_step(self, step_index, status="current", substep=""):
        """Update current step with animation"""
        self.current_step = step_index
        progress = (step_index / self.total_steps) * 100
        
        # Update progress bar
        self.progress_bar.progress(int(progress))
        
        # Update main status
        if status == "current":
            if substep:
                status_html = f"""
                <div class="progress-text">
                    <div class="spinner"></div>
                    {self.step_names[step_index]} - {substep}
                </div>
                """
            else:
                status_html = f"""
                <div class="progress-text">
                    <div class="spinner"></div>
                    {self.step_names[step_index]}
                    <div class="loading-dots">
                        <div></div><div></div><div></div>
                    </div>
                </div>
                """
        else:
            status_html = f"""
            <div class="progress-text">
                ✅ {self.step_names[step_index]} - Complete
            </div>
            """
        
        self.status_container.markdown(status_html, unsafe_allow_html=True)
        
        # Update steps overview
        self.update_steps_overview()
        
    def update_steps_overview(self):
        """Update the steps overview with current progress"""
        steps_html = ""
        for i, step_name in enumerate(self.step_names):
            if i < self.current_step:
                css_class = "step-indicator completed-step"
                icon = "✅"
            elif i == self.current_step:
                css_class = "step-indicator current-step"
                icon = "🔄"
            else:
                css_class = "step-indicator"
                icon = "⏳"
            
            steps_html += f"""
            <div class="{css_class}">
                {icon} {step_name}
            </div>
            """
        
        self.steps_container.markdown(steps_html, unsafe_allow_html=True)
    
    def complete(self):
        """Mark processing as complete"""
        self.progress_bar.progress(100)
        self.status_container.success("🎉 Processing Complete!")
        
        # Update all steps as completed
        steps_html = ""
        for step_name in self.step_names:
            steps_html += f"""
            <div class="step-indicator completed-step">
                ✅ {step_name}
            </div>
            """
        self.steps_container.markdown(steps_html, unsafe_allow_html=True)

def check_api_status():
    """Check API status and return status info"""
    if not OPENROUTER_API_KEY:
        return False, "API key not configured"
    
    if not st.session_state.api_tested:
        try:
            success, message = test_openrouter_connection()
            st.session_state.api_tested = True
            return success, message
        except:
            return False, "Connection failed"
    return True, "Connected"

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="stTitle">📚 PDF Chat AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload your documents and have intelligent conversations with them</p>', unsafe_allow_html=True)

def display_status_bar():
    """Display a clean status bar"""
    col1, col2, col3, col4 = st.columns(4)
    
    # API Status
    with col1:
        api_success, api_msg = check_api_status()
        status_color = "🟢" if api_success else "🔴"
        st.markdown(f"**{status_color} AI Model**")
        st.caption("DeepSeek R1" if api_success else "Disconnected")
    
    # Document Status
    with col2:
        stats = get_processing_stats()
        files_count = stats.get("files", 0)
        st.markdown(f"**📄 Documents**")
        st.caption(f"{files_count} processed")
    
    # Index Status
    with col3:
        index_info = get_index_info()
        chunks = index_info.get("chunk_count", 0) if index_info.get("exists") else 0
        st.markdown(f"**🔍 Search Index**")
        st.caption(f"{chunks} sections" if chunks > 0 else "Not ready")
    
    # Chat Status
    with col4:
        chat_count = len(st.session_state.chat_history)
        st.markdown(f"**💬 Conversations**")
        st.caption(f"{chat_count} messages")

def display_upload_section():
    """Display the document upload section"""
    st.markdown("## 📁 Add Your Documents")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📎 Upload Files", "🔗 From URL"])
    
    with tab1:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to chat with",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) selected")
            for i, file in enumerate(uploaded_files):
                file_size = len(file.getvalue()) / 1024 / 1024  # Size in MB
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.2); 
                    padding: 0.6rem; 
                    border-radius: 8px; 
                    margin: 0.3rem 0;
                    color: white;
                    font-size: 0.9rem;
                ">
                    📄 {file.name} ({file_size:.1f} MB)
                </div>
                """, unsafe_allow_html=True)
            return "upload", uploaded_files
    
    with tab2:
        pdf_url = st.text_input(
            "PDF URL",
            placeholder="https://example.com/document.pdf",
            help="Enter a direct link to a PDF file",
            label_visibility="collapsed"
        )
        
        if pdf_url:
            if pdf_url.startswith("http") and pdf_url.endswith(".pdf"):
                st.success("✅ Valid PDF URL")
                return "url", pdf_url
            else:
                st.warning("⚠️ Please enter a valid PDF URL")
    
    return None, None

def process_documents_with_realtime_progress(doc_type, doc_data):
    """Process documents with real-time progress tracking"""
    
    # Create processing container
    with st.container():
        st.markdown("### 🚀 Processing Your Documents")
        
        # Create progress elements
        progress_bar = st.progress(0)
        status_container = st.empty()
        steps_container = st.empty()
        
        # Initialize progress tracker
        tracker = ProgressTracker(progress_bar, status_container, steps_container)
        
        try:
            processed_count = 0
            total_words = 0
            
            # Step 1: Download/Upload Documents
            tracker.update_step(0, "current")
            
            if doc_type == "url":
                tracker.update_step(0, "current", "Downloading from URL...")
                time.sleep(0.5)
                text_path, word_count = process_pdf_url(doc_data)
                processed_count = 1
                total_words = word_count
                
            elif doc_type == "upload":
                for i, uploaded_file in enumerate(doc_data):
                    tracker.update_step(0, "current", f"Processing {uploaded_file.name}...")
                    time.sleep(0.3)
                    text_path, word_count = handle_uploaded_pdf(uploaded_file)
                    processed_count += 1
                    total_words += word_count
            
            tracker.update_step(0, "complete")
            time.sleep(0.5)
            
            # Step 2: Extract Text
            tracker.update_step(1, "current")
            time.sleep(0.8)
            tracker.update_step(1, "complete")
            
            # Step 3: Split Text
            tracker.update_step(2, "current", "Analyzing document structure...")
            time.sleep(0.5)
            
            chunks, metadata = load_and_split_texts()
            if not chunks:
                st.error("❌ No text could be extracted from the documents.")
                return False
            
            tracker.update_step(2, "complete")
            time.sleep(0.5)
            
            # Step 4: Generate Embeddings
            tracker.update_step(3, "current", f"Processing {len(chunks)} sections...")
            time.sleep(1.0)
            
            # Step 5: Build Index
            tracker.update_step(4, "current", "Creating search database...")
            
            # Simulate real-time progress during index building
            def build_index_with_progress():
                substep_messages = [
                    "Initializing embedding model...",
                    "Converting text to vectors...", 
                    "Optimizing search structure...",
                    "Building search index...",
                    "Saving index to disk..."
                ]
                
                for i, message in enumerate(substep_messages):
                    tracker.update_step(4, "current", message)
                    time.sleep(0.6)
                
                return build_faiss_index(chunks, metadata)
            
            index, _ = build_index_with_progress()
            tracker.update_step(4, "complete")
            time.sleep(0.5)
            
            # Step 6: Finalize
            tracker.update_step(5, "current", "Preparing for chat...")
            time.sleep(0.8)
            tracker.update_step(5, "complete")
            
            # Complete the process
            tracker.complete()
            
            # Success message
            st.balloons()
            time.sleep(1)
            
            # Final success message
            success_html = f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        color: white; padding: 1.5rem; border-radius: 10px; 
                        text-align: center; margin: 1rem 0;">
                <h3>🎉 Success!</h3>
                <p>Processed <strong>{processed_count}</strong> document(s)</p>
                <p>Created <strong>{len(chunks)}</strong> searchable sections</p>
                <p>Total words: <strong>{total_words:,}</strong></p>
                <p style="margin-top: 1rem; opacity: 0.9;">Ready to chat! 💬</p>
            </div>
            """
            st.markdown(success_html, unsafe_allow_html=True)
            
            st.session_state.index_ready = True
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            return False
    
    return True

def display_chat_section():
    """Display the chat interface - Fixed session state issue"""
    st.markdown("## 💬 Chat with Your Documents")
    
    # Check if ready to chat
    if not st.session_state.index_ready:
        index_info = get_index_info()
        if index_info.get("exists"):
            st.session_state.index_ready = True
        else:
            st.info("👆 Please upload some documents first to start chatting")
            return
    
    # Chat input with unique key and proper handling
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Ask anything about your documents...",
            placeholder="What are the main topics discussed?",
            label_visibility="collapsed",
            key=f"chat_input_{st.session_state.input_counter}"  # Dynamic key
        )
    
    with col2:
        ask_button = st.button("Send", type="primary", use_container_width=True)
        
    # Process query
    if ask_button and query:
        process_chat_query_with_animation(query)
    
    # Display chat history
    display_chat_history()

def process_chat_query_with_animation(query):
    """Process chat query with animated loading - Fixed version"""
    try:
        # Create animated loading container
        loading_container = st.empty()
        
        # Animated thinking process
        thinking_steps = [
            "🔍 Searching your documents...",
            "🧠 Understanding your question...", 
            "📝 Preparing detailed response...",
            "✨ Almost ready..."
        ]
        
        for i, step in enumerate(thinking_steps):
            loading_html = f"""
            <div style="
                background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%); 
                color: white; 
                padding: 1rem; 
                border-radius: 8px; 
                text-align: center; 
                margin: 0.5rem 0;
            ">
                <div class="spinner"></div>
                {step}
            </div>
            """
            loading_container.markdown(loading_html, unsafe_allow_html=True)
            time.sleep(0.8)
        
        # Actually process the query
        index, chunks, metadata = load_faiss_index()
        response, chunk_info = run_rag_pipeline(query, index, chunks, metadata, max_tokens=2000)
        
        # Clear loading animation
        loading_container.empty()
        
        # Add to history
        st.session_state.chat_history.append({
            "question": query,
            "answer": response,
            "sources": chunk_info
        })
        
        # Clear input by incrementing counter (this forces a new widget)
        st.session_state.input_counter += 1
        
        # Rerun to show new message
        st.rerun()
        
    except Exception as e:
        loading_container.empty()
        st.error(f"❌ Something went wrong: {str(e)}")

def display_chat_history():
    """Display chat history with compact, visible chat bubbles"""
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; color: #666;'>
            <h3>🤖 Ready to help!</h3>
            <p>Ask me anything about your uploaded documents.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display recent conversations (last 5)
    recent_chats = list(reversed(st.session_state.chat_history[-5:]))
    
    for i, chat in enumerate(recent_chats):
        # User Message - Compact Bubble
        st.markdown(f"""
        <div style="
            background: #e3f2fd; 
            padding: 0.8rem 1rem; 
            border-radius: 18px 18px 4px 18px; 
            margin: 0.5rem 0 0.5rem 20%;
            border-left: 3px solid #2196f3;
            max-width: 75%;
            margin-left: auto;
            margin-right: 0;
        ">
            <div style="font-size: 0.9rem; color: #1976d2; font-weight: 500; margin-bottom: 0.3rem;">
                🙋 You
            </div>
            <div style="color: #333; line-height: 1.4;">
                {chat['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # AI Response - Compact Bubble
        st.markdown(f"""
        <div style="
            background: #f1f8e9; 
            padding: 0.8rem 1rem; 
            border-radius: 18px 18px 18px 4px; 
            margin: 0.5rem 20% 0.5rem 0;
            border-left: 3px solid #4caf50;
            max-width: 75%;
        ">
            <div style="font-size: 0.9rem; color: #388e3c; font-weight: 500; margin-bottom: 0.3rem;">
                🤖 AI Assistant
            </div>
            <div style="color: #333; line-height: 1.5;">
                {chat['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sources (if available) - Compact
        if chat.get('sources'):
            with st.expander(f"📚 Sources ({len(chat['sources'])})", expanded=False):
                for j, source in enumerate(chat['sources'][:3]):  # Show top 3 sources
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {j+1}</strong><br>
                        <div style="color: #666; margin-top: 0.2rem;">
                            {source.get('preview', 'No preview available')[:150]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add small separator
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

def display_sidebar():
    """Clean sidebar with essential controls"""
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # Reset all data
        if st.button("🔄 Reset All Data", use_container_width=True):
            clear_all_data()
            st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        stats = get_processing_stats()
        index_info = get_index_info()
        
        st.metric("Documents", stats.get("files", 0))
        st.metric("Words Processed", f"{stats.get('total_words', 0):,}")
        st.metric("Searchable Sections", index_info.get("chunk_count", 0) if index_info.get("exists") else 0)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.caption("This app uses AI to help you understand and analyze your PDF documents through natural conversation.")

def clear_all_data():
    """Clear all processed data"""
    try:
        import shutil
        from config import DATA_DIR
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)
        st.session_state.index_ready = False
        st.session_state.chat_history = []
        st.success("✅ All data cleared!")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def check_setup():
    """Check if the app is properly set up"""
    if not OPENROUTER_API_KEY:
        st.error("""
        ### 🔑 Setup Required
        
        You need an API key to use this app:
        
        1. **Get a free key** at [openrouter.ai](https://openrouter.ai)
        2. **Create a `.env` file** in your project folder
        3. **Add your key**: `OPENROUTER_API_KEY=your_key_here`
        4. **Restart the app**
        
        The AI model is completely free to use!
        """)
        return False
    return True

def main():
    """Main application function"""
    initialize_session_state()
    
    # Check setup
    if not check_setup():
        return
    
    # Header
    display_header()
    
    # Status bar
    display_status_bar()
    
    st.markdown("---")
    
    # Main layout
    col1, col2 = st.columns([1, 4], gap="large")
    
    with col1:
        # Upload section
        doc_type, doc_data = display_upload_section()
        
        if doc_type and doc_data:
            if st.button("🚀 Process Documents", type="primary", use_container_width=True):
                process_documents_with_realtime_progress(doc_type, doc_data)
    
    with col2:
        # Chat section
        display_chat_section()
    
    # Sidebar
    display_sidebar()

if __name__ == "__main__":
    main()