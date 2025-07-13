import streamlit as st
import os
from rag_system import RAGSystem
from document_processor import DocumentProcessor
from vector_store import VectorStore
import logging
from demo_responses import get_demo_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def initialize_rag_system():
    """Initialize the RAG system with API keys or demo mode"""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        
        # Force demo mode for now to avoid API issues
        st.success("🎭 Demo Mode: Using sample responses - no API keys needed!")
        return "demo_mode"
            
        # Try to initialize components
        vector_store = VectorStore(pinecone_api_key)
        document_processor = DocumentProcessor()
        rag_system = RAGSystem(openai_api_key, vector_store, document_processor)
        
        return rag_system
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        st.warning("🎭 Demo Mode: API initialization failed - using sample responses")
        return "demo_mode"

def process_sample_documents(rag_system):
    """Process sample business documents"""
    if st.session_state.documents_processed:
        return
        
    try:
        if rag_system == "demo_mode":
            st.info("🎭 Demo Mode: Sample documents are pre-loaded")
            st.session_state.documents_processed = True
            st.success("✅ Demo documents ready!")
            return
            
        with st.spinner("Processing sample business documents..."):
            # Process sample documents
            sample_docs = [
                "sample_documents/business_policy.txt",
                "sample_documents/employee_handbook.txt", 
                "sample_documents/company_faq.txt"
            ]
            
            for doc_path in sample_docs:
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    rag_system.add_document(content, doc_path)
                    
            st.session_state.documents_processed = True
            st.success("Sample documents processed successfully!")
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        st.error(f"Error processing documents: {e}")

def main():
    st.set_page_config(
        page_title="Business RAG QA Bot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for colorful styling
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 25%, #3498db 50%, #2980b9 75%, #1abc9c 100%);
        background-attachment: fixed;
    }
    .main > div {
        background: rgba(255, 255, 255, 0.98);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        color: #333333;
    }
    .stMarkdown {
        color: #333333 !important;
    }
    .stText {
        color: #333333 !important;
    }
    .stChatMessage {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        color: #333333 !important;
    }
    .stChatMessage * {
        color: #333333 !important;
    }
    .main-header {
        background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    .demo-section {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .success-box {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    .info-box {
        background: linear-gradient(90deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
    }
    .stSidebar {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    .stSidebar > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    .stButton > button {
        background: linear-gradient(45deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        background: linear-gradient(45deg, #2980b9 0%, #1abc9c 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Business RAG QA Bot</h1>
        <p>✨ Ask questions about your business knowledge base using advanced RAG technology ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAG system
    if st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_rag_system()
    
    if st.session_state.rag_system is None:
        st.session_state.rag_system = "demo_mode"
    
    # Demo section with sample questions
    st.markdown("""
    <div class="demo-section">
        <h3>🚀 Try These Sample Questions!</h3>
        <p>Click any button below to see the RAG system in action:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample questions in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💼 What vacation policy?", use_container_width=True):
            st.session_state.demo_query = "What is the company's vacation policy?"
        if st.button("👔 Dress code rules?", use_container_width=True):
            st.session_state.demo_query = "What are the dress code requirements?"
    
    with col2:
        if st.button("🏥 Health benefits?", use_container_width=True):
            st.session_state.demo_query = "What health benefits does the company offer?"
        if st.button("🚨 Report injury?", use_container_width=True):
            st.session_state.demo_query = "How do I report a workplace injury?"
    
    with col3:
        if st.button("💰 Expense reports?", use_container_width=True):
            st.session_state.demo_query = "What is the process for expense reports?"
        if st.button("🎯 Apply for job?", use_container_width=True):
            st.session_state.demo_query = "How can I apply for a job at ACME Corporation?"

    # Sidebar for configuration and document management
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Document processing section
        st.markdown("### 📚 Document Management")
        
        if st.button("📂 Process Sample Documents", use_container_width=True):
            process_sample_documents(st.session_state.rag_system)
        
        # File upload section
        st.markdown("### 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "📁 Upload business documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx']
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}"):
                    try:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            content = uploaded_file.read()
                            if uploaded_file.type == "text/plain":
                                content = content.decode("utf-8")
                            else:
                                # For PDF and DOCX, we'll need to implement proper extraction
                                content = content.decode("utf-8", errors="ignore")
                            
                            st.session_state.rag_system.add_document(content, uploaded_file.name)
                            st.success(f"Successfully processed {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
        
        # RAG Configuration
        st.markdown("### 🎛️ RAG Settings")
        top_k = st.slider("📊 Number of retrieved documents", 1, 10, 5)
        temperature = st.slider("🌡️ Response temperature", 0.0, 1.0, 0.1)
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.markdown("## 💬 Chat with your Business Knowledge Base")
    
    # Process demo query if selected
    if 'demo_query' in st.session_state and st.session_state.demo_query:
        demo_query = st.session_state.demo_query
        st.session_state.demo_query = None
        
        # Add demo query to messages
        st.session_state.messages.append({"role": "user", "content": demo_query})
        
        # Process the demo query
        with st.spinner("🤔 Thinking..."):
            try:
                if st.session_state.rag_system == "demo_mode":
                    response = get_demo_response(demo_query)
                else:
                    response = st.session_state.rag_system.query(
                        demo_query, 
                        top_k=top_k, 
                        temperature=temperature
                    )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response["answer"],
                    "sources": response["sources"]
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>📄 Source {i+1}:</strong> {source['metadata']['source']}<br>
                            <strong>🎯 Relevance Score:</strong> {source['score']:.3f}<br>
                            <strong>📖 Content:</strong> {source['content'][:200]}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("💭 Ask a question about your business..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    if st.session_state.rag_system == "demo_mode":
                        response = get_demo_response(prompt)
                    else:
                        response = st.session_state.rag_system.query(
                            prompt, 
                            top_k=top_k, 
                            temperature=temperature
                        )
                    
                    st.markdown(response["answer"])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                    # Display sources
                    if response["sources"]:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response["sources"]):
                                st.markdown(f"""
                                <div class="info-box">
                                    <strong>📄 Source {i+1}:</strong> {source['metadata']['source']}<br>
                                    <strong>🎯 Relevance Score:</strong> {source['score']:.3f}<br>
                                    <strong>📖 Content:</strong> {source['content'][:200]}...
                                </div>
                                """, unsafe_allow_html=True)
                
                except Exception as e:
                    error_msg = f"❌ Error generating response: {e}"
                    logger.error(error_msg)
                    st.error(error_msg)
    
    # Performance metrics section
    with st.expander("📊 System Performance"):
        if st.session_state.rag_system:
            st.markdown("""
            <div class="success-box">
                <h4>🎯 RAG System Metrics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            if st.session_state.rag_system == "demo_mode":
                with col1:
                    st.metric("📚 Total Documents", 3)
                with col2:
                    st.metric("🧩 Total Chunks", 15)
                with col3:
                    st.metric("❓ Queries Processed", len(st.session_state.messages) // 2)
            else:
                metrics = st.session_state.rag_system.get_metrics()
                with col1:
                    st.metric("📚 Total Documents", metrics.get('total_documents', 0))
                with col2:
                    st.metric("🧩 Total Chunks", metrics.get('total_chunks', 0))
                with col3:
                    st.metric("❓ Queries Processed", metrics.get('queries_processed', 0))

if __name__ == "__main__":
    main()
