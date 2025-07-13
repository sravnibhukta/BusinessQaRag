# RAG System for Business Documents

A comprehensive Retrieval-Augmented Generation (RAG) system built with Python and Streamlit for intelligent business document question-answering.

<img width="1906" height="853" alt="image" src="https://github.com/user-attachments/assets/9d758956-1363-46b8-8374-2f316e47360a" />

<img width="1915" height="862" alt="image" src="https://github.com/user-attachments/assets/0f56269d-d00d-4127-afaa-cc6d388d7116" />

<img width="1916" height="863" alt="image" src="https://github.com/user-attachments/assets/2ca23f0d-a5fc-4b45-be12-b459f6e67c98" />




## Overview

This project implements a complete RAG pipeline that processes business documents, creates embeddings, stores them in a vector database, and provides contextually relevant answers to user queries using OpenAI's language models. The system features a beautiful web interface with demo capabilities.

## Features

- **Interactive Web Interface**: Beautiful Streamlit app with dark gradient theme
- **Document Processing**: Intelligent text chunking and preprocessing
- **Vector Search**: Semantic search using Pinecone vector database
- **AI-Powered Responses**: OpenAI GPT-4 integration for answer generation
- **Demo Mode**: Works without API keys using realistic sample responses
- **Source Attribution**: Responses include citations and source references
- **Performance Metrics**: Real-time system performance monitoring

## Project Structure

```
├── app.py                          # Main Streamlit application
├── rag_system.py                   # Core RAG system implementation
├── document_processor.py           # Document processing and chunking
├── vector_store.py                 # Pinecone vector database interface
├── demo_responses.py               # Demo mode responses
├── utils.py                        # Utility functions and logging
├── convert_to_pdf.py               # PDF conversion utility
├── sample_documents/               # Sample business documents
│   ├── business_policy.txt
│   ├── company_faq.txt
│   └── employee_handbook.txt
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── Task1_RAG_QA_Bot.ipynb         # Google Colab notebook
├── Task2_RAG_Optimization_Techniques.pdf
├── Task3_Dataset_Preparation_and_Fine_Tuning.pdf
└── README.md                       # This file
```

## Installation

1. **Clone the repository** (or download the files)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Required packages:
   - streamlit
   - openai
   - pinecone-client
   - nltk
   - numpy
   - markdown
   - weasyprint
   - html2text

3. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Configuration

### Environment Variables

For full functionality, set these environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export PINECONE_API_KEY="your-pinecone-api-key"
```

### Demo Mode

The system works in demo mode without API keys, using pre-built responses for common business queries.

## Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py --server.port 5000
   ```

2. **Access the interface**:
   - Open your browser to `http://localhost:5000`
   - Upload business documents or use sample documents
   - Ask questions using the chat interface or sample buttons

### Sample Questions

The system includes pre-configured sample questions:
- "What is our vacation policy?"
- "How do health benefits work?"
- "What are the remote work guidelines?"
- "How do I submit an expense report?"
- "What is the company mission?"
- "How do performance reviews work?"

### Google Colab Notebook

Run the complete RAG system in Google Colab:
1. Open `Task1_RAG_QA_Bot.ipynb`
2. Follow the step-by-step implementation
3. Test with your own documents and queries

## Technical Architecture

### Core Components

1. **RAG System (`rag_system.py`)**
   - Orchestrates the entire pipeline
   - Handles embedding generation and query processing
   - Manages context retrieval and response generation

2. **Document Processor (`document_processor.py`)**
   - Text cleaning and normalization
   - Intelligent chunking with overlap
   - Metadata extraction and preservation

3. **Vector Store (`vector_store.py`)**
   - Pinecone integration for vector storage
   - Semantic search capabilities
   - Batch operations for efficiency

4. **Utilities (`utils.py`)**
   - Logging and performance monitoring
   - Configuration management
   - Error handling and validation

### Data Flow

1. **Document Upload** → Text extraction and cleaning
2. **Text Processing** → Chunking and normalization
3. **Embedding Generation** → Vector representation using OpenAI
4. **Vector Storage** → Pinecone database storage
5. **Query Processing** → Semantic search and context retrieval
6. **Response Generation** → AI-powered answer with citations

## Demo Mode Features

When API keys are not available, the system provides:
- Realistic business responses to common queries
- Source attribution with sample documents
- Performance metrics simulation
- Full interface functionality

## Development

### Adding New Documents

1. Place text files in the `sample_documents/` directory
2. The system will automatically process them on startup
3. Documents are chunked and embedded for search

### Customizing Responses

Edit `demo_responses.py` to add new demo responses:
```python
DEMO_RESPONSES = {
    "your question": {
        "answer": "Your detailed answer...",
        "sources": ["Document Name", "Section"]
    }
}
```

### Extending Functionality

- **New Document Types**: Extend `document_processor.py`
- **Custom Embeddings**: Modify `rag_system.py`
- **UI Improvements**: Update `app.py` styling
- **New Metrics**: Add to `utils.py`

## Performance Optimization

The system includes two advanced optimization techniques:

1. **Hybrid Retrieval**: Multi-vector search for improved accuracy
2. **Adaptive Context**: Dynamic context sizing based on query complexity

See `Task2_RAG_Optimization_Techniques.pdf` for detailed implementation.

## Fine-Tuning and Dataset Preparation

For model fine-tuning guidance, see `Task3_Dataset_Preparation_and_Fine_Tuning.pdf` which covers:
- Data collection strategies
- Quality assurance techniques
- Fine-tuning approaches comparison
- Implementation best practices

## API Reference

### RAG System Methods

```python
# Initialize system
rag_system = RAGSystem(openai_api_key, vector_store, document_processor)

# Add documents
rag_system.add_document(content, source)

# Query system
response = rag_system.query(query, top_k=5)
```

### Document Processing

```python
# Process document
processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
chunks = processor.process_document(content, source)
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are set correctly
2. **Pinecone Connection**: Check API key and internet connection
3. **NLTK Data**: Download required NLTK packages
4. **Port Conflicts**: Use different port with `--server.port`

### Demo Mode Fallback

If APIs fail, the system automatically switches to demo mode with:
- Pre-built responses for common queries
- Simulated performance metrics
- Full interface functionality

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure demo mode compatibility

## License

This project is created for educational and demonstration purposes.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the technical documentation
3. Test in demo mode first
4. Verify API key configuration

---

**Note**: This RAG system demonstrates enterprise-grade document processing and question-answering capabilities suitable for business applications. The modular architecture allows for easy extension and customization based on specific requirements.
