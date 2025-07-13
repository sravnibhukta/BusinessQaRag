import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGLogger:
    """
    Custom logger for RAG system operations
    """
    
    def __init__(self, log_file: str = "rag_system.log"):
        self.log_file = log_file
        self.setup_logger()
    
    def setup_logger(self):
        """Setup custom logger with file handler"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Get logger
        self.logger = logging.getLogger('RAGSystem')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_query(self, query: str, response: str, sources: List[Dict], processing_time: float):
        """Log query and response details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_length': len(response),
            'sources_count': len(sources),
            'processing_time': processing_time,
            'sources': [{'source': s['source'], 'score': s['score']} for s in sources]
        }
        
        self.logger.info(f"Query processed: {json.dumps(log_entry, indent=2)}")
    
    def log_document_processing(self, source: str, chunks_created: int, processing_time: float):
        """Log document processing details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'chunks_created': chunks_created,
            'processing_time': processing_time
        }
        
        self.logger.info(f"Document processed: {json.dumps(log_entry, indent=2)}")

class ConfigManager:
    """
    Configuration management for RAG system
    """
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults"""
        default_config = {
            'embedding_model': 'text-embedding-ada-002',
            'chat_model': 'gpt-4o',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_retrieved_chunks': 5,
            'temperature': 0.1,
            'max_tokens': 1000,
            'pinecone_environment': 'us-east-1',
            'pinecone_index_name': 'business-rag-index',
            'vector_dimension': 1536
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
                    logger.info("Configuration loaded from file")
            else:
                logger.info("Using default configuration")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values"""
        self.config.update(updates)

class PerformanceMonitor:
    """
    Performance monitoring for RAG system
    """
    
    def __init__(self):
        self.metrics = {
            'queries_processed': 0,
            'total_query_time': 0,
            'average_query_time': 0,
            'documents_processed': 0,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'embedding_generations': 0,
            'vector_store_operations': 0
        }
        self.query_history = []
    
    def record_query(self, query: str, processing_time: float, sources_count: int):
        """Record query performance metrics"""
        self.metrics['queries_processed'] += 1
        self.metrics['total_query_time'] += processing_time
        self.metrics['average_query_time'] = (
            self.metrics['total_query_time'] / self.metrics['queries_processed']
        )
        
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query[:100] + '...' if len(query) > 100 else query,
            'processing_time': processing_time,
            'sources_count': sources_count
        })
    
    def record_document_processing(self, processing_time: float):
        """Record document processing performance"""
        self.metrics['documents_processed'] += 1
        self.metrics['total_processing_time'] += processing_time
        self.metrics['average_processing_time'] = (
            self.metrics['total_processing_time'] / self.metrics['documents_processed']
        )
    
    def record_embedding_generation(self):
        """Record embedding generation event"""
        self.metrics['embedding_generations'] += 1
    
    def record_vector_operation(self):
        """Record vector store operation"""
        self.metrics['vector_store_operations'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history"""
        return self.query_history[-limit:]

class TextValidator:
    """
    Text validation utilities for RAG system
    """
    
    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """
        Validate user query
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        if len(query) > 1000:
            return False, "Query is too long (max 1000 characters)"
        
        if len(query.strip()) < 3:
            return False, "Query is too short (minimum 3 characters)"
        
        return True, ""
    
    @staticmethod
    def validate_document_content(content: str) -> tuple[bool, str]:
        """
        Validate document content
        
        Args:
            content: Document content string
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content or not content.strip():
            return False, "Document content cannot be empty"
        
        if len(content) > 1000000:  # 1MB limit
            return False, "Document is too large (max 1MB)"
        
        if len(content.strip()) < 10:
            return False, "Document content is too short"
        
        return True, ""

class ErrorHandler:
    """
    Centralized error handling for RAG system
    """
    
    @staticmethod
    def handle_openai_error(error: Exception) -> str:
        """Handle OpenAI API errors"""
        error_msg = str(error)
        
        if "rate_limit_exceeded" in error_msg:
            return "Rate limit exceeded. Please wait a moment and try again."
        elif "invalid_api_key" in error_msg:
            return "Invalid API key. Please check your OpenAI API key."
        elif "insufficient_quota" in error_msg:
            return "Insufficient quota. Please check your OpenAI account."
        else:
            return f"OpenAI API error: {error_msg}"
    
    @staticmethod
    def handle_pinecone_error(error: Exception) -> str:
        """Handle Pinecone API errors"""
        error_msg = str(error)
        
        if "Unauthorized" in error_msg:
            return "Unauthorized access to Pinecone. Please check your API key."
        elif "Index not found" in error_msg:
            return "Pinecone index not found. Please check your index configuration."
        elif "quota" in error_msg.lower():
            return "Pinecone quota exceeded. Please check your usage limits."
        else:
            return f"Pinecone error: {error_msg}"
    
    @staticmethod
    def handle_general_error(error: Exception) -> str:
        """Handle general errors"""
        return f"An error occurred: {str(error)}"

def format_response_with_sources(answer: str, sources: List[Dict[str, Any]]) -> str:
    """
    Format response with source citations
    
    Args:
        answer: Generated answer
        sources: List of source documents
        
    Returns:
        Formatted response with citations
    """
    if not sources:
        return answer
    
    formatted_response = answer + "\n\n**Sources:**\n"
    
    for i, source in enumerate(sources, 1):
        source_name = source.get('source', 'Unknown')
        score = source.get('score', 0)
        formatted_response += f"{i}. {source_name} (relevance: {score:.3f})\n"
    
    return formatted_response

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Truncate if too long
    if len(filename) > 255:
        filename = filename[:255]
    
    return filename

def calculate_similarity_score(score: float) -> str:
    """
    Convert similarity score to human-readable format
    
    Args:
        score: Similarity score (0-1)
        
    Returns:
        Human-readable similarity description
    """
    if score >= 0.9:
        return "Excellent match"
    elif score >= 0.8:
        return "Very good match"
    elif score >= 0.7:
        return "Good match"
    elif score >= 0.6:
        return "Fair match"
    else:
        return "Poor match"
