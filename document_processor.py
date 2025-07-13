import re
import logging
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Document processor for preparing business documents for RAG pipeline
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([\.!?]){2,}', r'\1', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except Exception as e:
            logger.error(f"Error extracting sentences: {e}")
            # Fallback to simple splitting
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def create_chunks(self, text: str, source: str) -> List[Dict[str, Any]]:
        """
        Create overlapping text chunks for better context preservation
        
        Args:
            text: Input text to chunk
            source: Source document name
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        text_length = len(text)
        
        # If text is shorter than chunk size, return as single chunk
        if text_length <= self.chunk_size:
            return [{
                'content': text,
                'source': source,
                'start_char': 0,
                'end_char': text_length
            }]
        
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # Try to break at sentence boundary if possible
            if end < text_length:
                # Look for sentence ending within the last 200 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size - 200:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'source': source,
                    'start_char': start,
                    'end_char': end,
                    'chunk_index': chunk_index
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we don't go backwards
            if start <= 0:
                start = end
        
        return chunks
    
    def preprocess_for_embedding(self, text: str) -> str:
        """
        Preprocess text specifically for embedding generation
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text optimized for embeddings
        """
        # Clean text
        text = self.clean_text(text)
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation, lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and token not in string.punctuation:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        # Join back to text
        processed_text = ' '.join(processed_tokens)
        
        return processed_text
    
    def extract_metadata(self, text: str, source: str) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            text: Document text
            source: Document source
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'source': source,
            'length': len(text),
            'word_count': len(word_tokenize(text)),
            'sentence_count': len(self.extract_sentences(text))
        }
        
        # Extract potential document type based on content patterns
        if re.search(r'(policy|procedure|guideline)', text.lower()):
            metadata['document_type'] = 'policy'
        elif re.search(r'(faq|question|answer)', text.lower()):
            metadata['document_type'] = 'faq'
        elif re.search(r'(manual|handbook|guide)', text.lower()):
            metadata['document_type'] = 'manual'
        else:
            metadata['document_type'] = 'general'
        
        return metadata
    
    def process_document(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Main document processing pipeline
        
        Args:
            content: Document content
            source: Document source/filename
            
        Returns:
            List of processed chunks ready for embedding
        """
        try:
            # Clean the text
            cleaned_content = self.clean_text(content)
            
            if not cleaned_content.strip():
                logger.warning(f"Empty content after cleaning for document: {source}")
                return []
            
            # Extract document metadata
            metadata = self.extract_metadata(cleaned_content, source)
            
            # Create chunks
            chunks = self.create_chunks(cleaned_content, source)
            
            # Process each chunk for embedding
            processed_chunks = []
            for chunk in chunks:
                # Keep original content for context
                chunk['original_content'] = chunk['content']
                
                # Add preprocessed version for better embedding
                chunk['preprocessed_content'] = self.preprocess_for_embedding(chunk['content'])
                
                # Add document metadata
                chunk['document_metadata'] = metadata
                
                processed_chunks.append(chunk)
            
            logger.info(f"Processed document {source} into {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {source}: {e}")
            return []
    
    def get_processing_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed chunks
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            Dictionary of processing statistics
        """
        if not chunks:
            return {'total_chunks': 0}
        
        total_chars = sum(len(chunk['content']) for chunk in chunks)
        total_words = sum(len(word_tokenize(chunk['content'])) for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_words': total_words,
            'average_chunk_size': total_chars / len(chunks),
            'average_words_per_chunk': total_words / len(chunks)
        }
