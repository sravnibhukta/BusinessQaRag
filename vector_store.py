import os
import logging
from typing import List, Dict, Any, Optional
import pinecone
from pinecone import Pinecone, ServerlessSpec
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store implementation using Pinecone for semantic search
    """
    
    def __init__(self, api_key: str, environment: str = "us-east-1", index_name: str = "business-rag-index"):
        """
        Initialize Pinecone vector store
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the index to use
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.metric = "cosine"
        self.pc = None
        self.index = None
        
    def initialize_pinecone(self):
        """Initialize Pinecone client"""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {e}")
            raise
    
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            if not self.pc:
                self.initialize_pinecone()
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating new index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                
                logger.info(f"Index {self.index_name} created and ready")
            else:
                logger.info(f"Index {self.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def connect_to_index(self):
        """Connect to existing index"""
        try:
            if not self.pc:
                self.initialize_pinecone()
            
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to index: {e}")
            raise
    
    def initialize_index(self):
        """Initialize the complete index setup"""
        try:
            self.initialize_pinecone()
            self.create_index()
            self.connect_to_index()
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert vectors to Pinecone index
        
        Args:
            vectors: List of vectors with id, values, and metadata
            batch_size: Number of vectors to process in each batch
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call initialize_index() first.")
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Format vectors for Pinecone
                formatted_vectors = []
                for vector in batch:
                    formatted_vectors.append({
                        'id': vector['id'],
                        'values': vector['values'],
                        'metadata': vector['metadata']
                    })
                
                # Upsert batch
                self.index.upsert(vectors=formatted_vectors)
                logger.info(f"Upserted batch {i // batch_size + 1} with {len(batch)} vectors")
            
            logger.info(f"Successfully upserted {len(vectors)} vectors")
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise
    
    def query_vectors(self, query_vector: List[float], top_k: int = 5, 
                     include_metadata: bool = True, filter_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query vectors from Pinecone index
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of top results to return
            include_metadata: Whether to include metadata in results
            filter_dict: Optional metadata filter
            
        Returns:
            Query results from Pinecone
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call initialize_index() first.")
            
            # Perform query
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=include_metadata,
                filter=filter_dict
            )
            
            logger.info(f"Query returned {len(results['matches'])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying vectors: {e}")
            raise
    
    def delete_vectors(self, vector_ids: List[str]):
        """
        Delete vectors from index
        
        Args:
            vector_ids: List of vector IDs to delete
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call initialize_index() first.")
            
            self.index.delete(ids=vector_ids)
            logger.info(f"Deleted {len(vector_ids)} vectors")
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get index statistics
        
        Returns:
            Dictionary of index statistics
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call initialize_index() first.")
            
            stats = self.index.describe_index_stats()
            return {
                'total_vector_count': stats['total_vector_count'],
                'dimension': stats['dimension'],
                'index_fullness': stats['index_fullness'],
                'namespaces': stats.get('namespaces', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}
    
    def clear_index(self):
        """Clear all vectors from index"""
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call initialize_index() first.")
            
            self.index.delete(delete_all=True)
            logger.info("Cleared all vectors from index")
            
        except Exception as e:
            logger.error(f"Error clearing index: {e}")
            raise
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search vectors by metadata filter
        
        Args:
            metadata_filter: Metadata filter conditions
            top_k: Number of results to return
            
        Returns:
            List of matching vectors
        """
        try:
            if not self.index:
                raise ValueError("Index not initialized. Call initialize_index() first.")
            
            # Create a dummy query vector (all zeros) since we're filtering by metadata
            dummy_vector = [0.0] * self.dimension
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter
            )
            
            return results['matches']
            
        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []
