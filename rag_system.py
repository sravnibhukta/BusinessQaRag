import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
import numpy as np
from vector_store import VectorStore
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Main RAG system that orchestrates document processing, embedding generation,
    vector storage, and query processing for business QA
    """
    
    def __init__(self, openai_api_key: str, vector_store: VectorStore, document_processor: DocumentProcessor):
        """
        Initialize the RAG system
        
        Args:
            openai_api_key: OpenAI API key
            vector_store: Vector store instance
            document_processor: Document processor instance
        """
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.vector_store = vector_store
        self.document_processor = document_processor
        self.metrics = {
            'total_documents': 0,
            'total_chunks': 0,
            'queries_processed': 0
        }
        
        # Initialize vector store
        self.vector_store.initialize_index()
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding model
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use OpenAI's text-embedding-ada-002 model
            response = self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            
            embeddings = [embedding.embedding for embedding in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_document(self, content: str, source: str) -> None:
        """
        Add a document to the knowledge base
        
        Args:
            content: Document content
            source: Document source/filename
        """
        try:
            # Process document into chunks
            chunks = self.document_processor.process_document(content, source)
            
            if not chunks:
                logger.warning(f"No chunks generated for document: {source}")
                return
            
            # Generate embeddings for chunks
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            # Store in vector database
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{source}_{i}"
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': {
                        'content': chunk['content'],
                        'source': source,
                        'chunk_index': i,
                        'start_char': chunk['start_char'],
                        'end_char': chunk['end_char']
                    }
                })
            
            self.vector_store.upsert_vectors(vectors)
            
            # Update metrics
            self.metrics['total_documents'] += 1
            self.metrics['total_chunks'] += len(chunks)
            
            logger.info(f"Successfully added document: {source} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document {source}: {e}")
            raise
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in vector store
            results = self.vector_store.query_vectors(
                query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            relevant_chunks = []
            for match in results['matches']:
                relevant_chunks.append({
                    'content': match['metadata']['content'],
                    'source': match['metadata']['source'],
                    'score': match['score'],
                    'metadata': match['metadata']
                })
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            raise
    
    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        """
        Generate answer using retrieved context and OpenAI
        
        Args:
            query: User query
            context_chunks: Retrieved relevant chunks
            temperature: Response temperature
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context from retrieved chunks
            context = "\n\n".join([
                f"Source: {chunk['source']}\nContent: {chunk['content']}"
                for chunk in context_chunks
            ])
            
            # Create prompt for answer generation
            system_prompt = """You are a helpful business assistant that answers questions based on the provided business knowledge base. 
            
            Instructions:
            1. Use only the information provided in the context to answer questions
            2. If the answer cannot be found in the context, clearly state that you don't have enough information
            3. Be concise but comprehensive in your responses
            4. Reference specific sources when possible
            5. Focus on business-relevant information and practical advice
            
            Context:
            {context}
            
            Question: {query}
            
            Please provide a helpful and accurate answer based on the context above."""
            
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt.format(context=context, query=query)
                    }
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            logger.info("Generated answer using OpenAI")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def query(self, query: str, top_k: int = 5, temperature: float = 0.1) -> Dict[str, Any]:
        """
        Main query method that retrieves relevant information and generates an answer
        
        Args:
            query: User query
            top_k: Number of top results to retrieve
            temperature: Response temperature
            
        Returns:
            Dictionary containing answer and sources
        """
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, top_k)
            
            if not relevant_chunks:
                return {
                    'answer': "I don't have enough information in the knowledge base to answer your question.",
                    'sources': []
                }
            
            # Generate answer
            answer = self.generate_answer(query, relevant_chunks, temperature)
            
            # Update metrics
            self.metrics['queries_processed'] += 1
            
            return {
                'answer': answer,
                'sources': relevant_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"Sorry, I encountered an error while processing your question: {str(e)}",
                'sources': []
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
