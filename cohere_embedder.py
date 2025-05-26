"""
Implementation of Cohere embedding strategy.
"""
from typing import List, Dict, Any, Optional
import os
import cohere

from src.components.base.base_embedder import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    """
    Embedding strategy using Cohere's embedding models.
    
    This embedder uses Cohere's API to generate embeddings for text.
    """
    
    def __init__(self, model_name: str = "embed-english-v3.0", 
                 api_key: Optional[str] = None, 
                 embedding_dim: int = 1024, 
                 **kwargs):
        """
        Initialize the Cohere embedder.
        
        Args:
            model_name: The name of the Cohere embedding model to use
            api_key: Cohere API key (if None, will look for COHERE_API_KEY env var)
            embedding_dim: The dimensionality of the embedding vectors
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.model_name = model_name
        
        # Set up API key
        if api_key is not None:
            self.api_key = api_key
        elif os.environ.get("COHERE_API_KEY") is not None:
            self.api_key = os.environ.get("COHERE_API_KEY")
        else:
            raise ValueError("Cohere API key must be provided either as an argument or as an environment variable")
        
        # Initialize client
        self.client = cohere.Client(self.api_key)
        
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text using Cohere.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        response = self.client.embed(
            texts=[text],
            model=self.model_name
        )
        
        return response.embeddings[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts using Cohere.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors, each a list of floats
        """
        response = self.client.embed(
            texts=texts,
            model=self.model_name
        )
        
        return response.embeddings
