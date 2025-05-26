"""
Implementation of OpenAI embedding strategy.
"""
from typing import List, Dict, Any, Optional
import os
import openai

from src.components.base.base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding strategy using OpenAI's embedding models.
    
    This embedder uses OpenAI's API to generate embeddings for text.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None, 
                 embedding_dim: int = 1536, 
                 **kwargs):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model_name: The name of the OpenAI embedding model to use
            api_key: OpenAI API key (if None, will look for OPENAI_API_KEY env var)
            embedding_dim: The dimensionality of the embedding vectors
            **kwargs: Additional parameters
        """
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.model_name = model_name
        
        # Set up API key
        if api_key is not None:
            openai.api_key = api_key
        elif os.environ.get("OPENAI_API_KEY") is not None:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
        else:
            raise ValueError("OpenAI API key must be provided either as an argument or as an environment variable")
        
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text using OpenAI.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        response = openai.Embedding.create(
            model=self.model_name,
            input=text
        )
        
        return response["data"][0]["embedding"]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts using OpenAI.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors, each a list of floats
        """
        response = openai.Embedding.create(
            model=self.model_name,
            input=texts
        )
        
        return [item["embedding"] for item in response["data"]]
