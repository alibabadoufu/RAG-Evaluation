"""
Implementation of SentenceTransformer embedding strategy.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from src.components.base.base_embedder import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedding strategy using SentenceTransformer models.
    
    This embedder uses the sentence-transformers library to generate embeddings for text.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 embedding_dim: Optional[int] = None, 
                 device: str = "cpu", 
                 **kwargs):
        """
        Initialize the SentenceTransformer embedder.
        
        Args:
            model_name: The name of the SentenceTransformer model to use
            embedding_dim: The dimensionality of the embedding vectors (if None, determined from model)
            device: The device to run the model on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension from model if not specified
        if embedding_dim is None:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            
        super().__init__(embedding_dim=embedding_dim, **kwargs)
        self.model_name = model_name
        self.device = device
        
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text using SentenceTransformer.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts using SentenceTransformer.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors, each a list of floats
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
