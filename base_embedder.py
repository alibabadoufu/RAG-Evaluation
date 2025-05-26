"""
Base class for embedding strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding strategies.
    
    All embedding implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, embedding_dim: int = 768, **kwargs):
        """
        Initialize the embedder with configurable parameters.
        
        Args:
            embedding_dim: The dimensionality of the embedding vectors
            **kwargs: Additional parameters specific to the embedding strategy
        """
        self.embedding_dim = embedding_dim
        self.config = kwargs
        
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of texts.
        
        Args:
            texts: A list of texts to embed
            
        Returns:
            A list of embedding vectors, each a list of floats
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the embedder."""
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim})"
