"""
Base class for document chunking strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseChunker(ABC):
    """
    Abstract base class for document chunking strategies.
    
    All chunking implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        """
        Initialize the chunker with configurable parameters.
        
        Args:
            chunk_size: The target size of each chunk in characters or tokens
            chunk_overlap: The overlap between consecutive chunks
            **kwargs: Additional parameters specific to the chunking strategy
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs
        
    @abstractmethod
    def chunk_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split a document into chunks according to the chunking strategy.
        
        Args:
            document: The document text to be chunked
            metadata: Optional metadata associated with the document
            
        Returns:
            A list of dictionaries, each containing:
                - 'text': The chunk text
                - 'metadata': Metadata for the chunk, including original document metadata
                  and chunk-specific metadata like position
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the chunker."""
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
