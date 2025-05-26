"""
Base class for retrieval strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseRetriever(ABC):
    """
    Abstract base class for document retrieval strategies.
    
    All retriever implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, top_k: int = 5, **kwargs):
        """
        Initialize the retriever with configurable parameters.
        
        Args:
            top_k: The number of documents to retrieve
            **kwargs: Additional parameters specific to the retrieval strategy
        """
        self.top_k = top_k
        self.config = kwargs
        
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retrieval index.
        
        Args:
            documents: A list of document dictionaries, each containing at minimum:
                - 'text': The document text
                - 'metadata': Document metadata
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The query text
            top_k: Optional override for the number of documents to retrieve
            
        Returns:
            A list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A relevance score (higher is more relevant)
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the retriever."""
        return f"{self.__class__.__name__}(top_k={self.top_k})"
