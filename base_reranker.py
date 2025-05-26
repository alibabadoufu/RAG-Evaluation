"""
Base class for reranking strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseReranker(ABC):
    """
    Abstract base class for document reranking strategies.
    
    All reranker implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, top_k: int = 3, **kwargs):
        """
        Initialize the reranker with configurable parameters.
        
        Args:
            top_k: The number of documents to return after reranking
            **kwargs: Additional parameters specific to the reranking strategy
        """
        self.top_k = top_k
        self.config = kwargs
        
    @abstractmethod
    def rerank(self, query: str, documents: List[Tuple[Dict[str, Any], float]], top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank a list of retrieved documents based on their relevance to the query.
        
        Args:
            query: The query text
            documents: A list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A relevance score from the retriever
            top_k: Optional override for the number of documents to return
            
        Returns:
            A reordered list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A new relevance score (higher is more relevant)
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the reranker."""
        return f"{self.__class__.__name__}(top_k={self.top_k})"
