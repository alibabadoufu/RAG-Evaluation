"""
Base class for evaluation metrics.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.
    
    All evaluation metric implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the evaluation metric with configurable parameters.
        
        Args:
            name: The name of the metric
            **kwargs: Additional parameters specific to the metric
        """
        self.name = name
        self.config = kwargs
        
    @abstractmethod
    def evaluate(self, 
                query: str, 
                response: str, 
                context: List[Dict[str, Any]], 
                ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a single query-response pair.
        
        Args:
            query: The user query
            response: The generated response
            context: The context documents used for generation
            ground_truth: Optional ground truth information for reference
            
        Returns:
            A dictionary containing:
                - 'score': The primary metric score
                - 'details': Additional details about the evaluation
        """
        pass
    
    @abstractmethod
    def evaluate_batch(self, 
                      queries: List[str], 
                      responses: List[str], 
                      contexts: List[List[Dict[str, Any]]], 
                      ground_truths: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate a batch of query-response pairs.
        
        Args:
            queries: List of user queries
            responses: List of generated responses
            contexts: List of context document lists used for generation
            ground_truths: Optional list of ground truth information for reference
            
        Returns:
            A dictionary containing:
                - 'scores': List of individual scores
                - 'average': Average score across all examples
                - 'details': Additional details about the evaluation
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(name={self.name})"
