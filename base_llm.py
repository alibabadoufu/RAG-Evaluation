"""
Base class for LLM inference strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union


class BaseLLM(ABC):
    """
    Abstract base class for LLM inference strategies.
    
    All LLM implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, **kwargs):
        """
        Initialize the LLM with configurable parameters.
        
        Args:
            model_name: The name or identifier of the model to use
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters specific to the LLM
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs
        
    @abstractmethod
    def generate(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text based on a prompt and optional context.
        
        Args:
            prompt: The input prompt for the LLM
            context: Optional list of context documents, each a dictionary with 'text' and 'metadata'
            
        Returns:
            The generated text response
        """
        pass
    
    @abstractmethod
    def generate_with_chat_history(self, 
                                  messages: List[Dict[str, str]], 
                                  context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text based on a chat history and optional context.
        
        Args:
            messages: List of message dictionaries, each with 'role' (user/assistant/system) and 'content'
            context: Optional list of context documents, each a dictionary with 'text' and 'metadata'
            
        Returns:
            The generated text response
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the LLM."""
        return f"{self.__class__.__name__}(model_name={self.model_name}, temperature={self.temperature})"
