"""
Base class for document parsing strategies.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
import os


class BaseParser(ABC):
    """
    Abstract base class for document parsing strategies.
    
    All parser implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the parser with configurable parameters.
        
        Args:
            **kwargs: Parameters specific to the parsing strategy
        """
        self.config = kwargs
        
    @abstractmethod
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document file into text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            A dictionary containing:
                - 'text': The extracted text content
                - 'metadata': Metadata about the document (e.g., filename, file type, etc.)
        """
        pass
    
    @abstractmethod
    def parse_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse raw text into a structured format.
        
        Args:
            text: The text to parse
            metadata: Optional metadata to associate with the text
            
        Returns:
            A dictionary containing:
                - 'text': The processed text
                - 'metadata': Metadata about the text
        """
        pass
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Parse all documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            A list of dictionaries, each containing parsed document text and metadata
        """
        documents = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    document = self.parse_file(file_path)
                    documents.append(document)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
        return documents
    
    def __str__(self) -> str:
        """String representation of the parser."""
        return f"{self.__class__.__name__}()"
