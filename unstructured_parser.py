"""
Implementation of unstructured parser for document parsing.
"""
from typing import List, Dict, Any, Optional
import os
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element, Text

from src.components.base.base_parser import BaseParser


class UnstructuredParser(BaseParser):
    """
    Parser using the unstructured library for document parsing.
    
    This parser can handle various document formats including PDF, DOCX, HTML, etc.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the unstructured parser.
        
        Args:
            **kwargs: Additional parameters for the unstructured library
        """
        super().__init__(**kwargs)
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document file into text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            A dictionary containing:
                - 'text': The extracted text content
                - 'metadata': Metadata about the document
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Get file metadata
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        file_size = os.path.getsize(file_path)
        
        # Parse the document using unstructured
        elements = partition(file_path, **self.config)
        
        # Extract text from elements
        text_elements = [el for el in elements if isinstance(el, Text)]
        text = "\n\n".join([str(el) for el in text_elements])
        
        # Create metadata
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_ext,
            "file_size": file_size,
            "element_count": len(elements),
            "text_element_count": len(text_elements)
        }
        
        return {
            "text": text,
            "metadata": metadata
        }
    
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
        if metadata is None:
            metadata = {}
            
        # For plain text, we just return it as is with metadata
        return {
            "text": text,
            "metadata": metadata
        }
