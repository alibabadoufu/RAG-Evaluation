"""
Implementation of Mistral OCR parser for document parsing.
"""
from typing import List, Dict, Any, Optional
import os
import requests
import base64

from src.components.base.base_parser import BaseParser


class MistralOCRParser(BaseParser):
    """
    Parser using Mistral's OCR capabilities for document parsing.
    
    This parser specializes in extracting text from images and scanned documents
    using Mistral's OCR capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "mistral-large-latest", **kwargs):
        """
        Initialize the Mistral OCR parser.
        
        Args:
            api_key: Mistral API key (if None, will look for MISTRAL_API_KEY env var)
            model_name: The name of the Mistral model to use
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        
        # Set up API key
        if api_key is not None:
            self.api_key = api_key
        elif os.environ.get("MISTRAL_API_KEY") is not None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")
        else:
            raise ValueError("Mistral API key must be provided either as an argument or as an environment variable")
        
        # OCR prompt template
        self.ocr_prompt_template = """
        Extract all the text from the provided image. 
        Maintain the original formatting as much as possible.
        Include all text, tables, and structured content.
        """
        
    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a document file into text and metadata using Mistral OCR.
        
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
        
        # Check if file is an image
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        if file_ext.lower() in image_extensions:
            # Encode the image
            base64_image = self._encode_image(file_path)
            
            # Call Mistral API for OCR
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.ocr_prompt_template},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_text = result["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OCR API error: {response.status_code} - {response.text}")
        else:
            # For non-image files, use a different approach or raise an error
            raise ValueError(f"File type {file_ext} not supported by MistralOCRParser. Use for image files only.")
        
        # Create metadata
        metadata = {
            "file_name": file_name,
            "file_path": file_path,
            "file_type": file_ext,
            "file_size": file_size,
            "parser": "MistralOCR",
            "model": self.model_name
        }
        
        return {
            "text": extracted_text,
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
        # This parser is primarily for OCR from images
        metadata["parser"] = "MistralOCR"
        metadata["model"] = self.model_name
        metadata["note"] = "Text was provided directly, no OCR performed"
        
        return {
            "text": text,
            "metadata": metadata
        }
