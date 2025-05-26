"""
Implementation of recursive chunking strategy.
"""
from typing import List, Dict, Any, Optional
import re

from src.components.base.base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):
    """
    Recursive chunking strategy that splits documents based on hierarchical structure.
    
    This chunker recursively splits documents by headers, paragraphs, and sentences
    to create semantically meaningful chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 split_level: str = "paragraph", **kwargs):
        """
        Initialize the recursive chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters
            chunk_overlap: The overlap between consecutive chunks
            split_level: The level at which to split ('header', 'paragraph', 'sentence')
            **kwargs: Additional parameters
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.split_level = split_level
        
    def chunk_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split a document into chunks using recursive splitting.
        
        Args:
            document: The document text to be chunked
            metadata: Optional metadata associated with the document
            
        Returns:
            A list of dictionaries, each containing chunk text and metadata
        """
        if metadata is None:
            metadata = {}
            
        # Initialize chunks list
        chunks = []
        
        # Split by headers first
        header_pattern = r'(#{1,6}\s+.+?(?:\n|$))'
        header_splits = re.split(header_pattern, document, flags=re.MULTILINE)
        
        current_header = ""
        current_text = ""
        current_size = 0
        
        for i, split in enumerate(header_splits):
            # If this is a header, store it
            if re.match(header_pattern, split):
                # If we have accumulated text, add it as a chunk
                if current_size > 0:
                    self._add_chunk_from_text(chunks, current_text, current_header, metadata)
                
                current_header = split.strip()
                current_text = ""
                current_size = 0
            else:
                # For content sections, split by paragraphs if needed
                if self.split_level in ["paragraph", "sentence"]:
                    paragraphs = re.split(r'\n\s*\n', split)
                    
                    for para in paragraphs:
                        para = para.strip()
                        if not para:
                            continue
                            
                        # For sentence-level splitting
                        if self.split_level == "sentence" and len(para) > self.chunk_size:
                            sentences = re.split(r'(?<=[.!?])\s+', para)
                            
                            for sentence in sentences:
                                if current_size + len(sentence) > self.chunk_size and current_size > 0:
                                    self._add_chunk_from_text(chunks, current_text, current_header, metadata)
                                    current_text = sentence
                                    current_size = len(sentence)
                                else:
                                    if current_size > 0:
                                        current_text += " "
                                    current_text += sentence
                                    current_size += len(sentence)
                        else:
                            # Paragraph-level handling
                            if current_size + len(para) > self.chunk_size and current_size > 0:
                                self._add_chunk_from_text(chunks, current_text, current_header, metadata)
                                current_text = para
                                current_size = len(para)
                            else:
                                if current_size > 0:
                                    current_text += "\n\n"
                                current_text += para
                                current_size += len(para)
                else:
                    # Header-level only, just add the whole content
                    if current_size + len(split) > self.chunk_size and current_size > 0:
                        self._add_chunk_from_text(chunks, current_text, current_header, metadata)
                        current_text = split
                        current_size = len(split)
                    else:
                        current_text += split
                        current_size += len(split)
        
        # Add the last chunk if there's any text left
        if current_size > 0:
            self._add_chunk_from_text(chunks, current_text, current_header, metadata)
            
        return chunks
    
    def _add_chunk_from_text(self, chunks: List[Dict[str, Any]], text: str, header: str, metadata: Dict[str, Any]) -> None:
        """
        Helper method to add a chunk to the chunks list.
        
        Args:
            chunks: The list of chunks to append to
            text: The chunk text
            header: The current header
            metadata: The document metadata
        """
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "header": header,
            "chunk_index": len(chunks),
        })
        
        chunks.append({
            "text": text.strip(),
            "metadata": chunk_metadata
        })
