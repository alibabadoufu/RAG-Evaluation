"""
Implementation of sentence chunking strategy.
"""
from typing import List, Dict, Any, Optional
import re

from src.components.base.base_chunker import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Sentence chunking strategy that splits documents by sentences.
    
    This chunker creates chunks by grouping sentences together up to the
    specified chunk size, with optional overlap between chunks.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 separator: str = " ", **kwargs):
        """
        Initialize the sentence chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters
            chunk_overlap: The overlap between consecutive chunks
            separator: The string to use when joining sentences
            **kwargs: Additional parameters
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separator = separator
        
    def chunk_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split a document into chunks by sentences.
        
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
        
        # Split the document into sentences
        sentences = re.split(r'(?<=[.!?])\s+', document)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_sentences = []
        sentence_indices = []
        
        for i, sentence in enumerate(sentences):
            # If adding this sentence would exceed the chunk size
            if len(current_chunk) + len(self.separator) + len(sentence) > self.chunk_size and current_chunk:
                # Add the current chunk to our list
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "sentence_indices": sentence_indices
                })
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
                
                # Start a new chunk with overlap
                if self.chunk_overlap > 0 and current_sentences:
                    # Calculate how many sentences to include for overlap
                    overlap_size = 0
                    overlap_sentences = []
                    overlap_indices = []
                    
                    for j, s in enumerate(reversed(current_sentences)):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_indices.insert(0, sentence_indices[len(sentence_indices) - j - 1])
                            overlap_size += len(s) + len(self.separator)
                        else:
                            break
                    
                    current_chunk = self.separator.join(overlap_sentences)
                    current_sentences = overlap_sentences
                    sentence_indices = overlap_indices
                else:
                    current_chunk = ""
                    current_sentences = []
                    sentence_indices = []
            
            # Add the current sentence
            if current_chunk:
                current_chunk += self.separator
            current_chunk += sentence
            current_sentences.append(sentence)
            sentence_indices.append(i)
            
            # Special case: if a single sentence exceeds chunk size
            if len(sentence) > self.chunk_size and len(current_sentences) == 1:
                # Just add it as its own chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "sentence_indices": [i],
                    "is_long_sentence": True
                })
                
                chunks.append({
                    "text": sentence.strip(),
                    "metadata": chunk_metadata
                })
                
                # Reset for next sentence
                current_chunk = ""
                current_sentences = []
                sentence_indices = []
        
        # Add the last chunk if there's any text left
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "sentence_indices": sentence_indices
            })
            
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
            
        return chunks
