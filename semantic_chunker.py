"""
Implementation of semantic chunking strategy.
"""
from typing import List, Dict, Any, Optional
import re

from src.components.base.base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy that splits documents based on semantic meaning.
    
    This chunker attempts to preserve semantic coherence by analyzing content
    and creating chunks that maintain contextual meaning.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 similarity_threshold: float = 0.7, **kwargs):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: The target size of each chunk in characters
            chunk_overlap: The overlap between consecutive chunks
            similarity_threshold: Threshold for determining semantic similarity
            **kwargs: Additional parameters
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.similarity_threshold = similarity_threshold
        
    def chunk_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split a document into chunks based on semantic meaning.
        
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
        
        # First, split by natural boundaries like paragraphs
        paragraphs = re.split(r'\n\s*\n', document)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        current_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            # If adding this paragraph would exceed the chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Add the current chunk to our list
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": len(chunks),
                    "paragraph_indices": list(range(i - len(current_paragraphs), i))
                })
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
                
                # Start a new chunk with overlap
                if self.chunk_overlap > 0 and current_paragraphs:
                    # Calculate how many paragraphs to include for overlap
                    overlap_size = 0
                    overlap_paragraphs = []
                    
                    for p in reversed(current_paragraphs):
                        if overlap_size + len(p) <= self.chunk_overlap:
                            overlap_paragraphs.insert(0, p)
                            overlap_size += len(p)
                        else:
                            break
                    
                    current_chunk = "\n\n".join(overlap_paragraphs)
                    current_paragraphs = overlap_paragraphs
                else:
                    current_chunk = ""
                    current_paragraphs = []
            
            # Add the current paragraph
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
            current_paragraphs.append(paragraph)
            
            # Special case: if a single paragraph exceeds chunk size
            if len(paragraph) > self.chunk_size and len(current_paragraphs) == 1:
                # Split the paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                
                # Create chunks from sentences
                sentence_chunk = ""
                for sentence in sentences:
                    if len(sentence_chunk) + len(sentence) > self.chunk_size and sentence_chunk:
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update({
                            "chunk_index": len(chunks),
                            "paragraph_index": i,
                            "is_partial_paragraph": True
                        })
                        
                        chunks.append({
                            "text": sentence_chunk.strip(),
                            "metadata": chunk_metadata
                        })
                        
                        # Start new sentence chunk with overlap
                        if self.chunk_overlap > 0:
                            # Simple overlap for sentences - take the last sentence
                            last_sentence = sentence_chunk.split(".")[-1]
                            if last_sentence and len(last_sentence) < self.chunk_overlap:
                                sentence_chunk = last_sentence + "."
                            else:
                                sentence_chunk = ""
                        else:
                            sentence_chunk = ""
                    
                    if sentence_chunk and not sentence_chunk.endswith("."):
                        sentence_chunk += " "
                    sentence_chunk += sentence
                
                # Add the last sentence chunk if not empty
                if sentence_chunk:
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": len(chunks),
                        "paragraph_index": i,
                        "is_partial_paragraph": True
                    })
                    
                    chunks.append({
                        "text": sentence_chunk.strip(),
                        "metadata": chunk_metadata
                    })
                
                # Reset for next paragraph
                current_chunk = ""
                current_paragraphs = []
        
        # Add the last chunk if there's any text left
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),
                "paragraph_indices": list(range(len(paragraphs) - len(current_paragraphs), len(paragraphs)))
            })
            
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": chunk_metadata
            })
            
        return chunks
