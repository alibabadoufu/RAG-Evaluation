from typing import List, Optional, Callable
from langchain.schema import Document
import re


class ContextualChunker:
    """
    A contextual document chunker that adds document summaries to each chunk
    for better RAG performance by providing global context.
    """
    
    def __init__(self, chunk_size: int, chunk_overlap: int, summarizer: Optional[Callable[[str], str]] = None):
        """
        Initialize the ContextualChunker with specified parameters.
        
        Args:
            chunk_size: Maximum characters allowed in each original chunk
            chunk_overlap: Number of characters to overlap between successive chunks
            summarizer: Optional function to summarize documents (defaults to simple summarizer)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summarizer = summarizer or self._default_summarizer
    
    def chunk_document(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents with contextual summaries added to each chunk.
        
        Args:
            documents: List of Document objects from langchain
            
        Returns:
            List of chunked Document objects with contextualized content
        """
        contextualized_documents = []
        
        for doc in documents:
            # Step 1: Generate summary of the entire document
            document_summary = self.summarizer(doc.page_content)
            
            # Step 2: Split document into chunks with overlap
            original_chunks = self._split_with_overlap(doc.page_content)
            
            # Step 3: Create contextualized chunks
            for i, original_chunk in enumerate(original_chunks):
                # Create the contextualized content using the template
                contextualized_content = self._create_contextualized_content(
                    original_chunk, document_summary
                )
                
                # Create new Document object with contextualized content
                chunk_doc = Document(
                    page_content=contextualized_content,
                    metadata=doc.metadata.copy()  # Copy original metadata
                )
                
                # Add chunk-specific metadata
                chunk_doc.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(original_chunks),
                    'original_chunk': original_chunk,
                    'document_summary': document_summary,
                    'chunk_type': 'contextualized'
                })
                
                contextualized_documents.append(chunk_doc)
        
        return contextualized_documents
    
    def _split_with_overlap(self, content: str) -> List[str]:
        """
        Split content into overlapping chunks of specified size.
        
        Args:
            content: The document content to split
            
        Returns:
            List of overlapping chunks
        """
        chunks = []
        
        # If content is smaller than chunk_size, return as single chunk
        if len(content) <= self.chunk_size:
            return [content]
        
        start = 0
        while start < len(content):
            # Calculate end position for current chunk
            end = start + self.chunk_size
            
            # If we're not at the end of content, try to break at word boundary
            if end < len(content):
                # Look for the last space within the chunk to avoid breaking words
                last_space = content.rfind(' ', start, end)
                if last_space > start:  # Found a space within the chunk
                    end = last_space
            
            # Extract the chunk
            chunk = content[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            # If this is the last chunk, break to avoid infinite loop
            if end >= len(content):
                break
                
            # Move start position back by chunk_overlap for next chunk
            next_start = end - self.chunk_overlap
            
            # Ensure we make progress (avoid infinite loop)
            if next_start <= start:
                next_start = start + 1
            
            start = next_start
            
            # Skip leading whitespace for next chunk
            while start < len(content) and content[start].isspace():
                start += 1
        
        return chunks
    
    def _create_contextualized_content(self, original_chunk: str, summarized_chunk: str) -> str:
        """
        Create contextualized content using the specified template.
        
        Args:
            original_chunk: The original chunk content
            summarized_chunk: The document summary
            
        Returns:
            Formatted contextualized content
        """
        template = """original_chunk = {original_chunk}

contextualized_chunk = {summarized_chunk}"""
        
        return template.format(
            original_chunk=original_chunk,
            summarized_chunk=summarized_chunk
        )
    
    def _default_summarizer(self, content: str) -> str:
        """
        Default summarizer that creates a simple extractive summary.
        In practice, you would replace this with a more sophisticated summarizer
        like using OpenAI GPT, Claude, or other language models.
        
        Args:
            content: Content to summarize
            
        Returns:
            Summary of the content
        """
        # Simple extractive summarizer - takes first and key sentences
        sentences = self._split_into_sentences(content)
        
        if len(sentences) <= 3:
            return content  # If very short, return as is
        
        # Take first sentence and a few key sentences from middle/end
        summary_sentences = []
        
        # Always include first sentence
        summary_sentences.append(sentences[0])
        
        # Add some middle sentences (every nth sentence)
        if len(sentences) > 6:
            step = len(sentences) // 4
            for i in range(step, len(sentences) - 1, step):
                summary_sentences.append(sentences[i])
        
        # Add last sentence if not already included
        if len(sentences) > 1 and sentences[-1] not in summary_sentences:
            summary_sentences.append(sentences[-1])
        
        # Join sentences and limit length
        summary = ' '.join(summary_sentences)
        
        # Limit summary length to prevent it from being too long
        max_summary_length = min(500, len(content) // 4)
        if len(summary) > max_summary_length:
            summary = summary[:max_summary_length].rsplit(' ', 1)[0] + '...'
        
        return summary
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple regex patterns.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # This handles basic cases - for production, consider using spaCy or NLTK
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean up sentences - remove empty ones and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def set_custom_summarizer(self, summarizer_func: Callable[[str], str]):
        """
        Set a custom summarizer function.
        
        Args:
            summarizer_func: Function that takes content string and returns summary string
        """
        self.summarizer = summarizer_func


# Example usage with custom summarizer
def custom_summarizer(content: str) -> str:
    """
    Example custom summarizer - you would integrate with your preferred LLM here.
    """
    # This is a placeholder - in practice you'd call your LLM API here
    # For example: openai.chat.completions.create() or anthropic.messages.create()
    
    # Simple keyword-based summary for demonstration
    words = content.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Only consider longer words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    keywords = [word for word, freq in top_words]
    
    # Create summary with keywords
    summary = f"This document discusses topics related to: {', '.join(keywords[:5])}. "
    
    # Add first sentence for context
    sentences = content.split('.')
    if sentences:
        summary += sentences[0] + '.'
    
    return summary


if __name__ == "__main__":
    # Create sample documents
    sample_docs = [
        Document(
            page_content="""
            Machine learning is a subset of artificial intelligence that focuses on algorithms 
            that can learn from data. Deep learning is a specialized area within machine learning 
            that uses neural networks with multiple layers. These neural networks can automatically 
            learn complex patterns in data without explicit programming. Applications include 
            image recognition, natural language processing, and autonomous vehicles. The field 
            has seen rapid advancement in recent years due to increased computational power and 
            availability of large datasets. Companies like Google, Microsoft, and OpenAI have 
            made significant contributions to the field.
            """.strip(),
            metadata={"source": "ml_intro.txt", "type": "educational"}
        )
    ]
    
    # Initialize chunker with parameters
    chunker = ContextualChunker(
        chunk_size=200,      # Each original chunk max 200 characters
        chunk_overlap=50     # 50 character overlap between chunks
    )
    
    # Optional: Set custom summarizer
    chunker.set_custom_summarizer(custom_summarizer)
    
    # Chunk the documents
    contextualized_docs = chunker.chunk_document(sample_docs)
    
    # Display results
    for i, doc in enumerate(contextualized_docs):
        print(f"Contextualized Chunk {i + 1}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print("-" * 70)
        print()
