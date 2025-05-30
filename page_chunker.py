from typing import List
from langchain.schema import Document


class PageChunker:
    """
    A page-based document chunker for RAG systems that creates chunks with overlap
    and expanded context for better retrieval performance.
    """
    
    def __init__(self, max_size_page_content: int, max_size_per_chunk: int, page_overlap: int):
        """
        Initialize the PageChunker with specified parameters.
        
        Args:
            max_size_page_content: Maximum characters allowed in each chunk's page_content
            max_size_per_chunk: Maximum characters for the expanded chunk (with overlap)
            page_overlap: Number of pages before and after to include in expanded chunk
        """
        self.max_size_page_content = max_size_page_content
        self.max_size_per_chunk = max_size_per_chunk
        self.page_overlap = page_overlap
    
    def chunk_document(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents based on page content size and create expanded chunks.
        
        Args:
            documents: List of Document objects from langchain
            
        Returns:
            List of chunked Document objects with expanded context in metadata
        """
        chunked_documents = []
        
        for doc in documents:
            # Split the document content into initial chunks based on max_size_page_content
            initial_chunks = self._split_document_content(doc.page_content)
            
            # Create Document objects for each chunk with original metadata
            chunk_docs = []
            for i, chunk_content in enumerate(initial_chunks):
                # Create new document with chunk content and copy original metadata
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata=doc.metadata.copy()  # Copy original metadata
                )
                # Add chunk index to metadata for reference
                chunk_doc.metadata['chunk_index'] = i
                chunk_doc.metadata['total_chunks'] = len(initial_chunks)
                chunk_docs.append(chunk_doc)
            
            # Create expanded chunks with page overlap
            for i, chunk_doc in enumerate(chunk_docs):
                expanded_chunk = self._create_expanded_chunk(chunk_docs, i)
                # Store the expanded chunk in metadata
                chunk_doc.metadata['expanded_chunk'] = expanded_chunk
                chunked_documents.append(chunk_doc)
        
        return chunked_documents
    
    def _split_document_content(self, content: str) -> List[str]:
        """
        Split document content into chunks based on max_size_page_content.
        
        Args:
            content: The document content to split
            
        Returns:
            List of content chunks
        """
        chunks = []
        
        # If content is smaller than max size, return as single chunk
        if len(content) <= self.max_size_page_content:
            return [content]
        
        # Split content into chunks of max_size_page_content
        start = 0
        while start < len(content):
            end = start + self.max_size_page_content
            
            # If we're not at the end, try to break at a word boundary
            if end < len(content):
                # Look for the last space within the chunk to avoid breaking words
                last_space = content.rfind(' ', start, end)
                if last_space > start:  # Found a space within the chunk
                    end = last_space
            
            chunks.append(content[start:end].strip())
            start = end
            
            # Skip any leading whitespace for the next chunk
            while start < len(content) and content[start].isspace():
                start += 1
        
        return chunks
    
    def _create_expanded_chunk(self, chunk_docs: List[Document], current_index: int) -> str:
        """
        Create an expanded chunk by combining the current chunk with overlapping chunks.
        
        Args:
            chunk_docs: List of chunk documents
            current_index: Index of the current chunk
            
        Returns:
            Expanded chunk content as string
        """
        # Calculate the range of chunks to include based on page_overlap
        start_idx = max(0, current_index - self.page_overlap)
        end_idx = min(len(chunk_docs), current_index + self.page_overlap + 1)
        
        # Collect content from overlapping chunks
        expanded_content_parts = []
        for idx in range(start_idx, end_idx):
            expanded_content_parts.append(chunk_docs[idx].page_content)
        
        # Join all parts with double newline to separate chunks clearly
        expanded_content = '\n\n'.join(expanded_content_parts)
        
        # If expanded content exceeds max_size_per_chunk, truncate intelligently
        if len(expanded_content) > self.max_size_per_chunk:
            expanded_content = self._truncate_expanded_chunk(
                expanded_content_parts, current_index - start_idx
            )
        
        return expanded_content
    
    def _truncate_expanded_chunk(self, content_parts: List[str], current_chunk_idx: int) -> str:
        """
        Truncate expanded chunk content to fit within max_size_per_chunk while
        keeping the current chunk intact and balancing before/after content.
        
        Args:
            content_parts: List of content parts from overlapping chunks
            current_chunk_idx: Index of current chunk within content_parts
            
        Returns:
            Truncated expanded chunk content
        """
        current_chunk = content_parts[current_chunk_idx]
        
        # If current chunk alone exceeds max size, return just the current chunk
        if len(current_chunk) >= self.max_size_per_chunk:
            return current_chunk
        
        # Calculate available space for before and after content
        available_space = self.max_size_per_chunk - len(current_chunk) - 4  # -4 for '\n\n' separators
        half_space = available_space // 2
        
        # Get content before current chunk
        before_parts = content_parts[:current_chunk_idx]
        before_content = '\n\n'.join(before_parts) if before_parts else ''
        
        # Get content after current chunk
        after_parts = content_parts[current_chunk_idx + 1:]
        after_content = '\n\n'.join(after_parts) if after_parts else ''
        
        # Truncate before content if needed
        if len(before_content) > half_space:
            # Take the last half_space characters, trying to break at word boundary
            truncate_start = len(before_content) - half_space
            space_idx = before_content.find(' ', truncate_start)
            if space_idx != -1 and space_idx < len(before_content) - 10:  # Don't break too close to end
                before_content = before_content[space_idx:].strip()
            else:
                before_content = before_content[-half_space:].strip()
        
        # Truncate after content if needed
        remaining_space = available_space - len(before_content)
        if len(after_content) > remaining_space:
            # Take the first remaining_space characters, trying to break at word boundary
            space_idx = after_content.rfind(' ', 0, remaining_space)
            if space_idx != -1 and space_idx > 10:  # Don't break too close to start
                after_content = after_content[:space_idx].strip()
            else:
                after_content = after_content[:remaining_space].strip()
        
        # Combine all parts
        result_parts = []
        if before_content:
            result_parts.append(before_content)
        result_parts.append(current_chunk)
        if after_content:
            result_parts.append(after_content)
        
        return '\n\n'.join(result_parts)


# Example usage
if __name__ == "__main__":
    # Create sample documents
    sample_docs = [
        Document(
            page_content="This is the first document with some content that might be quite long and need chunking. " * 20,
            metadata={"source": "doc1.txt", "type": "text"}
        ),
        Document(
            page_content="This is the second document with different content that also needs processing. " * 15,
            metadata={"source": "doc2.txt", "type": "text"}
        )
    ]
    
    # Initialize chunker with parameters
    chunker = PageChunker(
        max_size_page_content=200,  # Each chunk max 200 characters
        max_size_per_chunk=500,     # Expanded chunk max 500 characters
        page_overlap=1              # Include 1 page before and after
    )
    
    # Chunk the documents
    chunked_docs = chunker.chunk_document(sample_docs)
    
    # Display results
    for i, doc in enumerate(chunked_docs):
        print(f"Chunk {i + 1}:")
        print(f"Page Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        print(f"Expanded Chunk Length: {len(doc.metadata.get('expanded_chunk', ''))}")
        print("-" * 50)
