"""
RAG Pipeline implementation.
"""
from typing import List, Dict, Any, Optional, Tuple

from src.components.base.base_chunker import BaseChunker
from src.components.base.base_embedder import BaseEmbedder
from src.components.base.base_retriever import BaseRetriever
from src.components.base.base_reranker import BaseReranker
from src.components.base.base_parser import BaseParser
from src.components.base.base_llm import BaseLLM


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.
    
    This pipeline combines document processing, retrieval, and generation
    components to create a complete RAG system.
    """
    
    def __init__(self, 
                 parser: BaseParser,
                 chunker: BaseChunker,
                 embedder: BaseEmbedder,
                 retriever: BaseRetriever,
                 reranker: Optional[BaseReranker] = None,
                 llm: BaseLLM = None,
                 **kwargs):
        """
        Initialize the RAG pipeline.
        
        Args:
            parser: Component for parsing documents
            chunker: Component for chunking documents
            embedder: Component for embedding text
            retriever: Component for retrieving relevant documents
            reranker: Optional component for reranking retrieved documents
            llm: Component for generating responses
            **kwargs: Additional parameters
        """
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.config = kwargs
        
        # Initialize document store
        self.documents = []
        self.chunks = []
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the pipeline.
        
        Args:
            documents: A list of document dictionaries, each containing at minimum:
                - 'text': The document text
                - 'metadata': Document metadata
        """
        self.documents.extend(documents)
        
        # Process documents: chunk and index
        for doc in documents:
            chunks = self.chunker.chunk_document(doc["text"], doc["metadata"])
            self.chunks.extend(chunks)
            
        # Add chunks to retriever
        self.retriever.add_documents(self.chunks)
        
    def add_document_from_file(self, file_path: str) -> None:
        """
        Parse and add a document from a file.
        
        Args:
            file_path: Path to the document file
        """
        # Parse the document
        doc = self.parser.parse_file(file_path)
        
        # Add to pipeline
        self.add_documents([doc])
        
    def add_documents_from_directory(self, directory_path: str) -> None:
        """
        Parse and add all documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
        """
        # Parse all documents in the directory
        docs = self.parser.parse_directory(directory_path)
        
        # Add to pipeline
        self.add_documents(docs)
        
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query text
            top_k: Optional override for the number of documents to retrieve
            
        Returns:
            A list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A relevance score
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(query, top_k)
        
        # Rerank if a reranker is available
        if self.reranker and retrieved_docs:
            retrieved_docs = self.reranker.rerank(query, retrieved_docs, top_k)
            
        return retrieved_docs
        
    def generate(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response for a query using the RAG pipeline.
        
        Args:
            query: The query text
            top_k: Optional override for the number of documents to retrieve
            
        Returns:
            A dictionary containing:
                - 'response': The generated response
                - 'context': The context documents used for generation
                - 'query': The original query
        """
        if not self.llm:
            raise ValueError("LLM component is required for generation")
            
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        # Extract just the document dictionaries for the LLM
        context_docs = [doc for doc, _ in retrieved_docs]
        
        # Generate response
        response = self.llm.generate(query, context_docs)
        
        return {
            'response': response,
            'context': context_docs,
            'query': query
        }
        
    def chat(self, 
            messages: List[Dict[str, str]], 
            top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a response for a chat history using the RAG pipeline.
        
        Args:
            messages: List of message dictionaries, each with 'role' (user/assistant/system) and 'content'
            top_k: Optional override for the number of documents to retrieve
            
        Returns:
            A dictionary containing:
                - 'response': The generated response
                - 'context': The context documents used for generation
                - 'messages': The chat history
        """
        if not self.llm:
            raise ValueError("LLM component is required for chat")
            
        # Extract the last user query
        last_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
                
        if not last_user_message:
            raise ValueError("No user message found in chat history")
            
        # Retrieve relevant documents based on the last user query
        retrieved_docs = self.retrieve(last_user_message, top_k)
        
        # Extract just the document dictionaries for the LLM
        context_docs = [doc for doc, _ in retrieved_docs]
        
        # Generate response
        response = self.llm.generate_with_chat_history(messages, context_docs)
        
        return {
            'response': response,
            'context': context_docs,
            'messages': messages
        }
        
    def __str__(self) -> str:
        """String representation of the pipeline."""
        components = [
            f"Parser: {self.parser.__class__.__name__}",
            f"Chunker: {self.chunker.__class__.__name__}",
            f"Embedder: {self.embedder.__class__.__name__}",
            f"Retriever: {self.retriever.__class__.__name__}"
        ]
        
        if self.reranker:
            components.append(f"Reranker: {self.reranker.__class__.__name__}")
            
        if self.llm:
            components.append(f"LLM: {self.llm.__class__.__name__}")
            
        return f"RAGPipeline(\n  " + "\n  ".join(components) + "\n)"
