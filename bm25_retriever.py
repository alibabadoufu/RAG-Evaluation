"""
Implementation of BM25 retrieval strategy.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import re

from src.components.base.base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    """
    BM25 retrieval strategy using lexical matching.
    
    This retriever uses the BM25 algorithm to find relevant documents based on
    term frequency and inverse document frequency.
    """
    
    def __init__(self, top_k: int = 5, tokenizer=None, **kwargs):
        """
        Initialize the BM25 retriever.
        
        Args:
            top_k: The number of documents to retrieve
            tokenizer: Optional custom tokenizer function
            **kwargs: Additional parameters
        """
        super().__init__(top_k=top_k, **kwargs)
        self.tokenizer = tokenizer or self._default_tokenizer
        self.bm25 = None
        self.documents = []
        self.corpus = []
        
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default tokenization function.
        
        Args:
            text: The text to tokenize
            
        Returns:
            A list of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()
        
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retrieval index.
        
        Args:
            documents: A list of document dictionaries, each containing at minimum:
                - 'text': The document text
                - 'metadata': Document metadata
        """
        # Store the documents
        self.documents.extend(documents)
        
        # Tokenize the documents
        tokenized_docs = [self.tokenizer(doc["text"]) for doc in documents]
        self.corpus.extend(tokenized_docs)
        
        # Create or update the BM25 index
        self.bm25 = BM25Okapi(self.corpus)
        
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents for a given query.
        
        Args:
            query: The query text
            top_k: Optional override for the number of documents to retrieve
            
        Returns:
            A list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A relevance score (higher is more relevant)
        """
        if not self.documents or self.bm25 is None:
            return []
            
        if top_k is None:
            top_k = self.top_k
            
        # Tokenize the query
        tokenized_query = self.tokenizer(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get the indices of the top-k highest scoring documents
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return the top-k documents with their scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                results.append((self.documents[idx], float(scores[idx])))
            
        return results
