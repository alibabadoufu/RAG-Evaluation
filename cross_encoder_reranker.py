"""
Implementation of cross-encoder reranking strategy.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder

from src.components.base.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranking strategy.
    
    This reranker uses a cross-encoder model to score query-document pairs
    for more accurate relevance assessment.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 top_k: int = 3, 
                 device: str = "cpu", 
                 **kwargs):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: The name of the cross-encoder model to use
            top_k: The number of documents to return after reranking
            device: The device to run the model on ('cpu' or 'cuda')
            **kwargs: Additional parameters
        """
        super().__init__(top_k=top_k, **kwargs)
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device)
        
    def rerank(self, query: str, documents: List[Tuple[Dict[str, Any], float]], top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank a list of retrieved documents based on their relevance to the query.
        
        Args:
            query: The query text
            documents: A list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A relevance score from the retriever
            top_k: Optional override for the number of documents to return
            
        Returns:
            A reordered list of tuples, each containing:
                - A document dictionary with 'text' and 'metadata'
                - A new relevance score (higher is more relevant)
        """
        if not documents:
            return []
            
        if top_k is None:
            top_k = self.top_k
            
        # Prepare query-document pairs for the cross-encoder
        pairs = [(query, doc[0]["text"]) for doc in documents]
        
        # Score the pairs
        scores = self.model.predict(pairs)
        
        # Create a list of (document, score) tuples
        scored_docs = [(documents[i][0], float(scores[i])) for i in range(len(documents))]
        
        # Sort by score in descending order and take top_k
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
        
        return reranked_docs
