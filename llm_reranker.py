"""
Implementation of LLM-based reranking strategy.
"""
from typing import List, Dict, Any, Tuple, Optional

from src.components.base.base_reranker import BaseReranker
from src.components.base.base_llm import BaseLLM


class LLMReranker(BaseReranker):
    """
    LLM-based reranking strategy.
    
    This reranker uses an LLM to score query-document pairs for more accurate
    relevance assessment.
    """
    
    def __init__(self, llm: BaseLLM, top_k: int = 3, **kwargs):
        """
        Initialize the LLM reranker.
        
        Args:
            llm: The LLM to use for reranking
            top_k: The number of documents to return after reranking
            **kwargs: Additional parameters
        """
        super().__init__(top_k=top_k, **kwargs)
        self.llm = llm
        
        # Template for reranking prompt
        self.rerank_prompt_template = """
        On a scale of 0 to 10, rate how relevant the following document is to the query.
        Only respond with a number between 0 and 10, where 0 means completely irrelevant and 10 means perfectly relevant.
        
        Query: {query}
        
        Document: {document}
        
        Relevance score (0-10):
        """
        
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
            
        # Score each document using the LLM
        scored_docs = []
        for doc, _ in documents:
            prompt = self.rerank_prompt_template.format(
                query=query,
                document=doc["text"]
            )
            
            # Get the score from the LLM
            response = self.llm.generate(prompt)
            
            try:
                # Extract the numeric score from the response
                score = float(response.strip())
                # Normalize to [0, 1] range
                normalized_score = score / 10.0
                scored_docs.append((doc, normalized_score))
            except ValueError:
                # If we can't parse a score, use a default low score
                scored_docs.append((doc, 0.0))
        
        # Sort by score in descending order and take top_k
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_k]
        
        return reranked_docs
