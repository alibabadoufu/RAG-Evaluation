"""
Implementation of dense retrieval strategy.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

from src.components.base.base_retriever import BaseRetriever
from src.components.base.base_embedder import BaseEmbedder


class DenseRetriever(BaseRetriever):
    """
    Dense retrieval strategy using vector embeddings.
    
    This retriever uses vector embeddings to find semantically similar documents.
    """
    
    def __init__(self, embedder: BaseEmbedder, 
                 similarity_metric: str = "cosine", 
                 top_k: int = 5, 
                 **kwargs):
        """
        Initialize the dense retriever.
        
        Args:
            embedder: The embedder to use for generating document and query embeddings
            similarity_metric: The metric to use for computing similarity ('cosine', 'euclidean', 'manhattan')
            top_k: The number of documents to retrieve
            **kwargs: Additional parameters
        """
        super().__init__(top_k=top_k, **kwargs)
        self.embedder = embedder
        self.similarity_metric = similarity_metric
        self.document_embeddings = []
        self.documents = []
        
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
        
        # Generate embeddings for the documents
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedder.embed_batch(texts)
        self.document_embeddings.extend(embeddings)
        
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
        if not self.documents:
            return []
            
        if top_k is None:
            top_k = self.top_k
            
        # Generate embedding for the query
        query_embedding = self.embedder.embed_text(query)
        
        # Compute similarities between query and documents
        document_embeddings_array = np.array(self.document_embeddings)
        query_embedding_array = np.array(query_embedding).reshape(1, -1)
        
        if self.similarity_metric == "cosine":
            similarities = cosine_similarity(query_embedding_array, document_embeddings_array)[0]
        elif self.similarity_metric == "euclidean":
            # Convert distances to similarities (smaller distance = higher similarity)
            distances = euclidean_distances(query_embedding_array, document_embeddings_array)[0]
            similarities = 1 / (1 + distances)  # Transform to [0, 1] range
        elif self.similarity_metric == "manhattan":
            # Convert distances to similarities (smaller distance = higher similarity)
            distances = manhattan_distances(query_embedding_array, document_embeddings_array)[0]
            similarities = 1 / (1 + distances)  # Transform to [0, 1] range
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
            
        # Get the indices of the top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return the top-k documents with their similarity scores
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))
            
        return results
