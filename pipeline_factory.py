"""
Pipeline factory for creating RAG pipelines with different strategies.
"""
from typing import Dict, Any, Optional, Type

from src.components.base.base_chunker import BaseChunker
from src.components.base.base_embedder import BaseEmbedder
from src.components.base.base_retriever import BaseRetriever
from src.components.base.base_reranker import BaseReranker
from src.components.base.base_parser import BaseParser
from src.components.base.base_llm import BaseLLM

from src.components.chunkers.recursive_chunker import RecursiveChunker
from src.components.chunkers.semantic_chunker import SemanticChunker
from src.components.chunkers.sentence_chunker import SentenceChunker

from src.components.embedders.openai_embedder import OpenAIEmbedder
from src.components.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.components.embedders.cohere_embedder import CohereEmbedder

from src.components.retrievers.dense_retriever import DenseRetriever
from src.components.retrievers.bm25_retriever import BM25Retriever
from src.components.retrievers.hyde_retriever import HyDERetriever

from src.components.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.components.rerankers.llm_reranker import LLMReranker

from src.components.parsers.unstructured_parser import UnstructuredParser
from src.components.parsers.mistral_ocr_parser import MistralOCRParser

from src.components.llms.llama_llm import LlamaLLM
from src.components.llms.mistral_llm import MistralLLM
from src.components.llms.openai_llm import OpenAILLM

from src.pipeline.rag_pipeline import RAGPipeline


class PipelineFactory:
    """
    Factory for creating RAG pipelines with different component strategies.
    
    This factory provides methods to create and configure RAG pipelines
    with various combinations of components.
    """
    
    # Component registries
    CHUNKERS = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "sentence": SentenceChunker
    }
    
    EMBEDDERS = {
        "openai": OpenAIEmbedder,
        "sentence_transformer": SentenceTransformerEmbedder,
        "cohere": CohereEmbedder
    }
    
    RETRIEVERS = {
        "dense": DenseRetriever,
        "bm25": BM25Retriever,
        "hyde": HyDERetriever
    }
    
    RERANKERS = {
        "cross_encoder": CrossEncoderReranker,
        "llm": LLMReranker
    }
    
    PARSERS = {
        "unstructured": UnstructuredParser,
        "mistral_ocr": MistralOCRParser
    }
    
    LLMS = {
        "llama": LlamaLLM,
        "mistral": MistralLLM,
        "openai": OpenAILLM
    }
    
    @classmethod
    def create_pipeline(cls, config: Dict[str, Any]) -> RAGPipeline:
        """
        Create a RAG pipeline from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with component specifications
            
        Returns:
            A configured RAG pipeline
        """
        # Create parser
        parser_config = config.get("parser", {})
        parser_type = parser_config.pop("type", "unstructured")
        parser_class = cls.PARSERS.get(parser_type)
        if not parser_class:
            raise ValueError(f"Unknown parser type: {parser_type}")
        parser = parser_class(**parser_config)
        
        # Create chunker
        chunker_config = config.get("chunker", {})
        chunker_type = chunker_config.pop("type", "recursive")
        chunker_class = cls.CHUNKERS.get(chunker_type)
        if not chunker_class:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
        chunker = chunker_class(**chunker_config)
        
        # Create embedder
        embedder_config = config.get("embedder", {})
        embedder_type = embedder_config.pop("type", "sentence_transformer")
        embedder_class = cls.EMBEDDERS.get(embedder_type)
        if not embedder_class:
            raise ValueError(f"Unknown embedder type: {embedder_type}")
        embedder = embedder_class(**embedder_config)
        
        # Create retriever
        retriever_config = config.get("retriever", {})
        retriever_type = retriever_config.pop("type", "dense")
        retriever_class = cls.RETRIEVERS.get(retriever_type)
        if not retriever_class:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
            
        # Special case for retrievers that need embedder
        if retriever_type == "dense" or retriever_type == "hyde":
            # For HyDE, we also need an LLM
            if retriever_type == "hyde":
                hyde_llm_config = retriever_config.pop("llm", {})
                hyde_llm_type = hyde_llm_config.pop("type", "openai")
                hyde_llm_class = cls.LLMS.get(hyde_llm_type)
                if not hyde_llm_class:
                    raise ValueError(f"Unknown LLM type for HyDE: {hyde_llm_type}")
                hyde_llm = hyde_llm_class(**hyde_llm_config)
                retriever = retriever_class(embedder=embedder, llm=hyde_llm, **retriever_config)
            else:
                retriever = retriever_class(embedder=embedder, **retriever_config)
        else:
            retriever = retriever_class(**retriever_config)
        
        # Create reranker (optional)
        reranker = None
        if "reranker" in config:
            reranker_config = config.get("reranker", {})
            reranker_type = reranker_config.pop("type", "cross_encoder")
            reranker_class = cls.RERANKERS.get(reranker_type)
            if not reranker_class:
                raise ValueError(f"Unknown reranker type: {reranker_type}")
                
            # Special case for LLM reranker
            if reranker_type == "llm":
                reranker_llm_config = reranker_config.pop("llm", {})
                reranker_llm_type = reranker_llm_config.pop("type", "openai")
                reranker_llm_class = cls.LLMS.get(reranker_llm_type)
                if not reranker_llm_class:
                    raise ValueError(f"Unknown LLM type for reranker: {reranker_llm_type}")
                reranker_llm = reranker_llm_class(**reranker_llm_config)
                reranker = reranker_class(llm=reranker_llm, **reranker_config)
            else:
                reranker = reranker_class(**reranker_config)
        
        # Create LLM (optional)
        llm = None
        if "llm" in config:
            llm_config = config.get("llm", {})
            llm_type = llm_config.pop("type", "openai")
            llm_class = cls.LLMS.get(llm_type)
            if not llm_class:
                raise ValueError(f"Unknown LLM type: {llm_type}")
            llm = llm_class(**llm_config)
        
        # Create and return the pipeline
        return RAGPipeline(
            parser=parser,
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            llm=llm
        )
    
    @classmethod
    def create_from_preset(cls, preset_name: str, **overrides) -> RAGPipeline:
        """
        Create a RAG pipeline from a predefined preset with optional overrides.
        
        Args:
            preset_name: Name of the preset configuration
            **overrides: Override parameters for the preset
            
        Returns:
            A configured RAG pipeline
        """
        # Define presets
        presets = {
            "default": {
                "parser": {"type": "unstructured"},
                "chunker": {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
                "embedder": {"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
                "retriever": {"type": "dense", "similarity_metric": "cosine", "top_k": 5},
                "reranker": {"type": "cross_encoder", "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_k": 3},
                "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
            },
            "simple": {
                "parser": {"type": "unstructured"},
                "chunker": {"type": "sentence", "chunk_size": 1000, "chunk_overlap": 0},
                "embedder": {"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
                "retriever": {"type": "dense", "similarity_metric": "cosine", "top_k": 3},
                "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
            },
            "advanced": {
                "parser": {"type": "unstructured"},
                "chunker": {"type": "semantic", "chunk_size": 1000, "chunk_overlap": 200},
                "embedder": {"type": "openai", "model_name": "text-embedding-3-small"},
                "retriever": {"type": "hyde", 
                             "similarity_metric": "cosine", 
                             "top_k": 5,
                             "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.0}},
                "reranker": {"type": "llm", 
                            "top_k": 3,
                            "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.0}},
                "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
            },
            "local": {
                "parser": {"type": "unstructured"},
                "chunker": {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
                "embedder": {"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
                "retriever": {"type": "bm25", "top_k": 5},
                "llm": {"type": "llama", "model_path": "/path/to/llama/model.gguf", "temperature": 0.7}
            }
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(presets.keys())}")
            
        # Get the preset config and apply overrides
        config = presets[preset_name]
        
        # Apply overrides
        for key, value in overrides.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
                
        return cls.create_pipeline(config)
    
    @classmethod
    def register_component(cls, 
                          component_type: str, 
                          name: str, 
                          component_class: Type) -> None:
        """
        Register a new component class for use in the factory.
        
        Args:
            component_type: Type of component ('chunker', 'embedder', etc.)
            name: Name to register the component under
            component_class: The component class to register
        """
        registry_map = {
            "chunker": cls.CHUNKERS,
            "embedder": cls.EMBEDDERS,
            "retriever": cls.RETRIEVERS,
            "reranker": cls.RERANKERS,
            "parser": cls.PARSERS,
            "llm": cls.LLMS
        }
        
        if component_type not in registry_map:
            raise ValueError(f"Unknown component type: {component_type}")
            
        registry_map[component_type][name] = component_class
