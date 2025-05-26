"""
Basic test for the pipeline and factory components.
"""
import sys
import os
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.pipeline_factory import PipelineFactory
from src.pipeline.rag_pipeline import RAGPipeline
from src.components.chunkers.recursive_chunker import RecursiveChunker
from src.components.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
from src.components.retrievers.dense_retriever import DenseRetriever
from src.components.parsers.unstructured_parser import UnstructuredParser
from src.components.llms.openai_llm import OpenAILLM


class TestPipeline(unittest.TestCase):
    """Test cases for pipeline and factory components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_document = {
            "text": """
# Sample Document

This is a sample document for testing the RAG pipeline.
It contains information about retrieval-augmented generation.

RAG combines retrieval with generation to create more accurate responses.
            """,
            "metadata": {
                "source": "test",
                "id": "doc1"
            }
        }
        
        # Create components for manual pipeline creation
        self.parser = UnstructuredParser()
        self.chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        self.embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        self.retriever = DenseRetriever(embedder=self.embedder, similarity_metric="cosine", top_k=3)
        self.llm = OpenAILLM(model_name="gpt-4o", temperature=0.7)
    
    def test_manual_pipeline_creation(self):
        """Test creating a pipeline manually."""
        pipeline = RAGPipeline(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            retriever=self.retriever,
            llm=self.llm
        )
        
        # Basic validation
        self.assertIsInstance(pipeline, RAGPipeline)
        self.assertEqual(pipeline.parser, self.parser)
        self.assertEqual(pipeline.chunker, self.chunker)
        self.assertEqual(pipeline.embedder, self.embedder)
        self.assertEqual(pipeline.retriever, self.retriever)
        self.assertEqual(pipeline.llm, self.llm)
    
    def test_factory_pipeline_creation(self):
        """Test creating a pipeline using the factory."""
        # Create a pipeline using the factory
        pipeline = PipelineFactory.create_pipeline({
            "parser": {"type": "unstructured"},
            "chunker": {"type": "recursive", "chunk_size": 200, "chunk_overlap": 50},
            "embedder": {"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
            "retriever": {"type": "dense", "similarity_metric": "cosine", "top_k": 3},
            "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
        })
        
        # Basic validation
        self.assertIsInstance(pipeline, RAGPipeline)
        self.assertIsInstance(pipeline.parser, UnstructuredParser)
        self.assertIsInstance(pipeline.chunker, RecursiveChunker)
        self.assertIsInstance(pipeline.embedder, SentenceTransformerEmbedder)
        self.assertIsInstance(pipeline.retriever, DenseRetriever)
        self.assertIsInstance(pipeline.llm, OpenAILLM)
    
    def test_factory_preset_creation(self):
        """Test creating a pipeline using factory presets."""
        # Create a pipeline using a preset
        pipeline = PipelineFactory.create_from_preset("default")
        
        # Basic validation
        self.assertIsInstance(pipeline, RAGPipeline)
        self.assertIsInstance(pipeline.parser, UnstructuredParser)
        self.assertIsInstance(pipeline.chunker, RecursiveChunker)
        self.assertIsInstance(pipeline.embedder, SentenceTransformerEmbedder)
        self.assertIsInstance(pipeline.retriever, DenseRetriever)
        self.assertIsInstance(pipeline.llm, OpenAILLM)
    
    def test_pipeline_document_addition(self):
        """Test adding documents to a pipeline."""
        pipeline = RAGPipeline(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            retriever=self.retriever,
            llm=self.llm
        )
        
        # Add a document
        pipeline.add_documents([self.test_document])
        
        # Check that chunks were created
        self.assertGreater(len(pipeline.chunks), 0)
    
    def test_pipeline_retrieval(self):
        """Test document retrieval in a pipeline."""
        pipeline = RAGPipeline(
            parser=self.parser,
            chunker=self.chunker,
            embedder=self.embedder,
            retriever=self.retriever,
            llm=self.llm
        )
        
        # Add a document
        pipeline.add_documents([self.test_document])
        
        # Retrieve documents
        retrieved_docs = pipeline.retrieve("What is RAG?")
        
        # Basic validation
        self.assertIsInstance(retrieved_docs, list)
        self.assertLessEqual(len(retrieved_docs), 3)  # top_k=3
        
        # Check structure of retrieved documents
        for doc, score in retrieved_docs:
            self.assertIn('text', doc)
            self.assertIn('metadata', doc)
            self.assertIsInstance(score, float)
    
    def test_component_registration(self):
        """Test registering a custom component with the factory."""
        # Define a custom chunker
        class CustomChunker(RecursiveChunker):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.custom_property = True
        
        # Register the custom chunker
        PipelineFactory.register_component("chunker", "custom", CustomChunker)
        
        # Create a pipeline using the custom chunker
        pipeline = PipelineFactory.create_pipeline({
            "parser": {"type": "unstructured"},
            "chunker": {"type": "custom", "chunk_size": 200, "chunk_overlap": 50},
            "embedder": {"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
            "retriever": {"type": "dense", "similarity_metric": "cosine", "top_k": 3},
            "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
        })
        
        # Validate the custom chunker was used
        self.assertIsInstance(pipeline.chunker, CustomChunker)
        self.assertTrue(pipeline.chunker.custom_property)


if __name__ == '__main__':
    unittest.main()
