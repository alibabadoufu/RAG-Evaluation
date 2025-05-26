"""
Basic test for the chunker components.
"""
import sys
import os
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.components.chunkers.recursive_chunker import RecursiveChunker
from src.components.chunkers.semantic_chunker import SemanticChunker
from src.components.chunkers.sentence_chunker import SentenceChunker


class TestChunkers(unittest.TestCase):
    """Test cases for chunker components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_document = """
# Sample Document

## Introduction

This is a sample document for testing chunking strategies.
It contains multiple paragraphs and sections to test different chunking approaches.

## Section 1

This is the first section of the document.
It has multiple sentences that should be processed correctly.
Each chunking strategy should handle this content appropriately.

## Section 2

The second section contains more content.
This will help test how chunkers handle different sections.

### Subsection 2.1

This is a subsection with even more detailed content.
The recursive chunker should recognize this hierarchical structure.
"""
        
        self.recursive_chunker = RecursiveChunker(chunk_size=200, chunk_overlap=50)
        self.semantic_chunker = SemanticChunker(chunk_size=200, chunk_overlap=50)
        self.sentence_chunker = SentenceChunker(chunk_size=200, chunk_overlap=50)
    
    def test_recursive_chunker(self):
        """Test the recursive chunker."""
        chunks = self.recursive_chunker.chunk_document(self.test_document)
        
        # Basic validation
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Check chunk structure
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('metadata', chunk)
            self.assertIsInstance(chunk['text'], str)
            self.assertIsInstance(chunk['metadata'], dict)
            
        # Check that headers are preserved
        headers_found = False
        for chunk in chunks:
            if '# Sample Document' in chunk['text'] or '## Introduction' in chunk['text']:
                headers_found = True
                break
        self.assertTrue(headers_found)
    
    def test_semantic_chunker(self):
        """Test the semantic chunker."""
        chunks = self.semantic_chunker.chunk_document(self.test_document)
        
        # Basic validation
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Check chunk structure
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('metadata', chunk)
            self.assertIsInstance(chunk['text'], str)
            self.assertIsInstance(chunk['metadata'], dict)
            
        # Check that paragraphs are preserved
        for chunk in chunks:
            # Each chunk should contain at least one complete sentence
            self.assertTrue(any(s.strip().endswith('.') for s in chunk['text'].split('\n') if s.strip()))
    
    def test_sentence_chunker(self):
        """Test the sentence chunker."""
        chunks = self.sentence_chunker.chunk_document(self.test_document)
        
        # Basic validation
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        # Check chunk structure
        for chunk in chunks:
            self.assertIn('text', chunk)
            self.assertIn('metadata', chunk)
            self.assertIsInstance(chunk['text'], str)
            self.assertIsInstance(chunk['metadata'], dict)
            
        # Check that sentences are preserved
        for chunk in chunks:
            # Each chunk should end with a sentence-ending punctuation
            self.assertTrue(chunk['text'].strip().endswith('.') or 
                           chunk['text'].strip().endswith('?') or 
                           chunk['text'].strip().endswith('!'))
    
    def test_chunk_size_respect(self):
        """Test that chunkers respect the chunk size parameter."""
        # Create chunkers with a very small chunk size
        small_recursive = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        small_semantic = SemanticChunker(chunk_size=50, chunk_overlap=0)
        small_sentence = SentenceChunker(chunk_size=50, chunk_overlap=0)
        
        # Get chunks
        recursive_chunks = small_recursive.chunk_document(self.test_document)
        semantic_chunks = small_semantic.chunk_document(self.test_document)
        sentence_chunks = small_sentence.chunk_document(self.test_document)
        
        # Check that most chunks are smaller than the chunk size
        # Note: Some chunkers might create slightly larger chunks in certain cases
        for chunks in [recursive_chunks, semantic_chunks, sentence_chunks]:
            small_chunks = sum(1 for chunk in chunks if len(chunk['text']) <= 60)  # Allow some flexibility
            self.assertGreater(small_chunks, len(chunks) * 0.7)  # At least 70% should be small


if __name__ == '__main__':
    unittest.main()
