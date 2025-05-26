# RAG Evaluation Framework - Todo List

## Project Setup
- [x] Initialize project structure with all required directories

## Base Classes Implementation
- [x] Implement base classes for pipeline components
  - [x] Implement BaseChunker
  - [x] Implement BaseEmbedder
  - [x] Implement BaseRetriever
  - [x] Implement BaseReranker
  - [x] Implement BaseParser
  - [x] Implement BaseLLM
- [x] Implement base classes for evaluation metrics
  - [x] Implement BaseMetric

## Concrete Implementations
- [x] Implement chunking strategies
  - [x] Implement RecursiveChunker
  - [x] Implement SemanticChunker
  - [x] Implement SentenceChunker
- [x] Implement embedding strategies
  - [x] Implement OpenAIEmbedder
  - [x] Implement SentenceTransformerEmbedder
  - [x] Implement CohereEmbedder
- [x] Implement retrieval strategies
  - [x] Implement DenseRetriever
  - [x] Implement BM25Retriever
  - [x] Implement HyDERetriever
- [x] Implement reranking strategies
  - [x] Implement CrossEncoderReranker
  - [x] Implement LLMReranker
- [x] Implement parsing strategies
  - [x] Implement UnstructuredParser
  - [x] Implement MistralOCRParser
- [x] Implement LLM strategies
  - [x] Implement LlamaLLM
  - [x] Implement MistralLLM
  - [x] Implement OpenAILLM
- [x] Implement evaluation metrics
  - [x] Implement GroundnessMetric
  - [x] Implement RelevanceMetric
  - [x] Implement ContextPrecisionMetric
  - [x] Implement ContextRecallMetric

## Pipeline and Factory Implementation
- [x] Implement RAG pipeline
- [x] Implement pipeline factory

## Evaluation and Reporting
- [x] Implement evaluator
- [x] Implement report generator

## UI Implementation
- [x] Implement Gradio UI
  - [x] Implement chat interface
  - [x] Implement evaluation interface

## Documentation and Examples
- [x] Write README.md
- [x] Create example files

## Testing and Validation
- [x] Write tests for components
