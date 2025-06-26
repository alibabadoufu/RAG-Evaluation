# RAG Evaluation Framework Documentation

## Introduction

The Retrieval-Augmented Generation (RAG) Evaluation Framework is a comprehensive toolkit designed to evaluate and compare different RAG pipeline strategies. This framework enables researchers and engineers to systematically assess the performance of various components in a RAG system, including chunking strategies, embedding models, retrieval methods, reranking approaches, parsing techniques, and language models.

This documentation provides a detailed overview of the framework's architecture, components, evaluation methodology, and usage guidelines. It serves as a reference for both new users getting started with the framework and experienced users looking to extend its functionality.

## Framework Architecture

### Overview

The RAG Evaluation Framework follows a modular architecture based on the Strategy design pattern. This design allows for flexible composition of different components to create customized RAG pipelines. The framework consists of the following main modules:

1. **Components Module**: Contains the core building blocks of RAG pipelines
2. **Pipeline Module**: Manages the assembly and execution of RAG pipelines
3. **Evaluation Module**: Handles the assessment of pipeline performance
4. **UI Module**: Provides a graphical interface for interacting with the framework

### Component Hierarchy

The framework implements a hierarchical structure for components:

```
BaseComponent
├── BaseChunker
│   ├── RecursiveChunker
│   ├── SemanticChunker
│   └── SentenceChunker
├── BaseEmbedder
│   ├── OpenAIEmbedder
│   ├── SentenceTransformerEmbedder
│   └── CohereEmbedder
├── BaseRetriever
│   ├── DenseRetriever
│   ├── BM25Retriever
│   └── HyDERetriever
├── BaseReranker
│   ├── CrossEncoderReranker
│   └── LLMReranker
├── BaseParser
│   ├── UnstructuredParser
│   └── MistralOCRParser
└── BaseLLM
    ├── LlamaLLM
    ├── MistralLLM
    └── OpenAILLM
```

This hierarchy enables easy extension of the framework with new component implementations while maintaining a consistent interface.

## Components in Detail

### Chunking Strategies

Chunking is the process of breaking documents into smaller, manageable pieces for embedding and retrieval. The framework supports the following chunking strategies:

1. **RecursiveChunker**: Recursively splits documents into chunks based on content structure, maintaining semantic coherence.
   - Parameters: chunk_size, chunk_overlap
   - Best for: Hierarchical documents with clear section boundaries

2. **SemanticChunker**: Creates chunks based on semantic similarity, ensuring related content stays together.
   - Parameters: chunk_size, chunk_overlap, similarity_threshold
   - Best for: Documents with varying content density and topic shifts

3. **SentenceChunker**: Splits documents at sentence boundaries to preserve linguistic structure.
   - Parameters: chunk_size, chunk_overlap
   - Best for: Narrative text where sentence context is important

### Embedding Strategies

Embeddings convert text chunks into vector representations for efficient retrieval. The framework supports:

1. **OpenAIEmbedder**: Uses OpenAI's embedding models (e.g., text-embedding-ada-002).
   - Parameters: model_name, dimensions
   - Best for: High-quality embeddings with state-of-the-art performance

2. **SentenceTransformerEmbedder**: Leverages Sentence Transformers library for embeddings.
   - Parameters: model_name
   - Best for: Local deployment with good performance-to-resource ratio

3. **CohereEmbedder**: Uses Cohere's embedding models.
   - Parameters: model_name
   - Best for: Multilingual content and specialized domains

### Retrieval Strategies

Retrieval strategies determine how relevant chunks are selected based on a query. The framework includes:

1. **DenseRetriever**: Performs similarity search using dense vector representations.
   - Parameters: similarity_metric (cosine, dot_product, euclidean), top_k
   - Best for: Semantic search capabilities

2. **BM25Retriever**: Uses the BM25 algorithm for keyword-based retrieval.
   - Parameters: top_k, b, k1
   - Best for: Keyword-heavy queries and exact matching

3. **HyDERetriever**: Implements Hypothetical Document Embeddings, generating a hypothetical answer before retrieval.
   - Parameters: top_k, llm_model
   - Best for: Complex queries requiring inference

### Reranking Strategies

Reranking improves retrieval results by applying more sophisticated models to reorder the initial results:

1. **CrossEncoderReranker**: Uses cross-encoder models to assess query-document relevance.
   - Parameters: model_name, top_k
   - Best for: High-precision ranking requirements

2. **LLMReranker**: Leverages language models to rerank results based on relevance to the query.
   - Parameters: model_name, temperature, top_k
   - Best for: Complex relevance judgments requiring reasoning

### Parsing Strategies

Parsing strategies handle the extraction of text from various document formats:

1. **UnstructuredParser**: Uses the unstructured library to extract text from various file formats.
   - Parameters: extract_images, extract_tables
   - Best for: Common document formats (PDF, DOCX, HTML)

2. **MistralOCRParser**: Combines OCR with Mistral's capabilities for image-heavy documents.
   - Parameters: image_resolution, language
   - Best for: Scanned documents and images containing text

### LLM Strategies

Language Model strategies determine how the final response is generated:

1. **LlamaLLM**: Uses Llama models for response generation.
   - Parameters: model_name, temperature, max_tokens
   - Best for: Local deployment with good performance

2. **MistralLLM**: Uses Mistral models for response generation.
   - Parameters: model_name, temperature, max_tokens
   - Best for: Efficient reasoning and instruction following

3. **OpenAILLM**: Uses OpenAI models for response generation.
   - Parameters: model_name, temperature, max_tokens
   - Best for: State-of-the-art performance and reliability

## Evaluation Metrics

The framework provides several metrics to evaluate RAG pipeline performance:

### Question Groundness

Measures how well the generated answer is grounded in the retrieved context.

- **Implementation**: Uses an LLM to assess if the answer contains information not present in the context.
- **Score Range**: 0 (ungrounded) to 1 (fully grounded)
- **Use Case**: Detecting hallucinations and unsupported claims

### Question Relevance

Evaluates how relevant the generated answer is to the original question.

- **Implementation**: Uses an LLM to assess if the answer addresses the question's intent.
- **Score Range**: 0 (irrelevant) to 1 (highly relevant)
- **Use Case**: Ensuring answers stay on topic and address user queries

### Context Precision

Measures the proportion of retrieved context that is relevant to the question.

- **Implementation**: Uses an LLM to assess what percentage of retrieved chunks contain information relevant to the question.
- **Score Range**: 0 (no relevant chunks) to 1 (all chunks relevant)
- **Use Case**: Optimizing retrieval efficiency and reducing noise

### Context Recall

Evaluates whether the retrieved context contains all the information needed to answer the question.

- **Implementation**: Uses an LLM to assess if the context contains all necessary information.
- **Score Range**: 0 (missing critical information) to 1 (complete information)
- **Use Case**: Ensuring comprehensive information retrieval

## Pipeline Factory

The Pipeline Factory is a key architectural component that enables flexible composition of RAG pipelines:

### Design Pattern

The framework uses the Factory design pattern to create pipelines from configurations:

1. **Component Registration**: Each component type is registered with the factory
2. **Configuration-Based Creation**: Pipelines are created from configuration dictionaries
3. **Preset Management**: Common configurations can be saved as presets

### Configuration Format

Pipelines are configured using a structured format:

```
{
  "parser": {"type": "unstructured", ...params},
  "chunker": {"type": "recursive", ...params},
  "embedder": {"type": "sentence_transformer", ...params},
  "retriever": {"type": "dense", ...params},
  "reranker": {"type": "cross_encoder", ...params},
  "llm": {"type": "openai", ...params}
}
```

This format allows for easy serialization, sharing, and modification of pipeline configurations.

## Evaluation Workflow

### Step 1: Pipeline Configuration

1. Define the components to be used in each pipeline
2. Set appropriate parameters for each component
3. Create multiple pipeline configurations for comparison

### Step 2: Document Processing

1. Load documents into the pipelines
2. Documents are parsed according to the selected parsing strategy
3. Documents are chunked according to the selected chunking strategy
4. Chunks are embedded according to the selected embedding strategy

### Step 3: Query Execution

1. Define evaluation queries
2. For each query and pipeline:
   - Retrieve relevant chunks using the retrieval strategy
   - Rerank chunks if a reranking strategy is specified
   - Generate a response using the LLM strategy

### Step 4: Evaluation

1. For each query, pipeline, and metric:
   - Calculate the metric score
   - Store individual and aggregate results
2. Compare results across pipelines and metrics

### Step 5: Reporting

1. Generate visualizations comparing pipeline performance
2. Create detailed reports with analysis and recommendations
3. Export results in various formats (PDF, HTML, Excel, JSON)

## Using the Framework

### Command Line Interface

The framework can be used via command line for batch processing and integration into workflows:

1. **Configuration**: Create a YAML or JSON configuration file
2. **Execution**: Run the evaluation script with the configuration file
3. **Results**: View results in the specified output directory

### Gradio UI

The framework provides a Gradio-based UI for interactive exploration:

#### Chat Interface

1. Select a pipeline configuration
2. Upload documents for processing
3. Chat with the RAG system to test its responses

#### Evaluation Interface

1. Select multiple pipeline configurations to compare
2. Upload test documents
3. Enter evaluation queries
4. Run evaluation across all metrics
5. Generate comparison plots and visualizations
6. Export comprehensive reports

### Extending the Framework

The framework is designed to be easily extensible:

1. **Adding New Components**:
   - Create a new class inheriting from the appropriate base class
   - Implement the required methods
   - Register the component with the factory

2. **Adding New Metrics**:
   - Create a new class inheriting from BaseMetric
   - Implement the evaluate and evaluate_batch methods
   - Add the metric to the evaluator

3. **Creating Custom Pipelines**:
   - Use the PipelineFactory to create custom pipelines
   - Or manually instantiate and connect components

## Best Practices

### Pipeline Configuration

1. **Match components to use cases**: Different components excel in different scenarios
2. **Balance performance and efficiency**: Consider computational requirements
3. **Start with presets**: Use preset configurations as starting points

### Evaluation Design

1. **Use diverse query sets**: Include various query types and complexities
2. **Include ground truth when possible**: Enables more accurate evaluation
3. **Consider multiple metrics**: Different metrics capture different aspects of performance

### Reporting and Analysis

1. **Compare across dimensions**: Analyze performance by component type
2. **Look for patterns**: Identify which strategies work best together
3. **Consider trade-offs**: Balance metrics based on application priorities

## Common Workflows

### Chunking Strategy Optimization

1. Configure multiple pipelines with different chunking strategies
2. Keep other components consistent
3. Evaluate with context precision and recall metrics
4. Analyze which chunking strategy provides the best balance

### Retrieval Method Comparison

1. Configure pipelines with different retrieval methods
2. Test with queries of varying complexity
3. Focus on relevance and recall metrics
4. Analyze performance across query types

### End-to-End Pipeline Optimization

1. Define baseline and experimental pipelines
2. Run comprehensive evaluation across all metrics
3. Generate detailed reports comparing performance
4. Identify the optimal configuration for your use case

## Troubleshooting

### Common Issues

1. **Low groundness scores**: Check for LLM hallucination or insufficient context
2. **Poor retrieval performance**: Review embedding and retrieval strategy compatibility
3. **Slow evaluation**: Consider batch processing or sampling for large datasets

### Performance Optimization

1. **Caching**: Enable embedding and retrieval caching
2. **Batch processing**: Process documents and queries in batches
3. **Resource allocation**: Allocate appropriate resources for embedding and LLM components

## Conclusion

The RAG Evaluation Framework provides a comprehensive solution for evaluating and optimizing RAG pipelines. By leveraging its modular architecture and extensive metrics, users can systematically improve their RAG systems and make data-driven decisions about component selection.

The framework's flexibility allows it to evolve with the rapidly advancing field of retrieval-augmented generation, providing a solid foundation for research and production applications alike.

## Appendix

### Glossary

- **RAG**: Retrieval-Augmented Generation, a technique that enhances LLM outputs with retrieved information
- **Chunking**: The process of dividing documents into smaller pieces for retrieval
- **Embedding**: The process of converting text into vector representations
- **Retrieval**: The process of finding relevant information based on a query
- **Reranking**: The process of refining retrieval results for improved relevance
- **Groundness**: The degree to which generated content is supported by retrieved context
- **Hallucination**: When an LLM generates information not present in the provided context

### References

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. Gao, L., et al. (2023). "Precise Zero-Shot Dense Retrieval without Relevance Labels"
3. Izacard, G., et al. (2022). "Few-shot Learning with Retrieval Augmented Language Models"
