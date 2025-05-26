# RAG Evaluation Framework

A comprehensive framework for evaluating different strategies in LLM RAG (Retrieval-Augmented Generation) pipelines.

## Overview

This framework allows you to evaluate and compare different components of RAG pipelines, including:

1. **Chunking strategies** - Different ways to split documents into chunks
2. **Embedding models** - Various embedding techniques for vector representation
3. **Retrieval methods** - Different approaches for retrieving relevant context (dense, BM25, HyDE)
4. **Reranking strategies** - Methods to improve retrieval precision
5. **Parsing techniques** - Different document parsing approaches
6. **LLM inference** - Various language models for generation

The framework provides evaluation metrics focused on:

- **Question groundness** - How well the response is grounded in the provided context
- **Question relevance** - How relevant the response is to the query
- **Context precision** - How much of the retrieved context is relevant
- **Context recall** - How much of the information needed is present in the context

## Features

- **Modular design** with base classes and concrete implementations
- **Extensible architecture** for adding new strategies and metrics
- **Interactive Gradio UI** for testing and evaluation
- **Comprehensive reporting** with visualizations and insights
- **Project-based evaluation** to compare performance across different projects

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-evaluation-framework.git
cd rag-evaluation-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (if using external services):
```bash
export OPENAI_API_KEY=your_openai_key
export COHERE_API_KEY=your_cohere_key
export MISTRAL_API_KEY=your_mistral_key
```

## Usage

### Running the Gradio UI

The easiest way to use the framework is through the Gradio UI:

```bash
python -m ui.gradio_app
```

This will launch a web interface with two main tabs:

1. **Chat** - For interacting with different RAG pipeline configurations
2. **Evaluation** - For running evaluations and generating reports

### Programmatic Usage

You can also use the framework programmatically:

```python
from src.pipeline.pipeline_factory import PipelineFactory
from src.evaluation.evaluator import Evaluator
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.metrics.groundness_metric import GroundnessMetric
from src.evaluation.metrics.relevance_metric import RelevanceMetric
from src.evaluation.metrics.context_precision_metric import ContextPrecisionMetric
from src.evaluation.metrics.context_recall_metric import ContextRecallMetric
from src.components.llms.openai_llm import OpenAILLM

# Create pipelines with different configurations
pipeline1 = PipelineFactory.create_from_preset("default")
pipeline2 = PipelineFactory.create_from_preset("simple")

# Add documents
pipeline1.add_document_from_file("path/to/document.pdf")
pipeline2.add_document_from_file("path/to/document.pdf")

# Create evaluation metrics
eval_llm = OpenAILLM(model_name="gpt-4o", temperature=0.0)
metrics = [
    GroundnessMetric(llm=eval_llm),
    RelevanceMetric(llm=eval_llm),
    ContextPrecisionMetric(llm=eval_llm),
    ContextRecallMetric(llm=eval_llm)
]

# Create evaluator
evaluator = Evaluator(metrics=metrics)

# Evaluate pipelines
queries = ["What is RAG?", "How does chunking affect retrieval?"]
results = evaluator.evaluate_multiple_pipelines(
    {"default": pipeline1, "simple": pipeline2},
    queries
)

# Generate reports
report_generator = ReportGenerator(results)
report_generator.generate_pdf_report("results/report.pdf")
report_generator.generate_interactive_dashboard("results/dashboard.html")
```

## Configuration

The framework supports various configuration options for each component:

### Chunking Strategies

- **RecursiveChunker** - Splits documents based on hierarchical structure
- **SemanticChunker** - Splits documents based on semantic meaning
- **SentenceChunker** - Splits documents by sentences

### Embedding Models

- **OpenAIEmbedder** - Uses OpenAI's embedding models
- **SentenceTransformerEmbedder** - Uses sentence-transformers models
- **CohereEmbedder** - Uses Cohere's embedding models

### Retrieval Methods

- **DenseRetriever** - Vector similarity search with configurable distance metrics
- **BM25Retriever** - Lexical search using BM25 algorithm
- **HyDERetriever** - Hypothetical Document Embedding for improved retrieval

### Reranking Strategies

- **CrossEncoderReranker** - Uses cross-encoder models for reranking
- **LLMReranker** - Uses LLMs to rerank retrieved documents

### Parsing Techniques

- **UnstructuredParser** - Uses the unstructured library for document parsing
- **MistralOCRParser** - Uses Mistral's OCR capabilities for image-based documents

### LLM Models

- **LlamaLLM** - Uses Llama models via llama-cpp-python
- **MistralLLM** - Uses Mistral AI's models
- **OpenAILLM** - Uses OpenAI's models

## Extending the Framework

The framework is designed to be easily extensible. To add a new component:

1. Create a new class that inherits from the appropriate base class
2. Implement the required methods
3. Register the component with the factory

Example of adding a new chunker:

```python
from src.components.base.base_chunker import BaseChunker

class MyCustomChunker(BaseChunker):
    def __init__(self, chunk_size=1000, chunk_overlap=200, **kwargs):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        
    def chunk_document(self, document, metadata=None):
        # Implement your chunking logic here
        chunks = []
        # ...
        return chunks

# Register with factory
from src.pipeline.pipeline_factory import PipelineFactory
PipelineFactory.register_component("chunker", "my_custom", MyCustomChunker)
```

## Project Structure

```
rag_evaluation_framework/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── strategy_configs.py
│   └── evaluation_configs.py
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── base/
│   │   ├── chunkers/
│   │   ├── embedders/
│   │   ├── retrievers/
│   │   ├── rerankers/
│   │   ├── parsers/
│   │   └── llms/
│   ├── pipeline/
│   ├── evaluation/
│   ├── data/
│   └── utils/
├── ui/
│   ├── gradio_app.py
│   └── components/
├── tests/
├── results/
└── examples/
```

## Examples

See the `examples/` directory for sample usage:

- `example_evaluation.py` - Example of evaluating multiple pipeline configurations
- `example_config.yaml` - Example configuration file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
