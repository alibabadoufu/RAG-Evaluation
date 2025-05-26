"""
Example evaluation script for the RAG evaluation framework.

This script demonstrates how to:
1. Create multiple pipeline configurations
2. Add documents to the pipelines
3. Evaluate the pipelines on a set of queries
4. Generate comparison reports and visualizations
"""
import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline.pipeline_factory import PipelineFactory
from src.evaluation.evaluator import Evaluator
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.metrics.groundness_metric import GroundnessMetric
from src.evaluation.metrics.relevance_metric import RelevanceMetric
from src.evaluation.metrics.context_precision_metric import ContextPrecisionMetric
from src.evaluation.metrics.context_recall_metric import ContextRecallMetric
from src.components.llms.openai_llm import OpenAILLM


def main():
    # Create output directories
    results_dir = project_root / "results"
    reports_dir = results_dir / "reports"
    logs_dir = results_dir / "logs"
    
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define document paths
    documents_dir = project_root / "examples" / "documents"
    document_paths = [
        documents_dir / "sample1.pdf",
        documents_dir / "sample2.txt"
    ]
    
    # Check if documents exist (for this example, we'll just print a message)
    for doc_path in document_paths:
        if not doc_path.exists():
            print(f"Warning: Document {doc_path} does not exist. This is just an example script.")
    
    # Create pipelines with different configurations
    print("Creating pipeline configurations...")
    
    # Default pipeline with recursive chunking, dense retrieval, and cross-encoder reranking
    default_pipeline = PipelineFactory.create_from_preset("default")
    
    # Simple pipeline with sentence chunking and no reranking
    simple_pipeline = PipelineFactory.create_from_preset("simple")
    
    # BM25 pipeline using lexical search instead of embeddings
    bm25_pipeline = PipelineFactory.create_from_preset(
        "default", 
        retriever={"type": "bm25", "top_k": 5}
    )
    
    # Advanced pipeline with semantic chunking and HyDE retrieval
    advanced_pipeline = PipelineFactory.create_pipeline({
        "parser": {"type": "unstructured"},
        "chunker": {"type": "semantic", "chunk_size": 1000, "chunk_overlap": 200},
        "embedder": {"type": "openai", "model_name": "text-embedding-3-small"},
        "retriever": {
            "type": "hyde", 
            "similarity_metric": "cosine", 
            "top_k": 5,
            "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.0}
        },
        "reranker": {
            "type": "llm", 
            "top_k": 3,
            "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.0}
        },
        "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
    })
    
    # Add documents to pipelines
    print("Adding documents to pipelines...")
    for doc_path in document_paths:
        if doc_path.exists():
            default_pipeline.add_document_from_file(str(doc_path))
            simple_pipeline.add_document_from_file(str(doc_path))
            bm25_pipeline.add_document_from_file(str(doc_path))
            advanced_pipeline.add_document_from_file(str(doc_path))
    
    # Create evaluation metrics
    print("Setting up evaluation metrics...")
    eval_llm = OpenAILLM(model_name="gpt-4o", temperature=0.0)
    metrics = [
        GroundnessMetric(llm=eval_llm),
        RelevanceMetric(llm=eval_llm),
        ContextPrecisionMetric(llm=eval_llm),
        ContextRecallMetric(llm=eval_llm)
    ]
    
    # Create evaluator
    evaluator = Evaluator(metrics=metrics)
    
    # Define evaluation queries
    queries = [
        "What are the key components of a RAG pipeline?",
        "How does chunking affect retrieval performance?",
        "What are the advantages of using a reranker?",
        "Compare dense retrieval and BM25 approaches."
    ]
    
    # Optional ground truth for evaluation
    ground_truths = [
        {
            "answer": "The key components of a RAG pipeline include document parsing, chunking, embedding, retrieval, optional reranking, and LLM generation."
        },
        {
            "answer": "Chunking affects retrieval performance by determining the granularity and context boundaries of the information. Smaller chunks may increase precision but reduce context, while larger chunks provide more context but may include irrelevant information."
        },
        {
            "answer": "Rerankers improve retrieval precision by applying more sophisticated relevance models to the initial retrieval results, often considering the query and document as a pair rather than independently."
        },
        {
            "answer": "Dense retrieval uses semantic vector representations to find similar documents, capturing meaning beyond exact word matches. BM25 is a lexical approach based on term frequency and inverse document frequency, excelling at keyword matching but potentially missing semantic relationships."
        }
    ]
    
    # Evaluate pipelines
    print("Evaluating pipelines...")
    pipelines = {
        "default": default_pipeline,
        "simple": simple_pipeline,
        "bm25": bm25_pipeline,
        "advanced": advanced_pipeline
    }
    
    results = evaluator.evaluate_multiple_pipelines(pipelines, queries, ground_truths)
    
    # Save evaluation results
    print("Saving evaluation results...")
    results_path = evaluator.save_results(str(reports_dir))
    print(f"Results saved to: {results_path}")
    
    # Generate reports
    print("Generating reports...")
    report_generator = ReportGenerator(results)
    
    # Generate PDF report
    pdf_path = reports_dir / "rag_evaluation_report.pdf"
    report_generator.generate_pdf_report(str(pdf_path))
    print(f"PDF report generated: {pdf_path}")
    
    # Generate interactive dashboard
    html_path = reports_dir / "rag_evaluation_dashboard.html"
    report_generator.generate_interactive_dashboard(str(html_path))
    print(f"Interactive dashboard generated: {html_path}")
    
    # Generate comparison plots
    print("Generating comparison plots...")
    
    # Bar chart comparing all metrics
    bar_plot_path = reports_dir / "metrics_comparison_bar.png"
    fig = report_generator.generate_metric_comparison_plot(plot_type="bar")
    fig.savefig(str(bar_plot_path), dpi=300, bbox_inches='tight')
    
    # Heatmap for detailed comparison
    heatmap_path = reports_dir / "metrics_comparison_heatmap.png"
    fig = report_generator.generate_metric_comparison_plot(plot_type="heatmap")
    fig.savefig(str(heatmap_path), dpi=300, bbox_inches='tight')
    
    print("Evaluation complete!")
    print(f"Check the reports directory: {reports_dir}")


if __name__ == "__main__":
    main()
