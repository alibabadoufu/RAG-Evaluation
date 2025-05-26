"""
Evaluation interface component for the Gradio UI.
"""
import gradio as gr
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple

from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.pipeline_factory import PipelineFactory
from src.evaluation.evaluator import Evaluator
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.metrics.groundness_metric import GroundnessMetric
from src.evaluation.metrics.relevance_metric import RelevanceMetric
from src.evaluation.metrics.context_precision_metric import ContextPrecisionMetric
from src.evaluation.metrics.context_recall_metric import ContextRecallMetric
from src.components.llms.openai_llm import OpenAILLM


class EvaluationInterface:
    """
    Evaluation interface component for evaluating RAG pipelines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluation interface.
        
        Args:
            config_path: Optional path to a JSON configuration file with pipeline presets
        """
        self.pipelines = {}
        self.evaluator = None
        self.report_generator = None
        self.evaluation_results = {}
        
        # Load configurations if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.configs = json.load(f)
        else:
            # Default configurations
            self.configs = {
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
                "bm25": {
                    "parser": {"type": "unstructured"},
                    "chunker": {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
                    "embedder": {"type": "sentence_transformer", "model_name": "all-MiniLM-L6-v2"},
                    "retriever": {"type": "bm25", "top_k": 5},
                    "llm": {"type": "openai", "model_name": "gpt-4o", "temperature": 0.7}
                }
            }
    
    def initialize_evaluator(self) -> str:
        """
        Initialize the evaluator with metrics.
        
        Returns:
            Status message
        """
        try:
            # Create an LLM for evaluation metrics
            eval_llm = OpenAILLM(model_name="gpt-4o", temperature=0.0)
            
            # Create metrics
            metrics = [
                GroundnessMetric(llm=eval_llm),
                RelevanceMetric(llm=eval_llm),
                ContextPrecisionMetric(llm=eval_llm),
                ContextRecallMetric(llm=eval_llm)
            ]
            
            # Create evaluator and report generator
            self.evaluator = Evaluator(metrics=metrics)
            self.report_generator = ReportGenerator()
            
            return "Evaluator initialized successfully."
        except Exception as e:
            return f"Error initializing evaluator: {str(e)}"
    
    def create_pipeline(self, config_name: str) -> Tuple[RAGPipeline, str]:
        """
        Create a pipeline from a named configuration.
        
        Args:
            config_name: Name of the configuration to use
            
        Returns:
            Tuple of (pipeline, status message)
        """
        if config_name not in self.configs:
            return None, f"Unknown configuration: {config_name}"
            
        try:
            pipeline = PipelineFactory.create_pipeline(self.configs[config_name])
            self.pipelines[config_name] = pipeline
            return pipeline, f"Pipeline '{config_name}' created successfully."
        except Exception as e:
            return None, f"Error creating pipeline: {str(e)}"
    
    def load_documents(self, config_names: List[str], file_paths: List[str]) -> str:
        """
        Load documents into multiple pipelines.
        
        Args:
            config_names: List of configuration names to load documents into
            file_paths: List of paths to document files
            
        Returns:
            Status message
        """
        try:
            for config_name in config_names:
                if config_name not in self.pipelines:
                    pipeline, status = self.create_pipeline(config_name)
                    if not pipeline:
                        return status
                else:
                    pipeline = self.pipelines[config_name]
                
                for file_path in file_paths:
                    pipeline.add_document_from_file(file_path)
                    
            return f"Successfully loaded {len(file_paths)} documents into {len(config_names)} pipelines."
        except Exception as e:
            return f"Error loading documents: {str(e)}"
    
    def evaluate_pipelines(self, 
                          config_names: List[str], 
                          queries: List[str],
                          ground_truths: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Evaluate multiple pipelines on a set of queries.
        
        Args:
            config_names: List of configuration names to evaluate
            queries: List of query strings
            ground_truths: Optional list of ground truth information
            
        Returns:
            Status message
        """
        if not self.evaluator:
            status = self.initialize_evaluator()
            if "Error" in status:
                return status
        
        try:
            # Ensure all pipelines are created
            for config_name in config_names:
                if config_name not in self.pipelines:
                    pipeline, status = self.create_pipeline(config_name)
                    if not pipeline:
                        return status
            
            # Evaluate each pipeline
            pipelines_dict = {name: self.pipelines[name] for name in config_names}
            results = self.evaluator.evaluate_multiple_pipelines(pipelines_dict, queries, ground_truths)
            
            # Store results for report generation
            self.evaluation_results = results
            self.report_generator.set_results(results)
            
            return f"Successfully evaluated {len(config_names)} pipelines on {len(queries)} queries."
        except Exception as e:
            return f"Error evaluating pipelines: {str(e)}"
    
    def generate_comparison_plot(self, 
                               metric_name: Optional[str] = None,
                               plot_type: str = "bar") -> Tuple[plt.Figure, str]:
        """
        Generate a comparison plot of evaluation results.
        
        Args:
            metric_name: Optional specific metric to plot
            plot_type: Type of plot ('bar', 'radar', 'heatmap')
            
        Returns:
            Tuple of (matplotlib figure, status message)
        """
        if not self.report_generator or not self.evaluation_results:
            return None, "No evaluation results available. Run evaluation first."
            
        try:
            if metric_name:
                fig = self.report_generator.generate_metric_comparison_plot(
                    metric_names=[metric_name],
                    plot_type=plot_type
                )
            else:
                fig = self.report_generator.generate_metric_comparison_plot(
                    plot_type=plot_type
                )
                
            return fig, "Plot generated successfully."
        except Exception as e:
            return None, f"Error generating plot: {str(e)}"
    
    def generate_report(self, output_dir: str, report_type: str = "pdf") -> str:
        """
        Generate an evaluation report.
        
        Args:
            output_dir: Directory to save the report
            report_type: Type of report ('pdf', 'html', 'json')
            
        Returns:
            Status message with path to the generated report
        """
        if not self.report_generator or not self.evaluation_results:
            return "No evaluation results available. Run evaluation first."
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            if report_type == "pdf":
                output_path = os.path.join(output_dir, "rag_evaluation_report.pdf")
                self.report_generator.generate_pdf_report(output_path)
                return f"PDF report generated successfully: {output_path}"
            
            elif report_type == "html":
                output_path = os.path.join(output_dir, "rag_evaluation_dashboard.html")
                self.report_generator.generate_interactive_dashboard(output_path)
                return f"HTML dashboard generated successfully: {output_path}"
            
            elif report_type == "json":
                output_path = os.path.join(output_dir, "rag_evaluation_results.json")
                self.evaluator.save_results(output_dir, "rag_evaluation_results.json")
                return f"JSON results saved successfully: {output_path}"
            
            else:
                return f"Unsupported report type: {report_type}"
                
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface for the evaluation component.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## RAG Pipeline Evaluation")
                    
                    with gr.Group():
                        gr.Markdown("### Pipeline Selection")
                        pipeline_checkboxes = gr.CheckboxGroup(
                            choices=list(self.configs.keys()),
                            value=["default"],
                            label="Select Pipelines to Evaluate"
                        )
                    
                    with gr.Group():
                        gr.Markdown("### Document Loading")
                        file_upload = gr.File(
                            file_count="multiple",
                            label="Upload Documents"
                        )
                        load_docs_btn = gr.Button("Load Documents")
                        docs_status = gr.Textbox(label="Document Status", interactive=False)
                    
                    with gr.Group():
                        gr.Markdown("### Evaluation Queries")
                        queries_input = gr.Textbox(
                            lines=5,
                            placeholder="Enter evaluation queries, one per line",
                            label="Evaluation Queries"
                        )
                        
                        # Optional ground truth input
                        ground_truth_input = gr.Textbox(
                            lines=5,
                            placeholder="Optional: Enter ground truth answers in JSON format",
                            label="Ground Truth (Optional)"
                        )
                        
                        evaluate_btn = gr.Button("Run Evaluation")
                        eval_status = gr.Textbox(label="Evaluation Status", interactive=False)
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Comparison Plot"):
                            with gr.Row():
                                metric_dropdown = gr.Dropdown(
                                    choices=["groundness", "relevance", "context_precision", "context_recall", "All Metrics"],
                                    value="All Metrics",
                                    label="Select Metric"
                                )
                                
                                plot_type_dropdown = gr.Dropdown(
                                    choices=["bar", "radar", "heatmap"],
                                    value="bar",
                                    label="Plot Type"
                                )
                                
                                generate_plot_btn = gr.Button("Generate Plot")
                            
                            plot_output = gr.Plot(label="Evaluation Results")
                            plot_status = gr.Textbox(label="Plot Status", interactive=False)
                        
                        with gr.TabItem("Report Generation"):
                            with gr.Row():
                                report_type_dropdown = gr.Dropdown(
                                    choices=["pdf", "html", "json"],
                                    value="pdf",
                                    label="Report Type"
                                )
                                
                                output_dir_input = gr.Textbox(
                                    value="/home/ubuntu/rag_evaluation_framework/results/reports",
                                    label="Output Directory"
                                )
                                
                                generate_report_btn = gr.Button("Generate Report")
                            
                            report_status = gr.Textbox(label="Report Status", interactive=False)
            
            # Event handlers
            load_docs_btn.click(
                fn=self.load_documents,
                inputs=[pipeline_checkboxes, file_upload],
                outputs=[docs_status]
            )
            
            def parse_queries_and_ground_truths(queries_text, ground_truth_text):
                # Parse queries (one per line)
                queries = [q.strip() for q in queries_text.split("\n") if q.strip()]
                
                # Parse ground truths if provided
                ground_truths = None
                if ground_truth_text.strip():
                    try:
                        ground_truths = json.loads(ground_truth_text)
                    except:
                        return [], None, "Error parsing ground truth JSON. Using queries without ground truth."
                
                return queries, ground_truths, f"Parsed {len(queries)} queries successfully."
            
            def run_evaluation(pipeline_names, queries_text, ground_truth_text):
                queries, ground_truths, parse_status = parse_queries_and_ground_truths(queries_text, ground_truth_text)
                if not queries:
                    return "No valid queries found. Please enter at least one query."
                
                status = self.evaluate_pipelines(pipeline_names, queries, ground_truths)
                return status
            
 
(Content truncated due to size limit. Use line ranges to read in chunks)