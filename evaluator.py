"""
Evaluator for RAG pipelines.
"""
from typing import List, Dict, Any, Optional, Union
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.base.base_metric import BaseMetric


class Evaluator:
    """
    Evaluator for RAG pipelines.
    
    This class handles the evaluation of RAG pipelines using various metrics.
    """
    
    def __init__(self, metrics: List[BaseMetric], **kwargs):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of evaluation metrics to use
            **kwargs: Additional parameters
        """
        self.metrics = metrics
        self.config = kwargs
        self.results = {}
        
    def evaluate_pipeline(self, 
                         pipeline: RAGPipeline, 
                         queries: List[str], 
                         ground_truths: Optional[List[Dict[str, Any]]] = None,
                         pipeline_name: str = "default") -> Dict[str, Any]:
        """
        Evaluate a RAG pipeline on a set of queries.
        
        Args:
            pipeline: The RAG pipeline to evaluate
            queries: List of query strings
            ground_truths: Optional list of ground truth information for reference
            pipeline_name: Name identifier for the pipeline
            
        Returns:
            A dictionary containing evaluation results
        """
        # Generate responses for all queries
        responses = []
        contexts = []
        
        for query in queries:
            result = pipeline.generate(query)
            responses.append(result["response"])
            contexts.append(result["context"])
            
        # Evaluate using each metric
        metric_results = {}
        for metric in self.metrics:
            result = metric.evaluate_batch(queries, responses, contexts, ground_truths)
            metric_results[metric.name] = result
            
        # Store results
        pipeline_result = {
            "pipeline_name": pipeline_name,
            "pipeline_config": str(pipeline),
            "metrics": metric_results,
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(queries)
        }
        
        self.results[pipeline_name] = pipeline_result
        return pipeline_result
    
    def evaluate_multiple_pipelines(self, 
                                   pipelines: Dict[str, RAGPipeline], 
                                   queries: List[str], 
                                   ground_truths: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple RAG pipelines on the same set of queries.
        
        Args:
            pipelines: Dictionary mapping pipeline names to RAG pipeline instances
            queries: List of query strings
            ground_truths: Optional list of ground truth information for reference
            
        Returns:
            A dictionary mapping pipeline names to their evaluation results
        """
        results = {}
        for name, pipeline in pipelines.items():
            results[name] = self.evaluate_pipeline(pipeline, queries, ground_truths, name)
            
        return results
    
    def compare_pipelines(self, pipeline_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare the performance of different pipelines.
        
        Args:
            pipeline_names: Optional list of pipeline names to compare (if None, compare all)
            
        Returns:
            A pandas DataFrame with comparison results
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_pipeline first.")
            
        if pipeline_names is None:
            pipeline_names = list(self.results.keys())
            
        # Extract metric scores for each pipeline
        comparison_data = []
        
        for name in pipeline_names:
            if name not in self.results:
                continue
                
            pipeline_result = self.results[name]
            row = {"Pipeline": name}
            
            for metric_name, metric_result in pipeline_result["metrics"].items():
                row[f"{metric_name}_avg"] = metric_result["average"]
                
            comparison_data.append(row)
            
        return pd.DataFrame(comparison_data)
    
    def save_results(self, output_dir: str, filename: str = "evaluation_results.json") -> str:
        """
        Save evaluation results to a file.
        
        Args:
            output_dir: Directory to save the results
            filename: Name of the output file
            
        Returns:
            Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Convert results to serializable format
        serializable_results = {}
        for name, result in self.results.items():
            serializable_results[name] = {
                "pipeline_name": result["pipeline_name"],
                "pipeline_config": result["pipeline_config"],
                "timestamp": result["timestamp"],
                "num_queries": result["num_queries"],
                "metrics": {}
            }
            
            for metric_name, metric_result in result["metrics"].items():
                serializable_results[name]["metrics"][metric_name] = {
                    "average": metric_result["average"],
                    "scores": metric_result["scores"]
                }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        return output_path
    
    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from a file.
        
        Args:
            input_path: Path to the results file
            
        Returns:
            The loaded results
        """
        with open(input_path, 'r') as f:
            self.results = json.load(f)
            
        return self.results
    
    def plot_comparison(self, 
                       metric_name: Optional[str] = None, 
                       pipeline_names: Optional[List[str]] = None,
                       output_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a comparison of pipeline performances.
        
        Args:
            metric_name: Optional specific metric to plot (if None, plot all metrics)
            pipeline_names: Optional list of pipeline names to compare (if None, compare all)
            output_path: Optional path to save the plot
            
        Returns:
            The matplotlib Figure object
        """
        comparison_df = self.compare_pipelines(pipeline_names)
        
        if metric_name:
            # Plot a specific metric across pipelines
            metric_col = f"{metric_name}_avg"
            if metric_col not in comparison_df.columns:
                raise ValueError(f"Metric {metric_name} not found in results")
                
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.plot(x="Pipeline", y=metric_col, kind="bar", ax=ax)
            ax.set_title(f"{metric_name} Comparison")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            plt.tight_layout()
        else:
            # Plot all metrics across pipelines
            metric_cols = [col for col in comparison_df.columns if col.endswith("_avg")]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            comparison_df.set_index("Pipeline")[metric_cols].plot(kind="bar", ax=ax)
            ax.set_title("Metrics Comparison")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.legend([col.replace("_avg", "") for col in metric_cols])
            plt.tight_layout()
            
        if output_path:
            plt.savefig(output_path)
            
        return fig
