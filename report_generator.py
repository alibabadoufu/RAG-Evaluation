"""
Report generator for RAG evaluation results.
"""
from typing import List, Dict, Any, Optional, Union
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ReportGenerator:
    """
    Report generator for RAG evaluation results.
    
    This class generates detailed reports and visualizations from evaluation results.
    """
    
    def __init__(self, results: Dict[str, Any] = None, **kwargs):
        """
        Initialize the report generator.
        
        Args:
            results: Optional evaluation results to use
            **kwargs: Additional parameters
        """
        self.results = results or {}
        self.config = kwargs
        
    def set_results(self, results: Dict[str, Any]) -> None:
        """
        Set the evaluation results to use for report generation.
        
        Args:
            results: Evaluation results
        """
        self.results = results
        
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
    
    def generate_summary_dataframe(self, pipeline_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a summary DataFrame of evaluation results.
        
        Args:
            pipeline_names: Optional list of pipeline names to include (if None, include all)
            
        Returns:
            A pandas DataFrame with summary results
        """
        if not self.results:
            raise ValueError("No evaluation results available.")
            
        if pipeline_names is None:
            pipeline_names = list(self.results.keys())
            
        # Extract metric scores for each pipeline
        summary_data = []
        
        for name in pipeline_names:
            if name not in self.results:
                continue
                
            pipeline_result = self.results[name]
            row = {"Pipeline": name}
            
            # Add pipeline configuration details
            pipeline_config = pipeline_result.get("pipeline_config", "")
            components = pipeline_config.split("\n")[1:-1]  # Skip first and last lines
            for component in components:
                component = component.strip()
                if component:
                    parts = component.split(":", 1)
                    if len(parts) == 2:
                        component_type, component_name = parts
                        row[component_type.strip()] = component_name.strip()
            
            # Add metric scores
            for metric_name, metric_result in pipeline_result["metrics"].items():
                row[f"{metric_name}"] = metric_result["average"]
                
            # Add timestamp and query count
            row["Timestamp"] = pipeline_result.get("timestamp", "")
            row["Queries"] = pipeline_result.get("num_queries", 0)
            
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)
    
    def generate_metric_comparison_plot(self, 
                                       metric_names: Optional[List[str]] = None,
                                       pipeline_names: Optional[List[str]] = None,
                                       output_path: Optional[str] = None,
                                       plot_type: str = "bar",
                                       figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Generate a plot comparing metrics across pipelines.
        
        Args:
            metric_names: Optional list of metrics to include (if None, include all)
            pipeline_names: Optional list of pipeline names to include (if None, include all)
            output_path: Optional path to save the plot
            plot_type: Type of plot ('bar', 'radar', 'heatmap')
            figsize: Figure size
            
        Returns:
            The matplotlib Figure object
        """
        summary_df = self.generate_summary_dataframe(pipeline_names)
        
        if not metric_names:
            # Find all metric columns
            metric_names = [col for col in summary_df.columns 
                           if col not in ["Pipeline", "Timestamp", "Queries"] 
                           and not col.startswith("Parser") 
                           and not col.startswith("Chunker")
                           and not col.startswith("Embedder")
                           and not col.startswith("Retriever")
                           and not col.startswith("Reranker")
                           and not col.startswith("LLM")]
        
        if plot_type == "bar":
            fig, ax = plt.subplots(figsize=figsize)
            summary_df.set_index("Pipeline")[metric_names].plot(kind="bar", ax=ax)
            ax.set_title("Metrics Comparison Across Pipelines")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
        elif plot_type == "radar":
            # Radar chart (spider plot)
            fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
            
            # Number of metrics (variables)
            N = len(metric_names)
            
            # Angle of each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Plot for each pipeline
            for _, row in summary_df.iterrows():
                values = [row[metric] for metric in metric_names]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=1, label=row["Pipeline"])
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels and title
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_names)
            ax.set_title("Metrics Comparison (Radar Chart)")
            ax.set_ylim(0, 1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.tight_layout()
            
        elif plot_type == "heatmap":
            # Heatmap
            plt.figure(figsize=figsize)
            heatmap_data = summary_df.set_index("Pipeline")[metric_names]
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
            plt.title("Metrics Comparison (Heatmap)")
            plt.tight_layout()
            
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
            
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_component_analysis_plot(self, 
                                        component_type: str,
                                        metric_name: str,
                                        output_path: Optional[str] = None,
                                        figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Generate a plot analyzing the impact of a specific component type on a metric.
        
        Args:
            component_type: The component type to analyze (e.g., 'Chunker', 'Retriever')
            metric_name: The metric to analyze
            output_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            The matplotlib Figure object
        """
        summary_df = self.generate_summary_dataframe()
        
        if component_type not in summary_df.columns:
            raise ValueError(f"Component type {component_type} not found in results")
            
        if metric_name not in summary_df.columns:
            raise ValueError(f"Metric {metric_name} not found in results")
            
        # Group by component and calculate mean metric score
        component_analysis = summary_df.groupby(component_type)[metric_name].agg(['mean', 'std', 'count']).reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(component_analysis[component_type], component_analysis['mean'], 
                     yerr=component_analysis['std'], capsize=10)
        
        # Add count labels
        for i, bar in enumerate(bars):
            count = component_analysis.iloc[i]['count']
            ax.text(bar.get_x() + bar.get_width()/2, 0.05, f"n={count}", 
                   ha='center', va='bottom', color='black', fontweight='bold')
        
        ax.set_title(f"Impact of {component_type} on {metric_name}")
        ax.set_ylabel(f"{metric_name} Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_project_comparison_plot(self, 
                                        project_column: str,
                                        metric_names: Optional[List[str]] = None,
                                        output_path: Optional[str] = None,
                                        figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Generate a plot comparing metrics across different projects.
        
        Args:
            project_column: The column containing project identifiers
            metric_names: Optional list of metrics to include (if None, include all)
            output_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            The matplotlib Figure object
        """
        summary_df = self.generate_summary_dataframe()
        
        if project_column not in summary_df.columns:
            raise ValueError(f"Project column {project_column} not found in results")
            
        if not metric_names:
            # Find all metric columns
            metric_names = [col for col in summary_df.columns 
                           if col not in ["Pipeline", "Timestamp", "Queries"] 
                           and not col.startswith("Parser") 
                           and not col.startswith("Chunker")
                           and not col.startswith("Embedder")
                           and not col.startswith("Retriever")
                           and not col.startswith("Reranker")
                           and not col.startswith("LLM")]
        
        # Group by project and calculate mean metric scores
        project_metrics = []
        
        for project in summary_df[project_column].unique():
            project_data = summary_df[summary_df[project_column] == project]
            row = {project_column: project}
            
            for metric in metric_names:
                row[metric] = project_data[metric].mean()
                
            project_metrics.append(row)
            
        project_df = pd.DataFrame(project_metrics)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        project_df.set_index(project_column)[metric_names].plot(kind="bar", ax=ax)
        ax.set_title(f"Metrics Comparison Across {project_column}s")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_interactive_dashboard(self, 
                                      output_path: str,
                                      pipeline_names: Optional[List[str]] = None) -> None:
        """
        Generate an interactive HTML dashboard with Plotly.
        
        Args:
            output_path: Path to save the HTML dashboard
            pipeline_names: Optional list of pipeline names to include (if None, include all)
        """
        summary_df = self.generate_summary_dataframe(pipeline_names)
        
        # Find all metric columns
        metric_names = [col for col in summary_df.columns 
                       if col not in ["Pipeline", "Timestamp", "Queries"] 
                       and not col.startswith("Parser") 
                       and not col.startswith("Chunker")
                       and not col.startswith("Embedder")
                       and not col.startswith("Retriever")
                       and not col.startswith("Reranker")
                       and not col.startswith("LLM")]
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Overall Metrics Comparison", "Metric Breakdown by Pipeline", 
                           "Component Analysis", "Performance Distribution"),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                  [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Plot 1: Overall metrics bar chart
        for i, metric in enumerate(metric_names):
            fig.add_trace(
                go.Bar(
                    x=summary_df["Pipeline"],
                    y=summary_df[metric],
                    name=metric
                ),
                row=1, col=1
            )
        
        # Plot 2: Heatmap of metrics by pipeline
        heatmap_data = summary_df.set_index("Pipeline")[metric_names]
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=metric_names,
                y=heatmap_data.index,
                colorscale="Viridis",
                zmin=0, zmax=1
            ),
            row=1, col=2
        )
        
        # Plot 3: Component analysis (for first component type found)
        component_types = [col for col in summary_df.columns 
                          if col.startswith("Parser") 
                          or col.startswith("Chunker")
                          or col.startswith("Embedder")
                          or col.startswith("Retriever")
                          or col.startswith("Reranker")
                          or col.startswith("LLM")]
        
        if component_types and metric_names:
            component_type = component_types[0]
            metric = metric_names[0]
            
            component_analysis = summary_df.groupby(component_type)[metric].mean().reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=component_analysis[component_type],
                    y=component_analysis[metric],
                    name=f"{component_type} impact on {metric}"
                ),
                row=2, col=1
            )
        
        # Plot 4: Box plot of metric distributions
        for metric in metric_names:
            fig.add_trace(
                go.Box(
                    y=summary_df[metric],
                    name=metric
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="RAG Evaluation Dashboard",
            height=800,
            showlegend=False
        )
        
        # Save to HTML
        fig.write_html(output_path)
    
    def generate_pdf_report(self, 
                           output_path: str,
                           title: str = "RAG Evaluation Report",
                           pipeline_names: Optional[List[str]] = None) -> None:
        """
        Generate a comprehensive PDF report.
        
        Args:
            output_path: Path to save the PDF report
            title: Report title
   
(Content truncated due to size limit. Use line ranges to read in chunks)