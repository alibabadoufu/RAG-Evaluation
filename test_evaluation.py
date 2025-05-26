"""
Basic test for the evaluation and reporting components.
"""
import sys
import os
import unittest
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.evaluator import Evaluator
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.metrics.groundness_metric import GroundnessMetric
from src.evaluation.metrics.relevance_metric import RelevanceMetric
from src.components.llms.openai_llm import OpenAILLM
from src.pipeline.pipeline_factory import PipelineFactory


class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation and reporting components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test LLM for metrics
        self.eval_llm = OpenAILLM(model_name="gpt-4o", temperature=0.0)
        
        # Create metrics
        self.metrics = [
            GroundnessMetric(llm=self.eval_llm),
            RelevanceMetric(llm=self.eval_llm)
        ]
        
        # Create evaluator
        self.evaluator = Evaluator(metrics=self.metrics)
        
        # Sample data for testing
        self.queries = ["What is RAG?", "How does chunking affect retrieval?"]
        self.responses = [
            "RAG stands for Retrieval-Augmented Generation, which combines retrieval with generation.",
            "Chunking affects retrieval by determining the granularity of information."
        ]
        self.contexts = [
            [
                {"text": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval with generation.", "metadata": {"source": "test"}}
            ],
            [
                {"text": "Chunking is the process of splitting documents into smaller pieces.", "metadata": {"source": "test"}},
                {"text": "The size and overlap of chunks affects retrieval performance.", "metadata": {"source": "test"}}
            ]
        ]
        self.ground_truths = [
            {"answer": "RAG stands for Retrieval-Augmented Generation, which combines retrieval with generation."},
            {"answer": "Chunking affects retrieval by determining the granularity of information."}
        ]
        
        # Create test pipelines
        self.pipelines = {
            "default": PipelineFactory.create_from_preset("default"),
            "simple": PipelineFactory.create_from_preset("simple")
        }
        
        # Mock evaluation results for testing report generation
        self.mock_results = {
            "default": {
                "pipeline_name": "default",
                "pipeline_config": "RAGPipeline(\n  Parser: UnstructuredParser\n  Chunker: RecursiveChunker\n  Embedder: SentenceTransformerEmbedder\n  Retriever: DenseRetriever\n  Reranker: CrossEncoderReranker\n  LLM: OpenAILLM\n)",
                "metrics": {
                    "groundness": {
                        "average": 0.85,
                        "scores": [0.9, 0.8]
                    },
                    "relevance": {
                        "average": 0.75,
                        "scores": [0.8, 0.7]
                    }
                },
                "timestamp": "2025-05-26T01:00:00",
                "num_queries": 2
            },
            "simple": {
                "pipeline_name": "simple",
                "pipeline_config": "RAGPipeline(\n  Parser: UnstructuredParser\n  Chunker: SentenceChunker\n  Embedder: SentenceTransformerEmbedder\n  Retriever: DenseRetriever\n  LLM: OpenAILLM\n)",
                "metrics": {
                    "groundness": {
                        "average": 0.75,
                        "scores": [0.8, 0.7]
                    },
                    "relevance": {
                        "average": 0.65,
                        "scores": [0.7, 0.6]
                    }
                },
                "timestamp": "2025-05-26T01:00:00",
                "num_queries": 2
            }
        }
    
    def test_metric_evaluation(self):
        """Test individual metric evaluation."""
        # Test groundness metric
        groundness_metric = self.metrics[0]
        result = groundness_metric.evaluate(
            self.queries[0],
            self.responses[0],
            self.contexts[0],
            self.ground_truths[0]
        )
        
        # Basic validation
        self.assertIn('score', result)
        self.assertIn('details', result)
        self.assertIsInstance(result['score'], float)
        self.assertGreaterEqual(result['score'], 0.0)
        self.assertLessEqual(result['score'], 1.0)
    
    def test_batch_evaluation(self):
        """Test batch evaluation of metrics."""
        # Test groundness metric batch evaluation
        groundness_metric = self.metrics[0]
        result = groundness_metric.evaluate_batch(
            self.queries,
            self.responses,
            self.contexts,
            self.ground_truths
        )
        
        # Basic validation
        self.assertIn('scores', result)
        self.assertIn('average', result)
        self.assertIn('details', result)
        self.assertEqual(len(result['scores']), 2)
        self.assertIsInstance(result['average'], float)
    
    def test_evaluator_pipeline_evaluation(self):
        """Test evaluator pipeline evaluation."""
        # Mock the evaluate_batch method to avoid actual LLM calls
        for metric in self.metrics:
            metric.evaluate_batch = lambda queries, responses, contexts, ground_truths: {
                'scores': [0.8, 0.7],
                'average': 0.75,
                'details': {'metric': metric.name}
            }
        
        # Evaluate a pipeline
        result = self.evaluator.evaluate_pipeline(
            self.pipelines['default'],
            self.queries,
            self.ground_truths,
            'default'
        )
        
        # Basic validation
        self.assertIn('pipeline_name', result)
        self.assertIn('metrics', result)
        self.assertIn('timestamp', result)
        self.assertIn('num_queries', result)
        self.assertEqual(result['pipeline_name'], 'default')
        self.assertEqual(result['num_queries'], 2)
        self.assertIn('groundness', result['metrics'])
        self.assertIn('relevance', result['metrics'])
    
    def test_evaluator_multiple_pipelines(self):
        """Test evaluator multiple pipeline evaluation."""
        # Mock the evaluate_batch method to avoid actual LLM calls
        for metric in self.metrics:
            metric.evaluate_batch = lambda queries, responses, contexts, ground_truths: {
                'scores': [0.8, 0.7],
                'average': 0.75,
                'details': {'metric': metric.name}
            }
        
        # Evaluate multiple pipelines
        results = self.evaluator.evaluate_multiple_pipelines(
            self.pipelines,
            self.queries,
            self.ground_truths
        )
        
        # Basic validation
        self.assertIn('default', results)
        self.assertIn('simple', results)
        self.assertIn('pipeline_name', results['default'])
        self.assertIn('metrics', results['default'])
        self.assertIn('groundness', results['default']['metrics'])
        self.assertIn('relevance', results['default']['metrics'])
    
    def test_report_generator_dataframe(self):
        """Test report generator summary dataframe."""
        # Create report generator with mock results
        report_generator = ReportGenerator(self.mock_results)
        
        # Generate summary dataframe
        df = report_generator.generate_summary_dataframe()
        
        # Basic validation
        self.assertEqual(len(df), 2)
        self.assertIn('Pipeline', df.columns)
        self.assertIn('groundness', df.columns)
        self.assertIn('relevance', df.columns)
    
    def test_report_generator_plot(self):
        """Test report generator plot creation."""
        # Create report generator with mock results
        report_generator = ReportGenerator(self.mock_results)
        
        # Generate plot
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_plot.png')
            fig = report_generator.generate_metric_comparison_plot(
                metric_names=['groundness', 'relevance'],
                output_path=output_path
            )
            
            # Basic validation
            self.assertIsInstance(fig, plt.Figure)
            self.assertTrue(os.path.exists(output_path))
    
    def test_report_generator_dashboard(self):
        """Test report generator dashboard creation."""
        # Create report generator with mock results
        report_generator = ReportGenerator(self.mock_results)
        
        # Generate dashboard
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_dashboard.html')
            report_generator.generate_interactive_dashboard(output_path)
            
            # Basic validation
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == '__main__':
    unittest.main()
