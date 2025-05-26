"""
Implementation of groundness evaluation metric.
"""
from typing import List, Dict, Any, Optional
import re

from src.components.base.base_llm import BaseLLM
from src.evaluation.base.base_metric import BaseMetric


class GroundnessMetric(BaseMetric):
    """
    Metric for evaluating the groundness of responses.
    
    This metric assesses whether the generated response is grounded in the
    provided context, or if it contains hallucinations.
    """
    
    def __init__(self, llm: BaseLLM, name: str = "groundness", **kwargs):
        """
        Initialize the groundness metric.
        
        Args:
            llm: The LLM to use for evaluation
            name: The name of the metric
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self.llm = llm
        
        # Template for groundness evaluation
        self.groundness_prompt_template = """
        You are evaluating the groundness of an AI assistant's response to a user query.
        A grounded response only contains information that is explicitly stated in or can be directly inferred from the provided context.
        
        User Query: {query}
        
        Context Information:
        {context}
        
        AI Response: {response}
        
        Evaluate the groundness of the AI response on a scale from 0 to 10:
        - 0: Completely ungrounded. The response contains significant hallucinations or information not supported by the context.
        - 5: Partially grounded. The response contains some information from the context but also includes unsupported claims.
        - 10: Fully grounded. The response only contains information explicitly stated in or directly inferable from the context.
        
        First, identify any claims or statements in the response and check if they are supported by the context.
        Then provide your score and reasoning.
        
        Score (0-10):
        """
        
    def evaluate(self, 
                query: str, 
                response: str, 
                context: List[Dict[str, Any]], 
                ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate the groundness of a single query-response pair.
        
        Args:
            query: The user query
            response: The generated response
            context: The context documents used for generation
            ground_truth: Optional ground truth information for reference
            
        Returns:
            A dictionary containing:
                - 'score': The groundness score (0-1)
                - 'details': Additional details about the evaluation
        """
        # Format the context text
        context_text = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(context)])
        
        # Create the evaluation prompt
        prompt = self.groundness_prompt_template.format(
            query=query,
            context=context_text,
            response=response
        )
        
        # Get the evaluation from the LLM
        evaluation = self.llm.generate(prompt)
        
        # Extract the score
        score_match = re.search(r'Score\s*(?:\(0-10\))?\s*:\s*(\d+(?:\.\d+)?)', evaluation)
        if score_match:
            raw_score = float(score_match.group(1))
            # Normalize to [0, 1]
            normalized_score = raw_score / 10.0
        else:
            # Default score if extraction fails
            normalized_score = 0.5
            
        return {
            'score': normalized_score,
            'details': {
                'raw_score': raw_score if score_match else 5.0,
                'evaluation': evaluation,
                'metric': self.name
            }
        }
    
    def evaluate_batch(self, 
                      queries: List[str], 
                      responses: List[str], 
                      contexts: List[List[Dict[str, Any]]], 
                      ground_truths: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Evaluate the groundness of a batch of query-response pairs.
        
        Args:
            queries: List of user queries
            responses: List of generated responses
            contexts: List of context document lists used for generation
            ground_truths: Optional list of ground truth information for reference
            
        Returns:
            A dictionary containing:
                - 'scores': List of individual scores
                - 'average': Average score across all examples
                - 'details': Additional details about the evaluation
        """
        if len(queries) != len(responses) or len(queries) != len(contexts):
            raise ValueError("Length of queries, responses, and contexts must be the same")
            
        # Evaluate each example
        results = []
        for i in range(len(queries)):
            ground_truth = None
            if ground_truths and i < len(ground_truths):
                ground_truth = ground_truths[i]
                
            result = self.evaluate(queries[i], responses[i], contexts[i], ground_truth)
            results.append(result)
            
        # Calculate average score
        scores = [result['score'] for result in results]
        average_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'scores': scores,
            'average': average_score,
            'details': {
                'individual_results': results,
                'metric': self.name
            }
        }
