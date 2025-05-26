"""
Chat interface component for the Gradio UI.
"""
import gradio as gr
import os
import json
from typing import Dict, Any, List, Optional

from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.pipeline_factory import PipelineFactory


class ChatInterface:
    """
    Chat interface component for interacting with RAG pipelines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the chat interface.
        
        Args:
            config_path: Optional path to a JSON configuration file with pipeline presets
        """
        self.pipelines = {}
        self.current_pipeline = None
        self.current_pipeline_name = None
        self.chat_history = []
        
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
                }
            }
    
    def create_pipeline(self, config_name: str) -> RAGPipeline:
        """
        Create a pipeline from a named configuration.
        
        Args:
            config_name: Name of the configuration to use
            
        Returns:
            The created RAG pipeline
        """
        if config_name not in self.configs:
            raise ValueError(f"Unknown configuration: {config_name}")
            
        pipeline = PipelineFactory.create_pipeline(self.configs[config_name])
        return pipeline
    
    def load_documents(self, pipeline: RAGPipeline, file_paths: List[str]) -> str:
        """
        Load documents into a pipeline.
        
        Args:
            pipeline: The RAG pipeline to load documents into
            file_paths: List of paths to document files
            
        Returns:
            Status message
        """
        try:
            for file_path in file_paths:
                pipeline.add_document_from_file(file_path)
                
            return f"Successfully loaded {len(file_paths)} documents."
        except Exception as e:
            return f"Error loading documents: {str(e)}"
    
    def chat(self, message: str, history: List[List[str]], config_name: str, pipeline_state: Dict[str, Any]) -> tuple:
        """
        Process a chat message using the selected RAG pipeline.
        
        Args:
            message: User message
            history: Chat history
            config_name: Name of the pipeline configuration to use
            pipeline_state: State dictionary for the pipeline
            
        Returns:
            Tuple of (response, updated history, updated pipeline state)
        """
        # Check if we need to create or switch pipelines
        if not pipeline_state or pipeline_state.get("config_name") != config_name:
            try:
                pipeline = self.create_pipeline(config_name)
                pipeline_state = {
                    "config_name": config_name,
                    "documents_loaded": False,
                    "chat_history": []
                }
            except Exception as e:
                return f"Error creating pipeline: {str(e)}", history, pipeline_state
        else:
            # Use existing pipeline
            pipeline = self.pipelines.get(config_name)
            if not pipeline:
                try:
                    pipeline = self.create_pipeline(config_name)
                except Exception as e:
                    return f"Error creating pipeline: {str(e)}", history, pipeline_state
        
        # Store the pipeline for future use
        self.pipelines[config_name] = pipeline
        
        # Convert history to messages format for the LLM
        messages = []
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        try:
            # Generate response
            result = pipeline.chat(messages)
            response = result["response"]
            
            # Update pipeline state
            pipeline_state["chat_history"] = messages + [{"role": "assistant", "content": response}]
            
            return response, history + [[message, response]], pipeline_state
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            return error_message, history + [[message, error_message]], pipeline_state
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface for the chat component.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=600)
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Enter your question here",
                            show_label=False,
                            container=False
                        )
                        submit_btn = gr.Button("Send")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat")
                        
                with gr.Column(scale=1):
                    config_dropdown = gr.Dropdown(
                        choices=list(self.configs.keys()),
                        value="default",
                        label="Pipeline Configuration"
                    )
                    
                    file_upload = gr.File(
                        file_count="multiple",
                        label="Upload Documents"
                    )
                    
                    load_docs_btn = gr.Button("Load Documents")
                    docs_status = gr.Textbox(label="Document Status", interactive=False)
            
            # Hidden state for the pipeline
            pipeline_state = gr.State({})
            
            # Event handlers
            submit_btn.click(
                fn=self.chat,
                inputs=[msg, chatbot, config_dropdown, pipeline_state],
                outputs=[chatbot, chatbot, pipeline_state],
                queue=False
            ).then(
                lambda: "",
                None,
                msg,
                queue=False
            )
            
            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot, config_dropdown, pipeline_state],
                outputs=[chatbot, chatbot, pipeline_state],
                queue=False
            ).then(
                lambda: "",
                None,
                msg,
                queue=False
            )
            
            clear_btn.click(
                lambda: ([], {"config_name": None}),
                None,
                [chatbot, pipeline_state],
                queue=False
            )
            
            def handle_load_docs(files, config_name, pipeline_state):
                if not pipeline_state or pipeline_state.get("config_name") != config_name:
                    try:
                        pipeline = self.create_pipeline(config_name)
                        pipeline_state = {
                            "config_name": config_name,
                            "documents_loaded": False,
                            "chat_history": []
                        }
                    except Exception as e:
                        return f"Error creating pipeline: {str(e)}", pipeline_state
                else:
                    pipeline = self.pipelines.get(config_name)
                    if not pipeline:
                        try:
                            pipeline = self.create_pipeline(config_name)
                        except Exception as e:
                            return f"Error creating pipeline: {str(e)}", pipeline_state
                
                self.pipelines[config_name] = pipeline
                
                try:
                    file_paths = [f.name for f in files]
                    status = self.load_documents(pipeline, file_paths)
                    pipeline_state["documents_loaded"] = True
                    return status, pipeline_state
                except Exception as e:
                    return f"Error loading documents: {str(e)}", pipeline_state
            
            load_docs_btn.click(
                fn=handle_load_docs,
                inputs=[file_upload, config_dropdown, pipeline_state],
                outputs=[docs_status, pipeline_state],
                queue=False
            )
            
        return interface
