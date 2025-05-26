"""
Main Gradio application for the RAG evaluation framework.
"""
import gradio as gr
import os
import sys

from ui.components.chat_interface import ChatInterface
from ui.components.evaluation_interface import EvaluationInterface


def create_app(config_path: str = None):
    """
    Create the Gradio application.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        Gradio Blocks application
    """
    # Initialize components
    chat_interface = ChatInterface(config_path)
    evaluation_interface = EvaluationInterface(config_path)
    
    # Create the application
    with gr.Blocks(title="RAG Evaluation Framework") as app:
        with gr.Tabs():
            with gr.TabItem("Chat"):
                chat_ui = chat_interface.build_interface()
            
            with gr.TabItem("Evaluation"):
                eval_ui = evaluation_interface.build_interface()
        
        # Add footer
        gr.Markdown("""
        ## RAG Evaluation Framework
        
        This application allows you to test and evaluate different RAG pipeline strategies.
        
        - Use the **Chat** tab to interact with different pipeline configurations
        - Use the **Evaluation** tab to run evaluations and generate reports
        """)
    
    return app


def main():
    """
    Main entry point for the application.
    """
    # Check for config file path in arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using default configurations.")
            config_path = None
    
    # Create and launch the app
    app = create_app(config_path)
    app.launch(share=False)


if __name__ == "__main__":
    main()
