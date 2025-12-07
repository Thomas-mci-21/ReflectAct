"""
ReAct Agent - MPO-aligned implementation.
Completely aligned with MPO: Agent is just an LLM caller.
"""
import logging
import os
import backoff
import openai
from openai import OpenAI
import config

logger = logging.getLogger("agent_eval")


class ReActAgent:
    """
    ReAct Agent aligned with MPO implementation.
    
    Key characteristics (matching MPO):
    - Agent is just an LLM caller
    - No state management
    - No action parsing
    - Simple __call__(messages) -> str interface
    """
    
    agent_type = "react"
    
    def __init__(self, config_dict: dict = None, verbose: bool = True):
        """
        Initialize agent.
        
        Args:
            config_dict: Agent config dict (for compatibility with MPO)
            verbose: Whether to print to terminal
        """
        self.verbose = verbose
        
        # Use config_dict if provided (MPO style), otherwise use config module
        if config_dict:
            self.config = config_dict
            self.client = OpenAI(
                base_url=config_dict.get("api_base", None),
                api_key=config_dict.get("api_key", os.environ.get("OPENAI_API_KEY")),
            )
            self.model_name = config_dict["model_name"]
            self.max_completion_tokens = config_dict.get("max_completion_tokens", 512)
            self.temperature = config_dict.get("temperature", 0)
        else:
            # ReflAct style (use config module)
            self.client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL
            )
            self.model_name = config.OPENAI_MODEL
            self.max_completion_tokens = 512
            self.temperature = config.TEMPERATURE
        
        self.stop_words = ["\nObservation:", "\nTask:", "\n---"]
    
    @backoff.on_exception(
        backoff.fibo,
        (
            openai.APIError,
            openai.Timeout,
            openai.RateLimitError,
            openai.APIConnectionError,
        ),
    )
    def __call__(self, messages) -> str:
        """
        Call LLM with messages, return raw LLM output.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Raw LLM output string
        """
        # Note: OpenAI API uses 'max_tokens', but MPO uses 'max_completion_tokens'
        # Using max_tokens for standard OpenAI SDK compatibility
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_completion_tokens,  # Standard OpenAI parameter
            temperature=self.temperature,
            stop=self.stop_words,
        )
        return response.choices[0].message.content
