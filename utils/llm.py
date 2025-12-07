"""
LLM utilities for calling OpenAI API.
"""
import re
from openai import OpenAI
import config


def get_client():
    """Get OpenAI client with optional base_url for proxy/relay API."""
    return OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL
    )


def call_llm(messages: list, temperature: float = None) -> str:
    """
    Call the LLM with given messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature (default from config)
    
    Returns:
        The assistant's response text
    """
    client = get_client()
    
    if temperature is None:
        temperature = config.TEMPERATURE
    
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=512,
    )
    
    return response.choices[0].message.content.strip()


def parse_action(response: str) -> str:
    """
    Parse the action from LLM response.
    
    The response may contain:
    - Just the action
    - Thought: ... followed by Action: ...
    - Reflection: ... followed by Action: ...
    - Plan: ... followed by Action: ...
    
    Args:
        response: Raw LLM response
    
    Returns:
        Extracted action string
    """
    # Try to find "Action: ..." pattern
    action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if action_match:
        return action_match.group(1).strip()
    
    # If no "Action:" prefix, try to extract from the last line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.lower().startswith(('thought:', 'reflection:', 'plan:', 'observation:')):
            # Remove any prefix like "Action:" if present
            if ':' in line:
                parts = line.split(':', 1)
                if parts[0].lower() in ['action']:
                    return parts[1].strip()
            return line
    
    # Fallback: return the whole response
    return response.strip()


def parse_thought_or_reflection(response: str) -> tuple:
    """
    Parse both the reasoning (thought/reflection/plan) and action from response.
    
    Returns:
        Tuple of (reasoning_type, reasoning_content, action)
        reasoning_type is one of: 'thought', 'reflection', 'plan', None
    """
    reasoning_type = None
    reasoning_content = None
    action = None
    
    # Try to find Thought
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.IGNORECASE | re.DOTALL)
    if thought_match:
        reasoning_type = 'thought'
        reasoning_content = thought_match.group(1).strip()
    
    # Try to find Reflection
    reflection_match = re.search(r'Reflection:\s*(.+?)(?=Action:|$)', response, re.IGNORECASE | re.DOTALL)
    if reflection_match:
        reasoning_type = 'reflection'
        reasoning_content = reflection_match.group(1).strip()
    
    # Try to find Plan
    plan_match = re.search(r'Plan:\s*(.+?)(?=Action:|$)', response, re.IGNORECASE | re.DOTALL)
    if plan_match:
        reasoning_type = 'plan'
        reasoning_content = plan_match.group(1).strip()
    
    # Parse action
    action = parse_action(response)
    
    return reasoning_type, reasoning_content, action

