"""
ReAct Agent - thinks about next action before acting.
Based on ReAct paper (Yao et al., 2023) and ReflAct paper (arXiv:2505.15182)
"""
from agents.base_agent import BaseAgent
from prompts.base import ALFWORLD_SYSTEM_PROMPT
from prompts.react import REACT_INSTRUCTION, REACT_EXAMPLE


class ReActAgent(BaseAgent):
    """
    ReAct Agent that reasons about the next action before acting.
    
    From the paper: "The agent first reasons about the next action at each 
    time step and then generates an action."
    
    Format: Thought: ... -> Action: ...
    """
    
    agent_type = "react"
    
    def get_system_prompt(self) -> str:
        """Return system prompt with one-shot example."""
        return ALFWORLD_SYSTEM_PROMPT + "\n" + REACT_EXAMPLE
    
    def get_instruction(self, step: int) -> str:
        """Return instruction for ReAct (same for all steps)."""
        return REACT_INSTRUCTION

