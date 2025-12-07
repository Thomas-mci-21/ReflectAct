"""
ReAct Agent - thinks about next action before acting.
Based on ReAct paper (Yao et al., 2023) and ReflAct paper (arXiv:2505.15182) - Appendix K.1.1
"""
from agents.base_agent import BaseAgent
from prompts.base import SYSTEM_INSTRUCTION, AVAILABLE_ACTIONS, REMINDER
from prompts.react import REACT_INSTRUCTION, REACT_EXAMPLE


class ReActAgent(BaseAgent):
    """
    ReAct Agent that reasons about the next action before acting.
    
    From the paper: "The agent first thinks about the current condition and 
    plans for future actions, and then generates an action."
    
    Format: Thought: ... -> Action: ...
    """
    
    agent_type = "react"
    
    def get_system_prompt(self) -> str:
        """Return complete system prompt with instruction and one-shot example."""
        return f"""{SYSTEM_INSTRUCTION}

{REACT_INSTRUCTION}

{AVAILABLE_ACTIONS}

{REMINDER}

Here is an example:
{REACT_EXAMPLE}"""
    
    def get_instruction(self, step: int) -> str:
        """Return instruction for ReAct (same for all steps)."""
        return REACT_INSTRUCTION
