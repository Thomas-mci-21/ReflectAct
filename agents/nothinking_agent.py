"""
NoThinking Agent - generates action directly without reasoning.
Based on ReflAct paper (arXiv:2505.15182) - Appendix K.1.1
"""
from agents.base_agent import BaseAgent
from prompts.base import SYSTEM_INSTRUCTION, AVAILABLE_ACTIONS, REMINDER
from prompts.nothinking import NOTHINKING_INSTRUCTION, NOTHINKING_EXAMPLE


class NoThinkingAgent(BaseAgent):
    """
    NoThinking Agent that generates actions without intermediate reasoning.
    
    From the paper: "The agent generates an action directly at each time step 
    without any reasoning step."
    """
    
    agent_type = "nothinking"
    
    def get_system_prompt(self) -> str:
        """Return complete system prompt with instruction and one-shot example."""
        return f"""{SYSTEM_INSTRUCTION}

{NOTHINKING_INSTRUCTION}

{AVAILABLE_ACTIONS}

{REMINDER}

Here is an example:
{NOTHINKING_EXAMPLE}"""
    
    def get_instruction(self, step: int) -> str:
        """Return instruction for NoThinking (same for all steps)."""
        return NOTHINKING_INSTRUCTION
