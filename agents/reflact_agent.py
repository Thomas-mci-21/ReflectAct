"""
ReflAct Agent - reflects on state in relation to goal before acting.
Based on ReflAct paper (arXiv:2505.15182) - Section 4

Key insight: "We introduce ReflAct, a novel backbone that shifts reasoning from 
merely planning next actions to continuously reflecting on the agent's state 
relative to its goal."
"""
from agents.base_agent import BaseAgent
from prompts.base import ALFWORLD_SYSTEM_PROMPT
from prompts.reflact import REFLACT_INSTRUCTION, REFLACT_EXAMPLE


class ReflActAgent(BaseAgent):
    """
    ReflAct Agent that reflects on current state in relation to goal.
    
    From the paper: "You should first reflect on the agent's state in relation 
    to the task goal, and then output the action for this turn."
    
    Key difference from ReAct:
    - ReAct Thought: "Now I find a spraybottle 2. Next, I need to take it."
    - ReflAct Reflection: "Currently, I am at cabinet 2 and have found a 
      spraybottle 2, which brings me closer to completing the task of placing 
      it on the toilet."
    
    ReflAct explicitly encodes:
    1. Current state/location (belief state)
    2. Progress toward the goal
    3. Relationship between state and goal
    """
    
    agent_type = "reflact"
    
    def get_system_prompt(self) -> str:
        """Return system prompt with one-shot example."""
        return ALFWORLD_SYSTEM_PROMPT + "\n" + REFLACT_EXAMPLE
    
    def get_instruction(self, step: int) -> str:
        """Return instruction for ReflAct (same for all steps)."""
        return REFLACT_INSTRUCTION

