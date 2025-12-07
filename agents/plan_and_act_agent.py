"""
Plan-and-Act Agent - plans at first step, then executes.
Based on ReflAct paper (arXiv:2505.15182) - Section 6.2
Inspired by Plan-and-Solve (Wang et al., 2023)
"""
from agents.base_agent import BaseAgent
from prompts.base import SYSTEM_INSTRUCTION, AVAILABLE_ACTIONS, REMINDER
from prompts.plan_and_act import (
    PLAN_AND_ACT_INSTRUCTION_FIRST,
    PLAN_AND_ACT_INSTRUCTION_LATER,
    PLAN_AND_ACT_EXAMPLE
)


class PlanAndActAgent(BaseAgent):
    """
    Plan-and-Act Agent that plans at the first step and then executes.
    
    From the paper: "The agent is instructed at the first timestep (t=1) to 
    'plan your approach to the task, and then output the action.' It performs 
    reasoning only at this step and outputs actions without further thoughts 
    in subsequent timesteps."
    
    Format: 
    - Step 1: Plan: ... -> Action: ...
    - Step 2+: Action: ...
    """
    
    agent_type = "plan_and_act"
    
    def get_system_prompt(self) -> str:
        """Return complete system prompt with instruction and one-shot example."""
        return f"""{SYSTEM_INSTRUCTION}

{PLAN_AND_ACT_INSTRUCTION_FIRST}

{AVAILABLE_ACTIONS}

{REMINDER}

Here is an example:
{PLAN_AND_ACT_EXAMPLE}"""
    
    def get_instruction(self, step: int) -> str:
        """Return instruction based on step number."""
        if step == 1:
            return PLAN_AND_ACT_INSTRUCTION_FIRST
        else:
            return PLAN_AND_ACT_INSTRUCTION_LATER
