"""
Prompt templates for different reasoning strategies.
Based on the ReflAct paper (arXiv:2505.15182)
"""

from prompts.base import ALFWORLD_SYSTEM_PROMPT, get_task_prompt
from prompts.nothinking import NOTHINKING_INSTRUCTION, NOTHINKING_EXAMPLE
from prompts.react import REACT_INSTRUCTION, REACT_EXAMPLE
from prompts.plan_and_act import PLAN_AND_ACT_INSTRUCTION_FIRST, PLAN_AND_ACT_INSTRUCTION_LATER, PLAN_AND_ACT_EXAMPLE
from prompts.reflact import REFLACT_INSTRUCTION, REFLACT_EXAMPLE

__all__ = [
    "ALFWORLD_SYSTEM_PROMPT",
    "get_task_prompt",
    "NOTHINKING_INSTRUCTION",
    "NOTHINKING_EXAMPLE",
    "REACT_INSTRUCTION",
    "REACT_EXAMPLE",
    "PLAN_AND_ACT_INSTRUCTION_FIRST",
    "PLAN_AND_ACT_INSTRUCTION_LATER",
    "PLAN_AND_ACT_EXAMPLE",
    "REFLACT_INSTRUCTION",
    "REFLACT_EXAMPLE",
]

