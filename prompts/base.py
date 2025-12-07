"""
Base prompt components for ALFWorld tasks.
Based on the original ReAct paper prompts and ReflAct paper.
"""

# System prompt for ALFWorld (from ReAct paper)
ALFWORLD_SYSTEM_PROMPT = """Interact with a household to solve a task. Here are some examples.
"""

# Available actions in ALFWorld
ALFWORLD_ACTIONS = """
Available actions:
- go to {recep}
- take {obj} from {recep}
- put {obj} in/on {recep}
- open {recep}
- close {recep}
- toggle {obj} {recep}
- clean {obj} with {recep}
- heat {obj} with {recep}
- cool {obj} with {recep}
- use {recep}
- look
- inventory
- examine {obj/recep}
"""

def get_task_prompt(task_description: str) -> str:
    """Format the task description for the agent."""
    return f"Your task is: {task_description}"


def format_observation(obs: str) -> str:
    """Format an observation from the environment."""
    return f"Observation: {obs}"


def format_action(action: str) -> str:
    """Format an action for the environment."""
    return f"Action: {action}"

