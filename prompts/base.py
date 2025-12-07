"""
Base prompts for ALFWorld - shared across all agents.
Based on ReflAct paper (arXiv:2505.15182) - Appendix K.1.1
"""

# System instruction shared by all agents (from Appendix K.1.1 Figure 14)
SYSTEM_INSTRUCTION = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish."""

# Available actions list (from Appendix K.1.1 Figure 14)
AVAILABLE_ACTIONS = """The available actions are:
1. go to recep
2. take obj from recep
3. put obj in/on recep
4. open recep
5. close recep
6. use obj
7. clean obj with recep
8. heat obj with recep
9. cool obj with recep
where obj and recep correspond to objects and receptacles."""

# Reminder section (from Appendix K.1.1 Figure 14)
REMINDER = """After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.
Reminder:
The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal."""
