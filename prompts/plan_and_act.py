"""
Plan-and-Act prompts - Agent plans at first step, then executes.
Based on ReflAct paper (arXiv:2505.15182) - Section 6.2
Inspired by Plan-and-Solve (Wang et al., 2023)
"""

# Instruction for Plan-and-Act (from paper)
# "The agent is instructed at the first timestep (t=1) to 'plan your approach to the task, 
#  and then output the action.' It performs reasoning only at this step and outputs actions 
#  without further thoughts in subsequent timesteps."
PLAN_AND_ACT_INSTRUCTION_FIRST = """Plan your approach to the task, and then output the action."""
PLAN_AND_ACT_INSTRUCTION_LATER = """Output the action for this turn."""

# One-shot example for Plan-and-Act (based on ReAct example, modified)
PLAN_AND_ACT_EXAMPLE = """Your task is: put some spraybottle on toilet.
Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Plan: To complete this task, I need to: 1) Search for a spraybottle in likely locations (cabinets, countertop, sinkbasin), 2) Take the spraybottle once found, 3) Go to the toilet, 4) Put the spraybottle on the toilet. I'll start by checking the cabinets.
Action: go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Action: go to cabinet 2
Observation: The cabinet 2 is closed.
Action: open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Action: take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Action: put spraybottle 2 in/on toilet 1
Observation: You put the spraybottle 2 in/on the toilet 1.
"""

