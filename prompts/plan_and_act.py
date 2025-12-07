"""
Plan-and-Act prompts - Agent plans at first step, then executes.
Based on ReflAct paper (arXiv:2505.15182) - Section 6.2

Note: The paper describes Plan-and-Act as:
"The agent is instructed at the first timestep (t=1) to 'plan your approach to the task, 
and then output the action.' It performs reasoning only at this step and outputs actions 
without further thoughts in subsequent timesteps."
"""

# Instruction for Plan-and-Act at first step (from paper Section 6.2)
PLAN_AND_ACT_INSTRUCTION_FIRST = """For each of your turn, you will be given the observation of the last turn. For this first turn, you should first plan your approach to the task, and then output your action. Your output must strictly follow this format:"Plan: your plan.\\n Action: your next action"."""

# Instruction for Plan-and-Act at later steps (same as NoThinking)
PLAN_AND_ACT_INSTRUCTION_LATER = """For each of your turn, you will be given the observation of the last turn. You should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action"."""

# One-shot example for Plan-and-Act - "pick and place" task
# First turn has Plan, later turns have only Action
PLAN_AND_ACT_EXAMPLE = """You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put some spraybottle on toilet.
Plan: To solve the task, I need to find and take a sparybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet 1-4, countertop 1, toilet 1, sinkbasin 1-2, garbagecan 1. I can check one by one, starting by going to cabinet 1.
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
Observation: You put the spraybottle 2 in/on the toilet 1."""
