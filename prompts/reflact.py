"""
ReflAct prompts - Agent reflects on state in relation to goal before acting.
Based on ReflAct paper (arXiv:2505.15182) - Appendix K.1.1 Figure 14 & Figure 16
"""

# Instruction for ReflAct (exact wording from paper Appendix K.1.1)
REFLACT_INSTRUCTION = """For each of your turn, you will be given the observation of the last turn. You should first reflect in one sentence on the agent's state in relation to the task goal, and then output the action for this turn. Your output must strictly follow this format:"Reflection: your reflection.\\n Action: your next action"."""

# One-shot example for ReflAct - "pick and place" task (from paper Figure 16)
REFLACT_EXAMPLE = """You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put some spraybottle on toilet.
Reflection: To solve the task, I need to find and take a sparybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet 1-4, countertop 1, toilet 1, sinkbasin 1-2, garbagecan 1. I can check one by one, starting by going to cabinet 1.
Action: go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Reflection: Currently, I am at cabinet 1, not holding anything, and am searching for a spraybottle to place on the toilet, but instead I find a cloth, a soapbar, and a soapbottle.
Action: go to cabinet 2
Observation: The cabinet 2 is closed.
Reflection: Currently, I am at cabinet 2, not holding anything, but the closed cabinet hinders me from finding the spraybottle needed to place on the toilet.
Action: open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Reflection: Currently, I am at cabinet 2 and have found a spraybottle 2, which brings me closer to completing the task of placing it on the toilet.
Action: take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Reflection: Currently I am located at cabinet 2, now holding a spraybottle 2, and I am ready to complete the task of placing the spraybottle on the toilet, as I have obtained spraybottle 2.
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Reflection: Currently, I am at toilet 1, still holding spraybottle 2, and the task is nearly complete, with only the placement remaining.
Action: put spraybottle 2 in/on toilet 1
Observation: You put the spraybottle 2 in/on the toilet 1."""
