"""
ReAct prompts - Agent reasons about next action before acting.
Based on ReAct paper (Yao et al., 2023) and ReflAct paper (arXiv:2505.15182)
"""

# Instruction for ReAct (from paper)
# "The agent first reasons about the next action at each time step and then generates an action."
REACT_INSTRUCTION = """Think about what to do next, and then output the action."""

# One-shot example for ReAct (from original ReAct paper for ALFWorld)
REACT_EXAMPLE = """Your task is: put some spraybottle on toilet.
Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Thought: I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). Let me start with cabinet 1.
Action: go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Thought: I don't see a spraybottle here. Let me check cabinet 2.
Action: go to cabinet 2
Observation: The cabinet 2 is closed.
Thought: I need to open cabinet 2 to see what's inside.
Action: open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Thought: I found a spraybottle 2. I need to take it.
Action: take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Thought: Now I have the spraybottle. I need to put it on the toilet.
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Thought: I'm at the toilet. Now I need to put the spraybottle on it.
Action: put spraybottle 2 in/on toilet 1
Observation: You put the spraybottle 2 in/on the toilet 1.
"""

