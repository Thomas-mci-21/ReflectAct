"""
NoThinking prompts - Agent generates action directly without reasoning.
Based on ReflAct paper (arXiv:2505.15182) - Section 6.2
"""

# Instruction for NoThinking (from paper)
# "The agent generates an action directly at each time step without any reasoning step."
NOTHINKING_INSTRUCTION = """Output the action for this turn."""

# One-shot example for NoThinking (based on ReAct paper's ALFWorld example, without Thought)
NOTHINKING_EXAMPLE = """Your task is: put some spraybottle on toilet.
Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
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

