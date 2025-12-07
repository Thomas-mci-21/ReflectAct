"""
ReflAct prompts - Agent reflects on state in relation to goal before acting.
Based on ReflAct paper (arXiv:2505.15182) - Section 4

Key insight from paper:
"Instruction for ReflAct: You should first reflect on the agent's state in relation to 
the task goal, and then output the action for this turn."

The paper notes the difference between Thought and Reflection:
- Thought: "Now I find a spraybottle 2. Next, I need to take it."
- Reflection: "Currently, I am at cabinet 2 and have found a spraybottle 2, which brings 
  me closer to completing the task of placing it on the toilet."
"""

# Instruction for ReflAct (from paper - exact wording from Section 4)
REFLACT_INSTRUCTION = """You should first reflect on the agent's state in relation to the task goal, and then output the action for this turn."""

# One-shot example for ReflAct (from paper - Appendix K style)
# Modified from ReAct example with Reflections that capture:
# 1. Current state/location
# 2. Progress toward the goal
# 3. Relationship between state and goal
REFLACT_EXAMPLE = """Your task is: put some spraybottle on toilet.
Observation: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Reflection: I am currently in the middle of the room. My goal is to put a spraybottle on the toilet. I haven't found a spraybottle yet, so I need to search the room. Cabinets are likely places to find a spraybottle.
Action: go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Reflection: I am at cabinet 1 and can see its contents. There is no spraybottle here, only a cloth, soapbar, and soapbottle. I need to continue searching other cabinets to find a spraybottle for my task.
Action: go to cabinet 2
Observation: The cabinet 2 is closed.
Reflection: I am at cabinet 2 but it is closed. I cannot see what's inside until I open it. Opening it may reveal the spraybottle I need for my task.
Action: open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Reflection: I have opened cabinet 2 and found a spraybottle 2 inside. This is exactly what I need for my task. I should take it so I can put it on the toilet.
Action: take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Reflection: I am now holding the spraybottle 2. I have completed the first part of my task (finding and taking the spraybottle). Now I need to go to the toilet and place the spraybottle there to complete the task.
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Reflection: I am at the toilet 1 and I have the spraybottle 2 in my possession. I can now complete my task by putting the spraybottle on the toilet.
Action: put spraybottle 2 in/on toilet 1
Observation: You put the spraybottle 2 in/on the toilet 1.
"""

