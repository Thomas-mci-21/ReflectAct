"""
ReAct prompts - Agent reasons about next action before acting.
Based on ReAct paper (Yao et al., 2023) and ReflAct paper (arXiv:2505.15182) - Appendix K.1.2
2-shot examples: Figure 15 + Figure 17
"""

# Instruction for ReAct (exact wording from paper Appendix K.1.1)
REACT_INSTRUCTION = """For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\\n Action: your next action"."""

# 2-shot examples for ReAct (from paper Figure 15 + Figure 17)

# Example 1: "pick and place" task (Figure 15)
REACT_EXAMPLE_1 = """You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put some spraybottle on toilet.
Thought: To solve the task, I need to find and take a sparybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet 1-4, countertop 1, toilet 1, sinkbasin 1-2, garbagecan 1. I can check one by one, starting by going to cabinet 1.
Action: go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Action: go to cabinet 2
Observation: The cabinet 2 is closed.
Action: open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Thought: Now I find a spraybottle 2. Next, I need to take it.
Action: take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Thought: Now I take a spraybottle 2. Next, I need to put it in/on toilet 1.
Action: go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Action: put spraybottle 2 in/on toilet 1
Observation: You put the spraybottle 2 in/on the toilet 1."""

# Example 2: "pick clean then place" task (Figure 17)
REACT_EXAMPLE_2 = """You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put a clean lettuce in diningtable.
Thought: To solve the task, I need to find and take a lettuce, then clean it with sinkbasin, then put it in diningtable. First I need to find a lettuce. A lettuce is more likely to appear in fridge 1, diningtable 1, sinkbasin 1, stoveburner 1-3, cabinet 1-13. I can check one by one, starting by going to fridge 1.
Action: go to fridge 1
Observation: The fridge 1 is closed.
Action: open fridge 1
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.
Action: go to diningtable 1
Observation: On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
Thought: Now I find a lettuce 1. Next, I need to take it.
Action: take lettuce 1 from diningtable 1
Observation: You pick up the lettuce 1 from the diningtable 1.
Thought: Now I take a lettuce 1. Next, I need to go to sinkbasin 1 and clean it.
Action: go to sinkbasin 1
Observation: On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.
Action: clean lettuce 1 with sinkbasin 1
Observation: You clean the lettuce 1 using the sinkbasin 1.
Thought: Now I clean a lettuce 1. Next, I need to put it in/on diningtable 1.
Action: go to diningtable 1
Observation: On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
Action: put lettuce 1 in/on diningtable 1
Observation: You put the lettuce 1 in/on the diningtable 1."""

# Combined 2-shot example
REACT_EXAMPLE = f"""Example 1:
{REACT_EXAMPLE_1}

Example 2:
{REACT_EXAMPLE_2}"""
