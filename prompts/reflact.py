"""
ReflAct prompts - Agent reflects on state in relation to goal before acting.
Based on ReflAct paper (arXiv:2505.15182) - Appendix K.1.2
2-shot examples: Figure 16 + Figure 18
"""

# Instruction for ReflAct (exact wording from paper Appendix K.1.1)
REFLACT_INSTRUCTION = """For each of your turn, you will be given the observation of the last turn. You should first reflect in one sentence on the agent's state in relation to the task goal, and then output the action for this turn. Your output must strictly follow this format:"Reflection: your reflection.\\n Action: your next action"."""

# 2-shot examples for ReflAct (from paper Figure 16 + Figure 18)

# Example 1: "pick and place" task (Figure 16)
REFLACT_EXAMPLE_1 = """You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
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

# Example 2: "pick clean then place" task (Figure 18)
REFLACT_EXAMPLE_2 = """You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put a clean lettuce in diningtable.
Reflection: To solve the task, I need to find and take a lettuce, then clean it with sinkbasin, then put it in diningtable. First I need to find a lettuce. A lettuce is more likely to appear in fridge 1, diningtable 1, sinkbasin 1, stoveburner 1-3, cabinet 1-13. I can check one by one, starting by going to fridge 1.
Action: go to fridge 1
Observation: The fridge 1 is closed.
Reflection: Currently, I am in front of the closed fridge 1, not holding anything, and need to open it to search for lettuce and continue the task.
Action: open fridge 1
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.
Reflection: Currently, I am in front of open fridge 1, not holding any object, and still need to find the lettuce to complete the task.
Action: go to diningtable 1
Observation: On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
Reflection: Currently, I am at dining table 1 and have found lettuce 1, which I need to clean before placing it on the dining table to complete the task.
Action: take lettuce 1 from diningtable 1
Observation: You pick up the lettuce 1 from the diningtable 1.
Reflection: Currently, I am at dining table 1, now holding lettuce 1, which I need to clean before placing it on the dining table to complete the task.
Action: go to sinkbasin 1
Observation: On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.
Reflection: Currently, I am at sinkbasin 1, I'm holding a lettuce 1, and I am now ready to clean it before placing it on the dining table to complete my task.
Action: clean lettuce 1 with sinkbasin 1
Observation: You clean the lettuce 1 using the sinkbasin 1.
Reflection: Currently, I am at sinkbasin 1 and I am holding a clean lettuce 1, while the task of putting the clean lettuce on the dining table is almost complete as I just need to return to the dining table.
Action: go to diningtable 1
Observation: On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.
Reflection: Currently, I am at diningtable 1 and I am holding a clean lettuce 1, while the task of putting the clean lettuce on the dining table is now complete as I just need to place it down.
Action: put lettuce 1 in/on diningtable 1
Observation: You put the lettuce 1 in/on the diningtable 1."""

# Combined 2-shot example
REFLACT_EXAMPLE = f"""Example 1:
{REFLACT_EXAMPLE_1}

Example 2:
{REFLACT_EXAMPLE_2}"""
