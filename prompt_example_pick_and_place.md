# "pick_and_place" 任务的完整 Prompt 示例

## 完整 Prompt（字符串格式）

```
Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. 
For each of your turn, you will be given the observation of the last turn. You should choose from two actions: "Thought" or "Action". If you choose "Thought", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\n Action: your next action"; If you choose "Action", you should directly output the action in this turn. Your output must strictly follow this format:"Action: your next action". 
The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. toggle {obj} {recep}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.
Reminder: 
1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Think when necessary, try to act directly more in the process.
---
Here is an example.

You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Your task is to: put some spraybottle on toilet.

Thought: To solve the task, I need to find and take a sparybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.
Action: go to cabinet 1

Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.

Action: go to cabinet 2

Observation: The cabinet 2 is closed.

Action: open cabinet 2

Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.

Thought: Now I find a spraybottle (2). Next, I need to take it.
Action: take spraybottle 2 from cabinet 2

Observation: You pick up the spraybottle 2 from the cabinet 2.

Thought: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.
Action: go to toilet 1

Observation: On the toilet 1, you see a soapbottle 2.

Action: put spraybottle 2 in/on toilet 1

Observation: You put the spraybottle 2 in/on the toilet 1.

---

Now, it's your turn and here is the task.
You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put some apple in fridge.
```

---

## Messages 格式（用于 API 调用）

```json
[
  {
    "role": "user",
    "content": "Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. \nFor each of your turn, you will be given the observation of the last turn. You should choose from two actions: \"Thought\" or \"Action\". If you choose \"Thought\", you should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:\"Thought: your thoughts.\\n Action: your next action\"; If you choose \"Action\", you should directly output the action in this turn. Your output must strictly follow this format:\"Action: your next action\". \nThe available actions are:\n1. go to {recep}\n2. take {obj} from {recep}\n3. put {obj} in/on {recep}\n4. open {recep}\n5. close {recep}\n6. toggle {obj} {recep}\n7. clean {obj} with {recep}\n8. heat {obj} with {recep}\n9. cool {obj} with {recep}\nwhere {obj} and {recep} correspond to objects and receptacles.\nAfter your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output \"Nothing happened\", that means the previous action is invalid and you should try more options.\nReminder: \n1. The action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.\n2. Think when necessary, try to act directly more in the process."
  },
  {
    "role": "assistant",
    "content": "OK"
  },
  {
    "role": "user",
    "content": "You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\nYour task is to: put some spraybottle on toilet."
  },
  {
    "role": "assistant",
    "content": "Thought: To solve the task, I need to find and take a sparybottle, then put it on toilet. First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.\nAction: go to cabinet 1"
  },
  {
    "role": "user",
    "content": "Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1."
  },
  {
    "role": "assistant",
    "content": "Action: go to cabinet 2"
  },
  {
    "role": "user",
    "content": "Observation: The cabinet 2 is closed."
  },
  {
    "role": "assistant",
    "content": "Action: open cabinet 2"
  },
  {
    "role": "user",
    "content": "Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2."
  },
  {
    "role": "assistant",
    "content": "Thought: Now I find a spraybottle (2). Next, I need to take it.\nAction: take spraybottle 2 from cabinet 2"
  },
  {
    "role": "user",
    "content": "Observation: You pick up the spraybottle 2 from the cabinet 2."
  },
  {
    "role": "assistant",
    "content": "Thought: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.\nAction: go to toilet 1"
  },
  {
    "role": "user",
    "content": "Observation: On the toilet 1, you see a soapbottle 2."
  },
  {
    "role": "assistant",
    "content": "Action: put spraybottle 2 in/on toilet 1"
  },
  {
    "role": "user",
    "content": "Observation: You put the spraybottle 2 in/on the toilet 1."
  },
  {
    "role": "user",
    "content": "You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\nYour task is to: put some apple in fridge."
  }
]
```

---

## Prompt 构建逻辑说明

### 1. Instruction（指令部分）
- **来源**：`prompt/instructions/alfworld_inst.txt`
- **内容**：定义任务目标、输出格式、可用动作、提醒事项

### 2. Few-shot Example（示例部分）
- **来源**：`prompt/icl_examples/alfworld_icl.json` 中的 `"pick_and_place"` 键
- **选择逻辑**：根据 `task.task_type` 选择对应的示例列表
- **使用数量**：`icl_num=1`（MPO 对齐，只使用第一个示例）
- **格式**：完整的对话历史（user/assistant 交替）

### 3. Current Task（当前任务）
- **来源**：从 ALFWorld 环境获取的初始观察和任务描述
- **格式**：`"You are in the middle of a room. Looking quickly around you, you see ...\nYour task is to: ..."`

### 4. 构建函数
- **函数**：`prompt_with_icl()` 在 `prompt/templates.py` 中
- **模板**：`PROMPT_WITH_ICL_TEMPLATE`
- **返回**：
  - `prompt`：完整的字符串 prompt（用于调试/日志）
  - `messages`：消息列表（用于 API 调用）

---

## 关键特点

1. **MPO 对齐**：使用 `icl_num=1`，只包含 1 个 few-shot 示例
2. **任务类型匹配**：根据 `task.task_type` 选择对应的示例类型
3. **格式统一**：所有 prompt 都遵循相同的模板结构
4. **消息格式**：第一个 user message 是 instruction，然后是示例对话，最后是当前任务

