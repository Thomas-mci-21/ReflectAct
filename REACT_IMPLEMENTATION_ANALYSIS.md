# ReAct Agent 在 ALFWorld 上的实现分析 (ReflAct 代码库)

## 📋 概述

这是一个基于 ReflAct 论文的代码库，实现了多种 Agent 方法（NoThinking, ReAct, Plan-and-Act, ReflAct），这里重点分析 **ReAct Agent 的实现**。

## 🏗️ 架构概览

```
reflectact/
├── agents/
│   ├── base_agent.py        # 基类，定义通用逻辑
│   └── react_agent.py       # ReAct Agent 实现
├── prompts/
│   ├── base.py              # 基础 prompt（指令、动作列表）
│   └── react.py             # ReAct 特定的 prompt（指令 + 2-shot examples）
├── environment/
│   └── alfworld_env.py      # ALFWorld 环境包装
├── utils/
│   ├── llm.py               # LLM API 调用和响应解析
│   └── logger.py            # 日志和结果保存
└── run_experiment.py        # 主实验脚本
```

## 🔑 核心实现组件

### 1. ReAct Agent 类 (`agents/react_agent.py`)

**关键特点**：
- 继承自 `BaseAgent`
- 实现 `get_system_prompt()` 和 `get_instruction()` 方法
- 使用 2-shot examples（来自 ReflAct 论文 Appendix K.1.2）

**核心代码**：

```python
class ReActAgent(BaseAgent):
    agent_type = "react"
    
    def get_system_prompt(self) -> str:
        """返回包含指令和2-shot示例的完整system prompt"""
        return f"""{SYSTEM_INSTRUCTION}

{REACT_INSTRUCTION}

{AVAILABLE_ACTIONS}

{REMINDER}

Here is an example:
{REACT_EXAMPLE}"""
    
    def get_instruction(self, step: int) -> str:
        """每个步骤使用相同的指令"""
        return REACT_INSTRUCTION
```

### 2. Few-shot Examples (`prompts/react.py`)

**来源**：ReflAct 论文 Appendix K.1.2（Figure 15 + Figure 17）

**结构**：
- **Example 1**: `pick_and_place` 任务（Figure 15）
- **Example 2**: `pick_clean_then_place` 任务（Figure 17）
- **固定使用 2 个示例**（不像 MPO 那样按任务类型选择）

**Example 1 示例**：
```
You are in the middle of a room. Looking quickly around you, you see...
Your task is to: put some spraybottle on toilet.
Thought: To solve the task, I need to find and take a sparybottle...
Action: go to cabinet 1
Observation: On the cabinet 1, you see...
Action: go to cabinet 2
...
```

**关键格式**：
- `Thought: ...` → `Action: ...`
- 有时直接 `Action: ...`（没有 Thought）

### 3. System Prompt 构建 (`agents/base_agent.py`)

**消息结构**：

```python
messages = [
    {"role": "system", "content": self.get_system_prompt()}  # ⭐ 使用 system role
]
```

**System Prompt 内容**（来自 `react_agent.py`）：
```
[SYSTEM_INSTRUCTION]          # 基础指令
[REACT_INSTRUCTION]           # ReAct 特定指令
[AVAILABLE_ACTIONS]           # 可用动作列表
[REMINDER]                    # 提醒事项
Here is an example:
[REACT_EXAMPLE]               # 2-shot examples
```

**与 MPO 的区别**：
- ✅ **使用 `role: "system"`**（MPO 将所有内容放在 user message 中）
- ✅ **固定使用 2 个示例**（MPO 按任务类型选择，默认 1 个）

### 4. 对话历史构建 (`agents/base_agent.py:38-69`)

**User Message 构建逻辑**：

```python
user_content = f"Your task is: {self.task_description}\n"

# 添加历史轨迹（每次交互都会累积）
for step_data in self.trajectory:
    user_content += f"Observation: {step_data['observation']}\n"
    if step_data.get('reasoning'):
        reasoning_type = step_data.get('reasoning_type', 'Thought')
        user_content += f"{reasoning_type.capitalize()}: {step_data['reasoning']}\n"
    user_content += f"Action: {step_data['action']}\n"

# 添加当前观察和指令
user_content += f"Observation: {observation}\n"
user_content += self.get_instruction(self.step_count + 1)

messages.append({"role": "user", "content": user_content})
```

**特点**：
- 每次调用都会包含完整的对话历史
- 格式：`Observation` → `Thought` → `Action` → `Observation` → ...
- 自动累积，无需手动管理历史

### 5. LLM 调用 (`utils/llm.py`)

**API 调用**：

```python
def call_llm(messages: list, temperature: float = None) -> str:
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL  # 支持代理/中转
    )
    
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        temperature=temperature or 0,  # 默认贪婪解码
        max_tokens=512,
    )
    
    return response.choices[0].message.content.strip()
```

**响应解析**：

```python
def parse_thought_or_reflection(response: str) -> tuple:
    """解析 Thought/Reflection 和 Action"""
    # 尝试匹配 Thought:
    thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', response, re.IGNORECASE | re.DOTALL)
    
    # 尝试匹配 Action:
    action = parse_action(response)
    
    return reasoning_type, reasoning_content, action
```

### 6. 环境交互 (`environment/alfworld_env.py`)

**初始化**：

```python
class ALFWorldEnv:
    def __init__(self, split: str = "eval_out_of_distribution"):
        # 使用官方 ALFWorld API
        from alfworld.agents.environment import get_environment
        
        config = generic.load_config()  # 从 configs/base_config.yaml 加载
        env_type = config['env']['type']
        
        self.env = get_environment(env_type)(config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
```

**Reset 方法**：

```python
def reset(self, task_idx: Optional[int] = None) -> Tuple[str, str]:
    obs, info = self.env.reset()
    
    # 解析观察和任务描述
    observation = obs[0] if isinstance(obs, list) else obs
    task_description = info.get('extra.goal', [""])[0]
    
    return observation, task_description
```

**Step 方法**：

```python
def step(self, action: str) -> Tuple[str, float, bool, Dict]:
    obs, scores, dones, infos = self.env.step([action])
    
    observation = obs[0] if isinstance(obs, list) else obs
    reward = scores[0] if isinstance(scores, list) else scores
    done = dones[0] if isinstance(dones, list) else dones
    
    return observation, reward, done, infos
```

### 7. 主执行循环 (`agents/base_agent.py:132-171`)

```python
def run_task(self, env, task_id: int = 0) -> Tuple[bool, List[Dict]]:
    # 1. 重置环境和 Agent
    obs, task_desc = env.reset()
    self.reset(task_desc)
    
    # 2. ReAct 循环（最多 MAX_STEPS 步）
    while not done and self.step_count < config.MAX_STEPS:
        action = self.step(obs)           # Agent 生成动作
        obs, reward, done, info = env.step(action)  # 环境执行
        
        if done:
            success = reward > 0
    
    return success, self.trajectory
```

**Step 方法**（单步执行）：

```python
def step(self, observation: str) -> str:
    self.step_count += 1
    
    # 1. 生成动作（包含 Thought）
    action, reasoning_type, reasoning = self.act(observation)
    
    # 2. 记录轨迹
    step_data = {
        "step": self.step_count,
        "observation": observation,
        "action": action,
        "reasoning_type": reasoning_type,
        "reasoning": reasoning,
    }
    self.trajectory.append(step_data)
    
    return action
```

## 📊 与 MPO 实现的对比

| 特性 | ReflAct (reflectact) | MPO |
|------|---------------------|-----|
| **System Prompt** | ✅ 使用 `role: "system"` | ❌ 所有内容在 user message 中 |
| **Few-shot 示例** | 固定 2 个（来自论文） | 按任务类型选择，默认 1 个 |
| **示例来源** | 论文 Figure 15 + 17 | `alfworld_icl.json`（按任务类型组织） |
| **对话历史** | 自动累积在 user message | 自动累积在 `state.history` |
| **Action 解析** | 正则表达式提取 | 正则表达式提取 |
| **环境包装** | 官方 ALFWorld API | 官方 ALFWorld API |
| **配置方式** | `.env` 文件 + `config.py` | JSON 配置文件 |

## 🔄 完整执行流程

```
1. 初始化
   ├─ 创建 ALFWorldEnv
   ├─ 创建 ReActAgent
   └─ Agent 构建 system prompt（包含 2-shot examples）

2. 任务循环（run_task）
   ├─ env.reset()
   │  └─ 返回 (observation, task_description)
   │
   ├─ agent.reset(task_description)
   │  └─ 清空 trajectory
   │
   └─ ReAct 循环（最多 50 步）
      ├─ agent.step(obs)
      │  ├─ build_messages(obs)
      │  │  ├─ system: [instruction + 2-shot examples]
      │  │  └─ user: [task + history + current_obs + instruction]
      │  │
      │  ├─ call_llm(messages)
      │  │  └─ 返回 "Thought: ...\n Action: ..."
      │  │
      │  ├─ parse_thought_or_reflection(response)
      │  │  └─ 提取 thought 和 action
      │  │
      │  └─ 更新 trajectory
      │
      ├─ env.step(action)
      │  └─ 返回 (observation, reward, done, info)
      │
      └─ 检查是否完成或超时
```

## 📝 Prompt 结构示例

### System Message（第一次调用时设置）

```
Interact with a household to solve a task...

For each of your turn, you will be given the observation...
Your output must strictly follow this format: "Thought: your thoughts.\n Action: your next action"

The available actions are:
1. go to recep
2. take obj from recep
...

Here is an example:
Example 1:
[完整的 2-shot 示例 1]
Example 2:
[完整的 2-shot 示例 2]
```

### User Message（每次调用时构建）

```
Your task is: put some spraybottle on toilet

Observation: You are in the middle of a room...
Thought: To solve the task, I need to...
Action: go to cabinet 1

Observation: On the cabinet 1, you see...
Action: go to cabinet 2

...

Observation: [当前观察]
For each of your turn, you will be given the observation...
```

## 🎯 关键设计特点

### 1. **模块化设计**
- Agent、Prompt、Environment 完全分离
- 易于扩展新的 Agent 类型

### 2. **标准 OpenAI API 格式**
- 使用标准的 `messages` 格式
- System/User 角色清晰分离

### 3. **自动历史管理**
- `trajectory` 自动累积所有步骤
- 每次调用 LLM 时自动包含完整历史

### 4. **灵活的配置**
- 支持环境变量和 `.env` 文件
- 支持代理 API（国内可用）

### 5. **Mock 模式支持**
- 可以在没有 ALFWorld 的情况下测试
- 便于开发和调试

## 🔍 关键代码位置

| 功能 | 文件 | 关键方法/变量 |
|------|------|--------------|
| **ReAct Agent** | `agents/react_agent.py` | `get_system_prompt()`, `get_instruction()` |
| **Few-shot 示例** | `prompts/react.py` | `REACT_EXAMPLE_1`, `REACT_EXAMPLE_2` |
| **基础指令** | `prompts/base.py` | `SYSTEM_INSTRUCTION`, `AVAILABLE_ACTIONS` |
| **消息构建** | `agents/base_agent.py` | `build_messages()` |
| **执行循环** | `agents/base_agent.py` | `run_task()`, `step()` |
| **LLM 调用** | `utils/llm.py` | `call_llm()`, `parse_thought_or_reflection()` |
| **环境包装** | `environment/alfworld_env.py` | `reset()`, `step()` |
| **主实验** | `run_experiment.py` | `run_agent_experiments()` |

## 🚀 运行方式

```bash
# 运行 ReAct Agent
python run_experiment.py --agent react --num_tasks 10

# 运行所有方法对比
python run_experiment.py --agent all --num_tasks 134
```

## 📈 预期结果

根据 ReflAct 论文（GPT-4o-mini）：
- **ReAct**: 53.0% 成功率
- **ReflAct**: 66.4% 成功率

## 🔑 关键洞察

1. **固定的 2-shot 示例**：与 MPO 不同，不按任务类型选择，而是使用论文中的两个固定示例
2. **System Prompt 使用**：明确使用 system role，使指令和示例与对话历史分离
3. **自动历史累积**：每次调用都会包含完整历史，无需手动管理
4. **清晰的模块化**：代码结构清晰，易于理解和扩展

