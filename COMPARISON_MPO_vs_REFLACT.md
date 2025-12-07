# MPO vs ReflAct 代码库：ReAct Agent 实现对比

## 📋 概述

本文档对比两个代码库在实现 ReAct Agent 时的关键差异：
- **MPO**: Meta Plan Optimization 代码库
- **ReflAct**: ReflAct 论文复现代码库

## 🔑 核心差异对比表

| 特性 | MPO | ReflAct (reflectact) |
|------|-----|---------------------|
| **System Prompt 使用** | ❌ 所有内容在 user message | ✅ 使用 `role: "system"` |
| **Few-shot 示例数量** | 默认 1 个（按任务类型选择） | 固定 2 个（来自论文） |
| **示例选择策略** | 按任务类型动态选择 | 固定使用论文中的 2 个示例 |
| **示例来源** | `alfworld_icl.json`（按任务类型组织，每种 3 个） | 硬编码在 `prompts/react.py` |
| **Prompt 模板** | `prompt/templates.py` 动态构建 | Agent 类中直接构建 |
| **对话历史管理** | `state.history` (List[Dict]) | `agent.trajectory` (List[Dict]) |
| **配置方式** | JSON 配置文件 | `.env` + `config.py` |
| **环境变量** | 命令行参数 + JSON | `.env` 文件 |

## 📝 详细对比

### 1. System Prompt 处理

#### MPO
```python
# envs/alfworld_env.py:120-124
self.state.history.append({
    "role": "user",  # ⚠️ 所有内容都在 user message 中
    "content": observation  # 包含: instruction + examples + task
})
```

#### ReflAct
```python
# agents/base_agent.py:48-50
messages = [
    {"role": "system", "content": self.get_system_prompt()}  # ✅ 使用 system role
]
```

**影响**：
- MPO: 所有内容混在一起，LLM 需要从 user message 中区分指令和对话
- ReflAct: 指令和示例在 system message 中，对话历史在 user message 中，结构更清晰

### 2. Few-shot 示例策略

#### MPO
```python
# envs/alfworld_env.py:107-109
raw_icl = self.raw_icl[self.task.task_type]  # 根据任务类型选择
icl_num=1,  # 默认只选 1 个示例
```

**特点**：
- 6 种任务类型，每种有 3 个示例可选
- 根据当前任务类型动态选择
- 默认使用 1 个示例（可配置）

**文件结构**：
```json
{
    "pick_and_place": [example1, example2, example3],
    "pick_clean_then_place": [example1, example2, example3],
    ...
}
```

#### ReflAct
```python
# prompts/react.py
REACT_EXAMPLE = f"""Example 1:
{REACT_EXAMPLE_1}  # pick_and_place 任务
Example 2:
{REACT_EXAMPLE_2}  # pick_clean_then_place 任务
"""
```

**特点**：
- 固定使用 2 个示例
- 来自 ReflAct 论文的 Figure 15 和 Figure 17
- 不随任务类型变化

**影响**：
- MPO: 更灵活的示例选择，可以针对不同任务类型提供相关示例
- ReflAct: 固定示例，保证实验一致性，但可能对某些任务类型不够相关

### 3. Prompt 构建方式

#### MPO
```python
# prompt/templates.py:32-104
def prompt_with_icl(instruction, raw_icl, cur_task, icl_num=2, workflow=None):
    # 动态构建 prompt
    prompt = PROMPT_WITH_ICL_TEMPLATE.format(
        instruction=instruction, 
        icl_prompt=icl_prompt, 
        examples=examples, 
        task=cur_task
    )
    return prompt, messages
```

**模板**：
```python
PROMPT_WITH_ICL_TEMPLATE = """{instruction}
---
{icl_prompt}

{examples}
---

Now, it's your turn and here is the task.
{task}"""
```

#### ReflAct
```python
# agents/react_agent.py:22-33
def get_system_prompt(self) -> str:
    return f"""{SYSTEM_INSTRUCTION}

{REACT_INSTRUCTION}

{AVAILABLE_ACTIONS}

{REMINDER}

Here is an example:
{REACT_EXAMPLE}"""
```

**特点**：
- 直接在 Agent 类中构建
- 代码更直观，但灵活性较低

### 4. 对话历史管理

#### MPO
```python
# utils/datatypes.py
class State:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []  # 对话历史

# main.py:50-54
state.history.append({
    "role": "assistant",
    "content": llm_output
})
state.history.append({
    "role": "user",
    "content": observation
})
```

#### ReflAct
```python
# agents/base_agent.py:24
self.trajectory = []  # 轨迹历史

# agents/base_agent.py:38-69
def build_messages(self, observation: str) -> List[Dict]:
    # 从 trajectory 重新构建 messages
    for step_data in self.trajectory:
        user_content += f"Observation: {step_data['observation']}\n"
        if step_data.get('reasoning'):
            user_content += f"{reasoning_type}: {reasoning}\n"
        user_content += f"Action: {step_data['action']}\n"
```

**差异**：
- MPO: 直接维护标准格式的对话历史
- ReflAct: 维护结构化轨迹，每次调用时重新构建 messages

### 5. 指令文件位置

#### MPO
- `prompt/instructions/alfworld_inst.txt` - 指令文本文件
- `prompt/icl_examples/alfworld_icl.json` - 示例 JSON 文件

#### ReflAct
- `prompts/base.py` - 基础指令（Python 字符串）
- `prompts/react.py` - ReAct 指令和示例（Python 字符串）

**影响**：
- MPO: 便于修改和版本控制，但需要文件读取
- ReflAct: 代码内嵌，更直接，但修改需要改代码

### 6. Action 解析

两者都使用正则表达式，实现类似：

#### MPO
```python
# envs/alfworld_env.py:34-42
def parse_action(self, llm_output: str) -> str:
    pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
    action = re.findall(pattern, llm_output)[0]
    # 处理 put 动作的特殊格式
    put_action = re.findall(r"put\s+(.*)\s+[io]n\s+(.*)", action)
    if put_action:
        action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
    return action
```

#### ReflAct
```python
# utils/llm.py:43-77
def parse_action(response: str) -> str:
    action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if action_match:
        return action_match.group(1).strip()
    # ... 其他回退逻辑
```

### 7. 环境包装

两者都使用官方 ALFWorld API，但包装方式略有不同：

#### MPO
```python
# envs/alfworld_env.py
class AlfWorldEnv(BaseEnv):
    def __init__(self, task: AlfWorldTask, **kwargs):
        self.task = task
        self.env = task.env  # 任务包含环境引用
```

#### ReflAct
```python
# environment/alfworld_env.py
class ALFWorldEnv:
    def __init__(self, split: str = "eval_out_of_distribution"):
        self.env = get_environment(env_type)(config, train_eval=split)
        self.env = self.env.init_env(batch_size=1)
```

### 8. 配置管理

#### MPO
- JSON 配置文件：`configs/task/alfworld.json`
- 命令行参数：`--exp_config`, `--agent_config`
- 环境变量：通过命令行覆盖

#### ReflAct
- Python 配置：`config.py`
- `.env` 文件：API key, base URL
- 环境变量：通过 `.env` 文件管理

## 🎯 使用场景建议

### 使用 MPO 的场景
- ✅ 需要灵活的示例选择策略
- ✅ 希望针对不同任务类型使用相关示例
- ✅ 需要与 Meta Plan Optimization 方法结合
- ✅ 偏好 JSON 配置文件

### 使用 ReflAct 的场景
- ✅ 需要严格复现论文结果
- ✅ 偏好标准 OpenAI API 格式（system/user roles）
- ✅ 代码结构更模块化，易于扩展
- ✅ 希望使用 `.env` 文件管理配置

## 📊 性能对比（基于论文）

| 方法 | MPO (Llama-3.1-8B + MPO) | ReflAct (GPT-4o-mini + ReAct) |
|------|---------------------------|-------------------------------|
| **ReAct Baseline** | ~53.6% | 53.0% |
| **优化后** | 83.1% (70B model + MPO) | 66.4% (ReflAct) |

**注意**：这些结果使用不同的模型和优化方法，不能直接对比。

## 🔍 代码质量对比

| 方面 | MPO | ReflAct |
|------|-----|---------|
| **代码组织** | 功能导向，文件较多 | 模块化，清晰的分层 |
| **可扩展性** | 中等（需要理解整体结构） | 高（基类设计良好） |
| **文档** | 较少 | 详细的 README |
| **测试** | 需要手动运行实验 | 包含 mock 模式测试 |

## 💡 关键洞察

1. **Prompt 设计理念不同**：
   - MPO: 将一切放在 user message 中，依赖 LLM 理解上下文
   - ReflAct: 明确分离 system prompt 和对话历史

2. **示例策略不同**：
   - MPO: 数据驱动，灵活但复杂
   - ReflAct: 固定示例，简单但可能不够相关

3. **代码风格**：
   - MPO: 更注重实验配置和灵活性
   - ReflAct: 更注重代码清晰度和可维护性

## 🔗 相关文件

### MPO
- `MPO/REACT_ALFWORLD_DETAILS.md` - MPO 实现详情
- `MPO/REACT_QUICK_SUMMARY.md` - MPO 快速总结

### ReflAct
- `reflectact/REACT_IMPLEMENTATION_ANALYSIS.md` - ReflAct 实现分析
- `reflectact/README.md` - 项目文档

