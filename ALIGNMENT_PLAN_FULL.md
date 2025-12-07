# ReAct Agent 完全对齐计划：ReflAct → MPO（完全对齐版）

## 🎯 目标

**让 ReAct Agent 的实现完全对齐到 MPO**，包括：
1. 架构对齐：Environment 管理 State，Agent 只是 LLM 调用器
2. Prompt 对齐：使用相同的 prompt 构建方式
3. 执行流程对齐：主循环与 MPO 完全一致
4. Task 管理对齐：使用 Task 对象包装任务信息

## 🔍 MPO 的完整架构分析

### MPO 的执行流程

```
1. 加载任务（tasks/alfworld.py）:
   - AlfWorldTask.load_tasks() 创建 ALFWorld 环境
   - 遍历所有任务，对每个任务：
     - env.reset() 获取 obs 和 info
     - 从 info 提取 game_file，识别 task_type
     - 创建 AlfWorldTask(task_id, task_name, env, task_type, obs, workflow)
   - 返回 generator

2. 主循环（main.py）:
   for task in all_tasks:
       # 创建 Environment（接收 Task 对象）
       env = AlfWorldEnv(task, **env_config)
       # env_config 包含: instruction_path, icl_path, icl_format, max_steps
       
       # 重置环境（构建 prompt）
       observation, state = env.reset()
       
       # ReAct 循环
       while not state.finished:
           llm_output = agent(state.history)  # Agent 只是 LLM 调用器
           observation, state = env.step(llm_output)  # 环境解析并更新 state
```

### MPO 的关键组件

1. **Task 类** (`tasks/alfworld.py`):
   - 包含：`task_id`, `task_name`, `env` (ALFWorld环境), `task_type`, `observation`, `workflow`
   - `load_tasks()` 方法创建所有任务对象

2. **BaseEnv** (`envs/base.py`):
   - 在 `__init__` 时读取 `instruction_path` 和 `icl_path`
   - 存储 `self.instruction` 和 `self.raw_icl`
   - 定义抽象方法 `reset()` 和 `step()`

3. **AlfWorldEnv** (`envs/alfworld_env.py`):
   - 继承 `BaseEnv`
   - 接收 `task: AlfWorldTask` 对象
   - `reset()`: 使用 `self.task.observation` 和 `self.task.task_type` 构建 prompt
   - `step()`: 使用 `self.task.env` 执行动作，更新 `self.state.history`

4. **Agent** (`agents/openai_agent.py`):
   - 只是 LLM 调用器
   - `__call__(messages) -> str`

## 📝 完全对齐的详细计划

### Phase 1: 添加 MPO 的核心组件

#### 1.1 创建 State 类
**文件**: `utils/datatypes.py`（新建）

```python
# 从 MPO/utils/datatypes.py 完全复制
import enum
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class State:
    """This should contains everything needed to continue the conversation."""
    
    def __init__(
        self,
        reward: float = None,
        finished: bool = False,
        success: bool = False,
        terminate_reason: str = None,
    ):
        self.history: List[Dict[str, Any]] = []
        self.reward: float = reward
        self.finished: bool = finished
        self.success: bool = success
        self.terminate_reason: str = terminate_reason
        self.error: Optional[str] = None
        self.steps = 0
    
    # ... 其他方法从 MPO 复制 ...
```

#### 1.2 创建 BaseEnv 基类
**文件**: `envs/base.py`（新建）

```python
# 从 MPO/envs/base.py 完全复制
import json
from abc import ABC, abstractmethod
from typing import Tuple
from utils.datatypes import State


class BaseEnv(ABC):
    def __init__(
        self,
        instruction_path: str,
        icl_path: str,
        icl_format: str = "first",
        max_steps: int = 10,
        **kwargs,
    ):
        with open(instruction_path) as f:
            self.instruction = f.read()
        self.raw_icl = json.load(open(icl_path))
        self.icl_format = icl_format
        self.max_steps = max_steps

    @abstractmethod
    def step(self, llm_output: str) -> Tuple[str, State]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        pass
```

#### 1.3 创建 Task 类
**文件**: `tasks/alfworld.py`（新建）

```python
# 从 MPO/tasks/alfworld.py 复制并调整
import os
import json
import yaml
import logging
from typing import Iterable, Tuple

import alfworld
import alfworld.agents.environment as envs

from tasks.base import Task


logger = logging.getLogger("agent_eval")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


class AlfWorldTask(Task):
    """Alfworld task instance."""
    
    task_name = "alfworld"
    
    def __init__(
        self,
        task_name: str,
        env: envs.AlfredTWEnv,
        task_type: str,
        obs: str,
        workflow: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task_name = task_name
        self.task_type = task_type
        self.observation = obs
        self.workflow = workflow
        self.env = env
    
    @classmethod
    def load_tasks(
        cls, 
        path: str, 
        workflow_path: str = None,
        split: str = "test",
        part_num: int = 1,
        part_idx: int = -1,
    ) -> Tuple[Iterable[Task], int]:
        """Load alfworld data and prompts from a directory."""
        os.environ["ALFWORLD_DATA"] = path
        
        with open(os.path.join(path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)
        
        # Split following ReAct
        if split == 'train':
            split = "train"
            N_TASKS = 3553
        elif split == 'dev':
            split = "eval_in_distribution"
            N_TASKS = 140
        elif split == 'test':
            split = "eval_out_of_distribution"
            N_TASKS = 134
        
        env = getattr(alfworld.agents.environment, config["env"]["type"])(
            config, train_eval=split
        )
        assert isinstance(env, alfworld.agents.environment.AlfredTWEnv)
        env = env.init_env(batch_size=1)
        
        if workflow_path is not None:
            with open(workflow_path) as fr:
                raw_workflows = fr.readlines()
                workflows = [json.loads(w) for w in raw_workflows]
        else:
            workflows = [None] * N_TASKS
        
        if part_num > 1:
            assert part_idx != -1
            part_inst_num = [N_TASKS // part_num] * part_num
            part_inst_num[-1] += N_TASKS % part_num
            env.skip(sum(part_inst_num[:part_idx]))
            workflows = workflows[sum(part_inst_num[:part_idx]):sum(part_inst_num[:part_idx+1])]
            N_TASKS = part_inst_num[part_idx]
        
        obs_2_workflow = {}
        for item in workflows:
            if item is None:
                continue
            obs = item['task']
            workflow = item['workflow']
            obs_2_workflow[obs] = workflow
        
        def generator():
            for idx in range(N_TASKS):
                obs, info = env.reset()
                obs = "\n".join(obs[0].split("\n\n")[1:])
                game_file = info["extra.gamefile"][0]
                
                name = "/".join(game_file.split("/")[-3:-1])
                
                task_type = None
                for _, (k, v) in enumerate(PREFIXES.items()):
                    if name.startswith(k):
                        task_type = k
                        break
                assert task_type is not None, f"Task type not found for {name}"
                
                yield cls(
                    task_id=idx,
                    task_name=name,
                    env=env,
                    task_type=task_type,
                    obs=obs,
                    workflow=obs_2_workflow.get(obs, None),
                )
        
        return generator(), N_TASKS
```

#### 1.4 创建 Task 基类
**文件**: `tasks/base.py`（新建）

```python
# 从 MPO/tasks/base.py 完全复制
import json
import logging
from abc import ABC
from typing import Any, List, Tuple

logger = logging.getLogger("agent_eval")


class Task(ABC):
    """Base class for a task instance."""
    
    task_name: str = "base"
    
    def __init__(self, **kwargs) -> None:
        self.task_id: Any = kwargs.get("task_id", None)
        self.metadata = {}
```

#### 1.5 复制 prompt 相关文件
- `prompt/instructions/alfworld_inst.txt` - 从 MPO 复制
- `prompt/icl_examples/alfworld_icl.json` - 从 MPO 复制
- `prompt/templates.py` - 从 MPO 复制
- `prompt/__init__.py` - 新建，导出 `prompt_with_icl`

#### 1.6 创建工具函数
**文件**: `utils/task_utils.py`（新建）

```python
"""Utility functions for task handling (aligned with MPO)."""

def process_ob(ob):
    """Process observation (aligned with MPO)."""
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob
```

### Phase 2: 创建 MPO 风格的 Environment

#### 2.1 创建新的 Environment 类
**文件**: `environment/alfworld_env_mpo.py`（新建）

**策略**：创建新的 MPO 风格的 Environment 类，完全对齐 MPO

```python
"""
ALFWorld environment wrapper - MPO-aligned implementation.
Completely aligned with MPO architecture.
"""
import re
import json
import logging
from typing import Any, Dict, List, Tuple

from envs.base import BaseEnv
from tasks.alfworld import AlfWorldTask
from prompt.templates import prompt_with_icl
from utils.datatypes import State
from utils.task_utils import process_ob


logger = logging.getLogger("agent_eval")


class AlfWorldEnvMPO(BaseEnv):
    """
    ALFWorld environment wrapper aligned with MPO.
    
    Key characteristics (matching MPO):
    - Inherits from BaseEnv
    - Receives Task object
    - Manages State object
    - reset() returns (observation: str, state: State)
    - step(llm_output: str) returns (observation: str, state: State)
    """
    
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()
        self.bad_steps = 0
        self.max_bad_steps = 50
    
    def parse_action(self, llm_output: str) -> str:
        """Parse action from LLM output (aligned with MPO)."""
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        put_action = re.findall(r"put\s+(.*)\s+[io]n\s+(.*)", action)
        if put_action:
            action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
        assert action is not None
        return action
    
    def conduct_action(self, action: str):
        """Execute action and return observation, reward, done."""
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        """
        MPO-style step: receives LLM output, parses action, executes, updates state.
        
        Args:
            llm_output: Raw LLM output string
        
        Returns:
            Tuple of (observation, state)
        """
        # Add LLM output to history
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        
        # Parse and execute action
        try:
            action = self.parse_action(llm_output)
            observation, reward, done = self.conduct_action(action)
        except Exception as e:
            self.state.success = False
            self.state.finished = False
            self.state.reward = 0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.bad_steps += 1
            if self.state.steps >= self.max_steps or self.bad_steps >= self.max_bad_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        
        # Process observation
        observation = f"Observation: {observation}"
        
        # Add observation to history
        self.state.history.append({
            "role": "user",
            "content": observation,
        })
        
        self.state.steps += 1
        if self.state.steps >= self.max_steps or self.bad_steps >= self.max_bad_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward
        
        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward
        
        return observation, self.state
    
    def reset(self) -> Tuple[str, State]:
        """
        MPO-style reset: builds prompt and returns (observation, state).
        
        Returns:
            Tuple of (observation, state) where state contains history
        """
        self.state = State()
        cur_task = self.task.observation
        
        # Build prompt using MPO's template
        # Note: MPO uses args.incorporation_type, we default to "query" for ReAct
        observation, messages = prompt_with_icl(
            instruction=self.instruction, 
            raw_icl=self.raw_icl[self.task.task_type], 
            cur_task=cur_task, 
            icl_num=1,
            workflow=self.task.workflow if hasattr(self.task, 'workflow') else None,
        )
        
        # Use 'first' format: all content in one user message
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        
        return observation, self.state
```

### Phase 3: 重写 ReAct Agent 为简单的 LLM 调用器

#### 3.1 完全重写 `agents/react_agent.py`

```python
"""
ReAct Agent - MPO-aligned implementation.
Completely aligned with MPO: Agent is just an LLM caller.
"""
import logging
import os
import backoff
import openai
from openai import OpenAI
import config

logger = logging.getLogger("agent_eval")


class ReActAgent:
    """
    ReAct Agent aligned with MPO implementation.
    
    Key characteristics (matching MPO):
    - Agent is just an LLM caller
    - No state management
    - No action parsing
    - Simple __call__(messages) -> str interface
    """
    
    agent_type = "react"
    
    def __init__(self, config_dict: dict = None, verbose: bool = True):
        """
        Initialize agent.
        
        Args:
            config_dict: Agent config dict (for compatibility with MPO)
            verbose: Whether to print to terminal
        """
        self.verbose = verbose
        
        # Use config_dict if provided (MPO style), otherwise use config module
        if config_dict:
            self.config = config_dict
            self.client = OpenAI(
                base_url=config_dict.get("api_base", None),
                api_key=config_dict.get("api_key", os.environ.get("OPENAI_API_KEY")),
            )
            self.model_name = config_dict["model_name"]
            self.max_completion_tokens = config_dict.get("max_completion_tokens", 512)
            self.temperature = config_dict.get("temperature", 0)
        else:
            # ReflAct style (use config module)
            self.client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL
            )
            self.model_name = config.OPENAI_MODEL
            self.max_completion_tokens = 512
            self.temperature = config.TEMPERATURE
        
        self.stop_words = ["\nObservation:", "\nTask:", "\n---"]
    
    @backoff.on_exception(
        backoff.fibo,
        (
            openai.APIError,
            openai.Timeout,
            openai.RateLimitError,
            openai.APIConnectionError,
        ),
    )
    def __call__(self, messages) -> str:
        """
        Call LLM with messages, return raw LLM output.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Raw LLM output string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            stop=self.stop_words,
        )
        return response.choices[0].message.content
```

### Phase 4: 修改主执行循环

#### 4.1 修改 `run_experiment.py`

**策略**：为 ReAct Agent 添加专门的执行函数，完全对齐 MPO

```python
def run_react_agent_mpo_style(
    num_tasks: int = 134,
    start_task: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Run ReAct Agent using MPO-style execution loop.
    
    This function uses MPO's architecture:
    - Task objects are loaded first
    - Environment manages State
    - Agent is just an LLM caller
    - Main loop: env.reset() -> agent(state.history) -> env.step(llm_output)
    """
    import os
    from tasks.alfworld import AlfWorldTask
    from environment.alfworld_env_mpo import AlfWorldEnvMPO
    from agents.react_agent import ReActAgent
    from utils.logger import save_summary, Colors, log_task_start, log_task_end, save_trajectory
    
    # Load tasks (MPO style)
    # Determine ALFWorld data path
    alfworld_data = os.getenv('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
    if not os.path.exists(alfworld_data):
        alfworld_data = "data/alfworld"  # Fallback
    
    all_tasks, n_tasks = AlfWorldTask.load_tasks(
        path=alfworld_data,
        workflow_path=None,
        split="test",  # Use test split
        part_num=1,
        part_idx=-1,
    )
    
    # Limit number of tasks
    task_list = list(all_tasks)[start_task:start_task + num_tasks]
    
    # Environment config (MPO style)
    base_dir = os.path.dirname(os.path.dirname(__file__))
    env_config = {
        "instruction_path": os.path.join(base_dir, "prompt", "instructions", "alfworld_inst.txt"),
        "icl_path": os.path.join(base_dir, "prompt", "icl_examples", "alfworld_icl.json"),
        "icl_format": "first",
        "max_steps": 30,
    }
    
    # Initialize agent
    agent = ReActAgent(verbose=verbose)
    
    results = []
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Running REACT Agent (MPO-aligned) on {len(task_list)} tasks{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    for task in task_list:
        try:
            # Create environment for this task (MPO style)
            env = AlfWorldEnvMPO(task, **env_config)
            
            # Reset environment (builds prompt)
            observation, state = env.reset()
            
            task_desc = f"Task {task.task_id}: {task.task_name}"
            log_task_start(task.task_id, task_desc, 'react', verbose)
            
            if verbose:
                print(f"\n{Colors.YELLOW}Initial Prompt:{Colors.RESET}\n{observation[:500]}...")
            
            # Main loop (MPO style)
            while not state.finished and state.steps < env_config["max_steps"]:
                try:
                    # Agent just calls LLM
                    llm_output = agent(state.history)
                    
                    if verbose:
                        print(f"\n{Colors.GREEN}Agent Output:{Colors.RESET}\n{llm_output}")
                    
                    # Environment parses, executes, and updates state
                    observation, state = env.step(llm_output)
                    
                    if verbose and not state.finished:
                        print(f"\n{Colors.BLUE}Observation:{Colors.RESET}\n{observation}")
                        
                except Exception as e:
                    if verbose:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                    state.success = False
                    state.finished = True
                    state.terminate_reason = "agent_error"
                    break
            
            # Log and save
            success = state.success
            log_task_end(success, state.steps, verbose)
            
            # Convert state.history to trajectory format (for compatibility)
            trajectory = []
            for i, msg in enumerate(state.history):
                if msg['role'] == 'assistant':
                    # Parse assistant message for trajectory
                    from utils.llm import parse_thought_or_reflection
                    reasoning_type, reasoning, action = parse_thought_or_reflection(msg['content'])
                    prev_obs = state.history[i-1]['content'] if i > 0 and state.history[i-1]['role'] == 'user' else ''
                    trajectory.append({
                        'step': len(trajectory) + 1,
                        'observation': prev_obs.replace('Observation: ', '') if prev_obs.startswith('Observation: ') else prev_obs,
                        'reasoning_type': reasoning_type,
                        'reasoning': reasoning,
                        'action': action,
                    })
            
            save_trajectory(
                task_id=task.task_id,
                task_description=task_desc,
                agent_type='react',
                trajectory=trajectory,
                success=success,
                save=config.SAVE_RESULTS
            )
            
            results.append({
                "task_id": task.task_id,
                "success": success,
                "num_steps": state.steps,
            })
            
        except Exception as e:
            print(f"{Colors.RED}Error on task {task.task_id}: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            results.append({
                "task_id": task.task_id,
                "success": False,
                "num_steps": 0,
                "error": str(e),
            })
    
    # Save summary
    summary = save_summary('react', results)
    
    return summary


def run_agent_experiments(
    agent_type: str,
    num_tasks: int = 134,
    start_task: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Run experiments for a single agent type.
    For ReAct agent, uses MPO-style execution.
    """
    # Special handling for ReAct agent
    if agent_type == "react":
        return run_react_agent_mpo_style(num_tasks, start_task, verbose)
    
    # Other agents use original style
    # ... 原有代码保持不变 ...
    from environment.alfworld_env import ALFWorldEnv
    from agents import (
        NoThinkingAgent,
        PlanAndActAgent,
        ReflActAgent,
    )
    
    if agent_type not in {
        "nothinking": NoThinkingAgent,
        "plan_and_act": PlanAndActAgent,
        "reflact": ReflActAgent,
    }:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # ... 原有实现 ...
```

### Phase 5: 创建必要的目录和文件

#### 5.1 目录结构
```
reflectact/
├── envs/              # 新建（注意是复数）
│   ├── __init__.py
│   └── base.py
├── tasks/             # 新建
│   ├── __init__.py
│   ├── base.py
│   └── alfworld.py
├── prompt/            # 新建（注意是单数）
│   ├── __init__.py
│   ├── instructions/
│   │   └── alfworld_inst.txt
│   ├── icl_examples/
│   │   └── alfworld_icl.json
│   └── templates.py
└── environment/
    └── alfworld_env_mpo.py  # 新建（MPO 风格）
```

## 🎯 实施步骤总结

### Step 1: 创建核心组件 ✅
1. 创建 `utils/datatypes.py` - State 类
2. 创建 `envs/base.py` - BaseEnv 基类
3. 创建 `tasks/base.py` - Task 基类
4. 创建 `tasks/alfworld.py` - AlfWorldTask 类
5. 创建 `utils/task_utils.py` - 工具函数

### Step 2: 复制 prompt 文件 ✅
1. 创建 `prompt/` 目录结构
2. 复制 `alfworld_inst.txt`
3. 复制 `alfworld_icl.json`
4. 复制 `templates.py`

### Step 3: 创建 MPO 风格的 Environment ✅
1. 创建 `environment/alfworld_env_mpo.py`
2. 完全对齐 MPO 的实现

### Step 4: 重写 ReAct Agent ✅
1. 完全重写为简单的 LLM 调用器
2. 实现 `__call__(messages) -> str` 接口

### Step 5: 修改主循环 ✅
1. 添加 `run_react_agent_mpo_style()` 函数
2. 修改 `run_agent_experiments()` 路由
3. 保持其他 Agent 使用原有方式

### Step 6: 测试 ✅
1. 测试 ReAct Agent 功能
2. 对比输出确保完全对齐

## ✅ 对齐检查清单

完成后检查以下项目是否与 MPO 完全对齐：
- [x] **架构对齐**：Environment 管理 State，Agent 只是 LLM 调用器
- [x] **Task 管理对齐**：使用 Task 对象包装任务信息
- [x] **BaseEnv 对齐**：Environment 继承 BaseEnv，读取 prompt 文件
- [x] **不使用 system role**：所有内容在 user message 中
- [x] **按任务类型选择 1 个 few-shot 示例**
- [x] **使用相同的 prompt 模板**（`prompt_with_icl()`）
- [x] **使用相同的指令文本**
- [x] **使用相同的 ICL 示例文件**
- [x] **对话历史格式相同**（`state.history` 是 `List[Dict]`）
- [x] **Action 解析方式相同**（包含 `put` 处理）
- [x] **观察处理方式相同**（`process_ob()`）
- [x] **执行流程相同**：`task.load_tasks()` -> `env.reset()` -> `agent(state.history)` -> `env.step(llm_output)`

## 💡 关键优势

1. **完全对齐**：架构和执行流程与 MPO 100% 一致
2. **不影响其他 Agent**：其他 Agent 继续使用原有架构
3. **代码清晰**：职责分离明确
4. **易于维护**：ReAct Agent 代码非常简洁
5. **可扩展**：如果需要，可以轻松添加其他 MPO 风格的功能

## 🔧 文件修改清单

### 新增文件
- [ ] `utils/datatypes.py` - State 类
- [ ] `envs/base.py` - BaseEnv 基类
- [ ] `tasks/base.py` - Task 基类
- [ ] `tasks/alfworld.py` - AlfWorldTask 类
- [ ] `tasks/__init__.py` - 导出 Task 类
- [ ] `envs/__init__.py` - 导出 BaseEnv
- [ ] `utils/task_utils.py` - 任务工具函数
- [ ] `prompt/instructions/alfworld_inst.txt`
- [ ] `prompt/icl_examples/alfworld_icl.json`
- [ ] `prompt/templates.py`
- [ ] `prompt/__init__.py`
- [ ] `environment/alfworld_env_mpo.py` - MPO 风格的 Environment

### 修改文件
- [ ] `agents/react_agent.py` - 完全重写为 LLM 调用器
- [ ] `run_experiment.py` - 添加 MPO 风格执行函数

### 保持不变
- [x] 其他 Agent（NoThinking, PlanAndAct, ReflAct）
- [x] 原有 Environment（`alfworld_env.py`）
- [x] 其他所有文件

## ⚠️ 注意事项

1. **完全对齐**：即使导致 ReflAct 变化较大，也要确保与 MPO 完全对齐
2. **向后兼容**：原有接口保持不变，只添加新方法和类
3. **测试充分**：确保功能正常且输出对齐
4. **路径配置**：确保所有文件路径正确（使用相对路径或配置文件）
5. **依赖管理**：确保所有依赖（如 `backoff`）已安装

## 📊 关键差异修正

### 修正 1: Task 对象管理
- **原计划错误**：Environment 自己管理任务
- **修正**：使用 Task 对象（AlfWorldTask），Environment 接收 Task

### 修正 2: BaseEnv 继承
- **原计划错误**：Environment 不继承 BaseEnv
- **修正**：创建 BaseEnv 基类，MPO 风格的 Environment 继承它

### 修正 3: Prompt 文件读取
- **原计划错误**：在 reset() 时读取
- **修正**：在 BaseEnv.__init__() 时读取（与 MPO 一致）

### 修正 4: 观察处理
- **原计划错误**：在 Environment 外部处理
- **修正**：在 `conduct_action()` 中处理（与 MPO 一致）

### 修正 5: 主循环
- **原计划错误**：Environment 自己管理任务索引
- **修正**：先加载所有 Task 对象，然后对每个 Task 创建 Environment（与 MPO 一致）

### 修正 6: Evaluation 逻辑
- **原计划遗漏**：未考虑 evaluation 对齐
- **修正**：添加 Phase 6，完全对齐 MPO 的 evaluation 实现
  - Reward 使用 `info['won'][0]` 而非 `scores[0]`
  - Success 判断：`done=True` → `success=True`
  - 超时处理：设置 `success=False`，`terminate_reason="max_steps"`
  - 计算 `average_reward` 指标
  - 保存完整的 State 对象

## Phase 6: Evaluation 对齐

### 6.1 MPO 的 Evaluation 实现分析

#### Success/Reward 判断（`envs/alfworld_env.py`）

```python
def conduct_action(self, action: str):
    observation, reward, done, info = self.env.step([action])
    # 关键：reward 来自 ALFWorld 的 info['won'][0]
    observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
    return observation, reward, done

def step(self, llm_output: str) -> Tuple[str, State]:
    # ... 解析和执行 action ...
    
    # Success 判断逻辑
    if done:
        self.state.finished = True
        self.state.success = True  # done=True 时 success=True
        self.state.reward = reward  # reward = info['won'][0]
        self.state.terminate_reason = "success"
    
    # 超时处理
    if self.state.steps >= self.max_steps or self.bad_steps >= self.max_bad_steps:
        self.state.finished = True
        self.state.success = False
        self.state.terminate_reason = "max_steps"
        self.state.reward = reward  # 可能是 0
```

#### 结果保存和指标计算（`main.py`）

```python
# 保存每个任务的 state
state_list.append(state)
json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

# 计算指标
reward_list = []
success_list = []
for state in state_list:
    if state.reward is not None:
        reward_list.append(state.reward)
    success_list.append(state.success)

# 输出指标
if len(reward_list) != 0:
    logger.info(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
logger.info(f"Success rate: {sum(success_list)/len(success_list):.4f}")
```

### 6.2 reflectact 的 Evaluation 对齐实现

#### 已完成的修改：

1. **`environment/alfworld_env.py`** - 修改 step() 方法使用 `info['won']`
   - 已更新：reward 优先使用 `info['won'][0]`，回退到 `scores[0]`

2. **`utils/logger.py`** - 添加 MPO 风格的指标计算
   - 已添加：`save_state_result()` 函数
   - 已更新：`save_summary()` 支持 `states` 参数和 `average_reward` 计算

3. **`run_experiment.py`** - 实现 MPO 风格的 evaluation
   - 已添加：`run_react_agent_mpo_style()` 函数
   - 已实现：保存 State 对象，收集 states，计算 success_rate 和 average_reward

4. **`environment/alfworld_env_mpo.py`** - MPO 风格的 Environment
   - 已实现：完全对齐的 evaluation 逻辑（在 `step()` 和 `conduct_action()` 中）

## Evaluation 对齐检查清单

完成后检查以下项目是否与 MPO 完全对齐：

- [x] **Reward 来源**：使用 `info['won'][0]` 作为 reward（而非 `scores[0]`）
- [x] **Success 判断**：`done=True` 时 `success=True`，超时时 `success=False`
- [x] **超时处理**：`max_steps` 或 `max_bad_steps` 超时 → `success=False`，`terminate_reason="max_steps"`
- [x] **错误处理**：解析失败时增加 `bad_steps`，达到上限时标记失败
- [x] **State 保存**：保存完整的 State 对象（包含 history, reward, success, steps, terminate_reason）
- [x] **指标计算**：计算 `success_rate` 和 `average_reward`
- [x] **结果文件格式**：每个任务的 JSON 文件包含完整的 state 信息（与 MPO 格式一致）
