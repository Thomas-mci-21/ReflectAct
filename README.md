# ReflAct: World-Grounded Decision Making in LLM Agents

This repository contains a reproduction of the **ReflAct** paper experiments on ALFWorld.

📄 **Paper**: [ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection](https://arxiv.org/abs/2505.15182)

## Overview

ReflAct is a novel reasoning backbone for LLM agents that shifts reasoning from merely planning next actions to **continuously reflecting on the agent's state relative to its goal**. This addresses common issues in ReAct-style agents:

1. **Lack of grounding in internal state** - agents fail to maintain coherent internal state
2. **Short-sighted planning** - decisions appear locally plausible but disregard long-term goals

### Key Difference: Thought vs Reflection

| ReAct (Thought) | ReflAct (Reflection) |
|-----------------|---------------------|
| "Now I find a spraybottle 2. Next, I need to take it." | "Currently, I am at cabinet 2 and have found a spraybottle 2, which brings me closer to completing the task of placing it on the toilet." |

## Methods Implemented

| Method | Description | Paper Reference |
|--------|-------------|-----------------|
| **NoThinking** | Direct action without reasoning | Section 6.2 |
| **ReAct** | Think about next action, then act | Yao et al., 2023 |
| **Plan-and-Act** | Plan at first step, then execute | Inspired by Wang et al., 2023 |
| **ReflAct** | Reflect on state in relation to goal | Section 4 |

## Expected Results (GPT-4o-mini on ALFWorld)

| Method | Success Rate |
|--------|-------------|
| NoThinking | 43.3% |
| ReAct | 53.0% |
| Plan-and-Act | 59.0% |
| ReflAct | **66.4%** |

## 🚀 Quick Start (快速开始)

如果你已经设置过环境，下次想要运行实验时：

```bash
# 1. 进入项目目录
cd ~/ReflectAct

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 检查API密钥是否设置
cat .env

# 4. 运行实验
python run_experiment.py --agent reflact --num_tasks 5
```

## Installation (首次安装)

### 1. Clone the repository

```bash
git clone https://github.com/Thomas-mci-21/ReflectAct.git
cd ReflectAct
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac/WSL
# or
.\venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
# Install core dependencies (works without ALFWorld)
pip install openai pyyaml python-dotenv tqdm pandas

# Optional: Install ALFWorld for real environment (may have build issues)
# pip install alfworld
```

### 4. Set up environment variables

Create a `.env` file:

```bash
# Your OpenAI API key
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# Model to use
OPENAI_MODEL=gpt-4o-mini

# API base URL - for China relay/proxy (国内中转)
OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
# Alternative: OPENAI_BASE_URL=https://api.chatanywhere.org/v1
# For official OpenAI API: OPENAI_BASE_URL=https://api.openai.com/v1
```

### 5. Test installation

```bash
# Test the code works
python test_mock.py

# Should see: "All tests passed!"
```

**China Users (国内用户)**: The default `base_url` is set to `https://api.chatanywhere.tech/v1` for relay access.

## Usage

### 🎯 Recommended Experiments

```bash
# 1. Test single agent (5 tasks, ~2-3 minutes)
python run_experiment.py --agent reflact --num_tasks 5

# 2. Compare all methods (10 tasks each, ~10-15 minutes)
python run_experiment.py --agent all --num_tasks 10

# 3. Reproduce Table 2 results (134 tasks each, ~2-3 hours)
python run_experiment.py --agent all --num_tasks 134
```

### 📊 Individual Agent Tests

```bash
# Test ReflAct (our main method)
python run_experiment.py --agent reflact --num_tasks 10

# Test ReAct baseline
python run_experiment.py --agent react --num_tasks 10

# Test NoThinking baseline
python run_experiment.py --agent nothinking --num_tasks 10

# Test Plan-and-Act baseline
python run_experiment.py --agent plan_and_act --num_tasks 10
```

### 🔧 Debug and Test

```bash
# Test code without API calls
python test_mock.py

# Run quietly (less output)
python run_experiment.py --agent reflact --num_tasks 5 --quiet
```

## Project Structure

```
reflectact/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── config.py                    # Configuration
│
├── prompts/                     # Prompt templates (from paper)
│   ├── base.py                  # Base prompts for ALFWorld
│   ├── nothinking.py            # NoThinking prompts
│   ├── react.py                 # ReAct prompts
│   ├── plan_and_act.py          # Plan-and-Act prompts
│   └── reflact.py               # ReflAct prompts
│
├── agents/                      # Agent implementations
│   ├── base_agent.py            # Base agent class
│   ├── nothinking_agent.py      # NoThinking agent
│   ├── react_agent.py           # ReAct agent
│   ├── plan_and_act_agent.py    # Plan-and-Act agent
│   └── reflact_agent.py         # ReflAct agent
│
├── environment/                 # Environment wrapper
│   └── alfworld_env.py          # ALFWorld wrapper
│
├── utils/                       # Utilities
│   ├── llm.py                   # LLM API calls
│   └── logger.py                # Logging and results
│
├── run_experiment.py            # Main experiment script
├── test_mock.py                 # Test script (mock mode)
│
└── results/                     # Results saved here
```

## Terminal Output

The script provides real-time colored output:

```
######################################################################
Task 0: reflact Agent
######################################################################
Goal: put some spraybottle on toilet

============================================================
Step 1
============================================================
Observation: You are in the middle of a room...
Reflection: I am currently in the middle of the room. My goal is to put a spraybottle on the toilet...
Action: go to cabinet 1

============================================================
Step 2
============================================================
Observation: On the cabinet 1, you see a cloth 1...
Reflection: I am at cabinet 1 and can see its contents...
Action: go to cabinet 2
...

Result: SUCCESS in 7 steps
======================================================================
```

## Results Files

Results are saved in JSON format:

- `results/{agent_type}_task_{id}.json` - Individual task trajectories
- `results/{agent_type}_summary.json` - Summary statistics

### 📈 Expected Results (GPT-4o-mini)

Based on paper Table 2, you should see approximately:

| Method | Success Rate | Our Target |
|--------|-------------|------------|
| NoThinking | 43.3% | ~40-45% |
| ReAct | 53.0% | ~50-55% |
| Plan-and-Act | 59.0% | ~55-60% |
| ReflAct | **66.4%** | **~65-70%** |

## 🔄 Daily Workflow

```bash
# Every time you want to run experiments:
cd ~/ReflectAct
source venv/bin/activate
python run_experiment.py --agent reflact --num_tasks 5

# When done:
deactivate
```

## Extending the Code

### Adding a new agent

1. Create `agents/new_agent.py`:

```python
from agents.base_agent import BaseAgent

class NewAgent(BaseAgent):
    agent_type = "new_agent"
    
    def get_system_prompt(self) -> str:
        return "..."
    
    def get_instruction(self, step: int) -> str:
        return "..."
```

2. Add to `agents/__init__.py`
3. Add to `AGENT_CLASSES` in `run_experiment.py`

### Adding a new environment

Create a wrapper in `environment/` that provides:
- `reset()` → (observation, task_description)
- `step(action)` → (observation, reward, done, info)

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{kim2025reflact,
  title={ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection},
  author={Kim, Jeonghye and Rhee, Sojeong and Kim, Minbeom and Kim, Dohyung and Lee, Sangmook and Sung, Youngchul and Jung, Kyomin},
  journal={arXiv preprint arXiv:2505.15182},
  year={2025}
}
```

## License

MIT License

