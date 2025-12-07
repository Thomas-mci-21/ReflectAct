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

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Thomas-mci-21/ReflectAct.git
cd ReflectAct
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ALFWorld (in WSL/Linux)

```bash
pip install alfworld
export ALFWORLD_DATA=/path/to/alfworld/data
```

### 5. Set up environment variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

## Usage

### Run a single agent

```bash
# Run ReflAct on 10 tasks
python run_experiment.py --agent reflact --num_tasks 10

# Run ReAct on all 134 tasks
python run_experiment.py --agent react --num_tasks 134

# Run NoThinking quietly
python run_experiment.py --agent nothinking --num_tasks 20 --quiet
```

### Run all agents (comparison)

```bash
python run_experiment.py --agent all --num_tasks 134
```

### Test without ALFWorld (mock mode)

```bash
python test_mock.py
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

