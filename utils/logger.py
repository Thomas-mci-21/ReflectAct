"""
Logging utilities for ReflAct experiments.
"""
import os
import json
import glob
from datetime import datetime
from typing import Optional, List
import config


class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'


def setup_logger():
    """Setup results directory."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)


def log_step(
    step: int,
    observation: str,
    reasoning_type: Optional[str] = None,
    reasoning: Optional[str] = None,
    action: str = "",
    verbose: bool = True
):
    """
    Log a single step to terminal.
    
    Args:
        step: Step number
        observation: Environment observation
        reasoning_type: Type of reasoning ('thought', 'reflection', 'plan', or None)
        reasoning: The reasoning content
        action: The action taken
        verbose: Whether to print to terminal
    """
    if not verbose:
        return
    
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}Step {step}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
    
    # Print observation (full, no truncation)
    print(f"{Colors.YELLOW}Observation:{Colors.RESET} {observation}")
    
    # Print reasoning if present
    if reasoning_type and reasoning:
        color = {
            'thought': Colors.BLUE,
            'reflection': Colors.MAGENTA,
            'plan': Colors.GREEN,
        }.get(reasoning_type, Colors.BLUE)
        print(f"{color}{reasoning_type.capitalize()}:{Colors.RESET} {reasoning}")
    
    # Print action
    print(f"{Colors.GREEN}Action:{Colors.RESET} {action}")


def log_task_start(task_id: int, task_description: str, agent_type: str, verbose: bool = True):
    """Log the start of a new task."""
    if not verbose:
        return
    
    import sys
    print(f"\n{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Task {task_id}: {agent_type} Agent{Colors.RESET}")
    print(f"{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"{Colors.CYAN}Goal:{Colors.RESET} {task_description}")
    sys.stdout.flush()  # 确保输出立即显示


def log_task_end(success: bool, steps: int, verbose: bool = True):
    """Log the end of a task."""
    if not verbose:
        return
    
    import sys
    status = f"{Colors.GREEN}SUCCESS{Colors.RESET}" if success else f"{Colors.RED}FAILED{Colors.RESET}"
    print(f"\n{Colors.BOLD}Result: {status} in {steps} steps{Colors.RESET}")
    print(f"{'='*70}\n")
    sys.stdout.flush()  # 确保输出立即显示


def save_trajectory(
    task_id: int,
    task_description: str,
    agent_type: str,
    trajectory: list,
    success: bool,
    save: bool = True
):
    """
    Save a complete trajectory to file.
    
    Args:
        task_id: Task identifier
        task_description: The task goal
        agent_type: Type of agent ('nothinking', 'react', 'plan_and_act', 'reflact')
        trajectory: List of step dicts
        success: Whether the task was successful
        save: Whether to actually save
    """
    if not save:
        return
    
    setup_logger()
    
    result = {
        "task_id": task_id,
        "task_description": task_description,
        "agent_type": agent_type,
        "success": success,
        "num_steps": len(trajectory),
        "trajectory": trajectory,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save individual trajectory
    filename = f"{config.RESULTS_DIR}/{agent_type}_task_{task_id}.json"
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2)


def save_state_result(task_id: int, state, agent_type: str):
    """Save MPO-style state result (aligned with MPO)."""
    setup_logger()
    filename = f"{config.RESULTS_DIR}/{agent_type}_task_{task_id}.json"
    with open(filename, 'w') as f:
        json.dump(state.to_dict(), f, indent=4)


def save_summary(agent_type: str, results: list, states: list = None, read_all_from_files: bool = True):
    """
    Save summary with MPO-style metrics (success_rate + average_reward).
    
    Args:
        agent_type: Type of agent
        results: List of result dicts with 'task_id' and 'success' keys
        states: Optional list of State objects (for MPO-style evaluation)
        read_all_from_files: If True, read all existing result files and merge with current results
    """
    setup_logger()
    
    # 如果启用从文件读取，读取所有已有的结果文件
    if read_all_from_files:
        from utils.datatypes import State
        
        pattern = os.path.join(config.RESULTS_DIR, f"{agent_type}_task_*.json")
        existing_files = glob.glob(pattern)
        
        # 创建 task_id -> result 和 task_id -> state 映射，避免重复
        all_results_dict = {}
        all_states_dict = {}
        
        # 先添加当前运行的结果
        for r in results:
            all_results_dict[r['task_id']] = r
        
        if states:
            for s, r in zip(states, results):
                all_states_dict[r['task_id']] = s
        
        # 读取所有已有的文件
        for filepath in existing_files:
            try:
                with open(filepath, 'r') as f:
                    state_dict = json.load(f)
                
                # 从文件名提取 task_id
                filename = os.path.basename(filepath)
                # 格式: {agent_type}_task_{task_id}.json
                task_id_str = filename.replace(f"{agent_type}_task_", "").replace(".json", "")
                task_id = int(task_id_str)
                
                # 如果当前结果中没有这个 task_id，才添加（优先使用当前运行的结果）
                if task_id not in all_results_dict:
                    state = State.load_json(state_dict)
                    all_states_dict[task_id] = state
                    all_results_dict[task_id] = {
                        "task_id": task_id,
                        "success": state.success,
                        "num_steps": state.steps,
                        "reward": state.reward,
                    }
            except Exception as e:
                print(f"{Colors.YELLOW}Warning: Failed to read {filepath}: {e}{Colors.RESET}")
        
        # 转换为列表并按 task_id 排序
        results = sorted(all_results_dict.values(), key=lambda x: x['task_id'])
        states = [all_states_dict[r['task_id']] for r in results if r['task_id'] in all_states_dict]
    
    total = len(results)
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / total * 100 if total > 0 else 0
    
    # MPO 对齐：计算 average_reward
    rewards = []
    if states:
        # MPO style: 从 State 对象获取 reward
        rewards = [s.reward for s in states if s.reward is not None]
    else:
        # Fallback: 从 results 获取（如果存在）
        rewards = [r.get('reward', 0) for r in results if r.get('reward') is not None]
    
    average_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    summary = {
        "agent_type": agent_type,
        "total_tasks": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": success_rate,
        "average_reward": average_reward,  # MPO 指标
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    filename = f"{config.RESULTS_DIR}/{agent_type}_summary.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # MPO 对齐：输出格式
    print(f"\n{Colors.BOLD}{'='*50}{Colors.RESET}")
    print(f"{Colors.BOLD}{agent_type.upper()} Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*50}{Colors.RESET}")
    print(f"Total Tasks: {total}")
    print(f"Successes: {Colors.GREEN}{successes}{Colors.RESET}")
    print(f"Failures: {Colors.RED}{total - successes}{Colors.RESET}")
    print(f"Success Rate: {Colors.CYAN}{success_rate:.1f}%{Colors.RESET}")
    # MPO 对齐：输出 average_reward
    if rewards:
        print(f"Average Reward: {Colors.CYAN}{average_reward:.4f}{Colors.RESET}")
    print(f"Results saved to: {filename}")
    
    return summary

