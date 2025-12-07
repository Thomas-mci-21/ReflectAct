#!/usr/bin/env python3
"""
Main experiment runner for ReflAct paper reproduction.
Paper: ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection
arXiv: https://arxiv.org/abs/2505.15182

This script runs experiments on ALFWorld with 4 different reasoning strategies:
1. NoThinking - Direct action without reasoning
2. ReAct - Think about next action, then act
3. Plan-and-Act - Plan at first step, then execute
4. ReflAct - Reflect on state in relation to goal, then act

Usage:
    python run_experiment.py --agent reflact --num_tasks 10
    python run_experiment.py --agent all --num_tasks 134
"""

import argparse
import os
import sys
from typing import List, Dict

import config
from environment.alfworld_env import ALFWorldEnv
from agents import (
    NoThinkingAgent,
    ReActAgent,
    PlanAndActAgent,
    ReflActAgent,
)
from utils.logger import save_summary, Colors


AGENT_CLASSES = {
    "nothinking": NoThinkingAgent,
    "react": ReActAgent,
    "plan_and_act": PlanAndActAgent,
    "reflact": ReflActAgent,
}


def run_agent_experiments(
    agent_type: str,
    num_tasks: int = 134,
    start_task: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Run experiments for a single agent type.
    
    Args:
        agent_type: One of 'nothinking', 'react', 'plan_and_act', 'reflact'
        num_tasks: Number of tasks to run
        start_task: Starting task index
        verbose: Whether to print progress
    
    Returns:
        Summary dict with results
    """
    if agent_type not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent type: {agent_type}. Choose from {list(AGENT_CLASSES.keys())}")
    
    # Initialize environment and agent
    env = ALFWorldEnv()
    agent = AGENT_CLASSES[agent_type](verbose=verbose)
    
    results = []
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Running {agent_type.upper()} Agent on {num_tasks} tasks{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    
    for task_idx in range(start_task, start_task + num_tasks):
        try:
            success, trajectory = agent.run_task(env, task_id=task_idx)
            results.append({
                "task_id": task_idx,
                "success": success,
                "num_steps": len(trajectory),
            })
        except Exception as e:
            print(f"{Colors.RED}Error on task {task_idx}: {e}{Colors.RESET}")
            results.append({
                "task_id": task_idx,
                "success": False,
                "num_steps": 0,
                "error": str(e),
            })
    
    # Save and print summary
    summary = save_summary(agent_type, results)
    
    env.close()
    
    return summary


def run_all_agents(num_tasks: int = 134, verbose: bool = True) -> Dict[str, Dict]:
    """
    Run experiments for all agent types.
    
    Args:
        num_tasks: Number of tasks to run per agent
        verbose: Whether to print progress
    
    Returns:
        Dict mapping agent_type to summary
    """
    all_results = {}
    
    for agent_type in AGENT_CLASSES.keys():
        summary = run_agent_experiments(agent_type, num_tasks, verbose=verbose)
        all_results[agent_type] = summary
    
    # Print comparison
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}COMPARISON SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{'Agent':<15} {'Success Rate':<15} {'Successes':<12} {'Total':<10}")
    print("-" * 52)
    
    for agent_type, summary in all_results.items():
        sr = summary['success_rate']
        s = summary['successes']
        t = summary['total_tasks']
        print(f"{agent_type:<15} {sr:>10.1f}% {s:>12} {t:>10}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run ReflAct experiments on ALFWorld",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --agent reflact --num_tasks 10
  python run_experiment.py --agent react --num_tasks 134
  python run_experiment.py --agent all --num_tasks 20
  python run_experiment.py --agent nothinking --num_tasks 5 --quiet
        """
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        default="reflact",
        choices=list(AGENT_CLASSES.keys()) + ["all"],
        help="Agent type to run (default: reflact)"
    )
    
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=134,
        help="Number of tasks to run (default: 134, full test set)"
    )
    
    parser.add_argument(
        "--start_task",
        type=int,
        default=0,
        help="Starting task index (default: 0)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not config.OPENAI_API_KEY:
        print(f"{Colors.RED}Error: OPENAI_API_KEY not set.{Colors.RESET}")
        print("Set it in your environment or in a .env file")
        sys.exit(1)
    
    verbose = not args.quiet
    
    # Create results directory
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    if args.agent == "all":
        run_all_agents(args.num_tasks, verbose=verbose)
    else:
        run_agent_experiments(
            args.agent,
            args.num_tasks,
            args.start_task,
            verbose=verbose
        )


if __name__ == "__main__":
    main()

