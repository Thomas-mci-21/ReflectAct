"""
Base Agent class for ALFWorld tasks.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional
from utils.llm import call_llm, parse_action, parse_thought_or_reflection
from utils.logger import log_step, log_task_start, log_task_end, save_trajectory
import config


class BaseAgent(ABC):
    """
    Base class for all agents.
    
    Attributes:
        agent_type: String identifier for the agent type
        verbose: Whether to print to terminal
    """
    
    agent_type: str = "base"
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.trajectory = []
        self.task_description = ""
        self.step_count = 0
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt including one-shot example."""
        pass
    
    @abstractmethod
    def get_instruction(self, step: int) -> str:
        """Return the instruction for the current step."""
        pass
    
    def build_messages(self, observation: str) -> List[Dict]:
        """
        Build the messages list for LLM call.
        
        Args:
            observation: Current observation from environment
        
        Returns:
            List of message dicts for OpenAI API
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        
        # Add task description
        user_content = f"Your task is: {self.task_description}\n"
        
        # Add trajectory history
        for step_data in self.trajectory:
            user_content += f"Observation: {step_data['observation']}\n"
            if step_data.get('reasoning'):
                reasoning_type = step_data.get('reasoning_type', 'Thought')
                user_content += f"{reasoning_type.capitalize()}: {step_data['reasoning']}\n"
            user_content += f"Action: {step_data['action']}\n"
        
        # Add current observation and instruction
        user_content += f"Observation: {observation}\n"
        user_content += self.get_instruction(self.step_count + 1)
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def act(self, observation: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Generate an action given the current observation.
        
        Args:
            observation: Current observation from environment
        
        Returns:
            Tuple of (action, reasoning_type, reasoning_content)
        """
        messages = self.build_messages(observation)
        response = call_llm(messages)
        
        reasoning_type, reasoning, action = parse_thought_or_reflection(response)
        
        return action, reasoning_type, reasoning
    
    def reset(self, task_description: str):
        """Reset the agent for a new task."""
        self.trajectory = []
        self.task_description = task_description
        self.step_count = 0
    
    def step(self, observation: str) -> str:
        """
        Execute one step: observe, reason, act.
        
        Args:
            observation: Current observation from environment
        
        Returns:
            The action to take
        """
        self.step_count += 1
        
        action, reasoning_type, reasoning = self.act(observation)
        
        # Log step
        log_step(
            step=self.step_count,
            observation=observation,
            reasoning_type=reasoning_type,
            reasoning=reasoning,
            action=action,
            verbose=self.verbose
        )
        
        # Store in trajectory
        step_data = {
            "step": self.step_count,
            "observation": observation,
            "action": action,
        }
        if reasoning:
            step_data["reasoning_type"] = reasoning_type
            step_data["reasoning"] = reasoning
        
        self.trajectory.append(step_data)
        
        return action
    
    def run_task(self, env, task_id: int = 0) -> Tuple[bool, List[Dict]]:
        """
        Run a complete task.
        
        Args:
            env: ALFWorld environment wrapper
            task_id: Task identifier for logging
        
        Returns:
            Tuple of (success, trajectory)
        """
        # Reset environment and agent
        obs, task_desc = env.reset()
        self.reset(task_desc)
        
        log_task_start(task_id, task_desc, self.agent_type, self.verbose)
        
        done = False
        success = False
        
        while not done and self.step_count < config.MAX_STEPS:
            action = self.step(obs)
            obs, reward, done, info = env.step(action)
            
            if done:
                success = reward > 0  # ALFWorld returns positive reward on success
        
        log_task_end(success, self.step_count, self.verbose)
        
        # Save trajectory
        save_trajectory(
            task_id=task_id,
            task_description=task_desc,
            agent_type=self.agent_type,
            trajectory=self.trajectory,
            success=success,
            save=config.SAVE_RESULTS
        )
        
        return success, self.trajectory

