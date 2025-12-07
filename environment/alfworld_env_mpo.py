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
        incorporation_type: str = "query",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()
        self.bad_steps = 0
        self.max_bad_steps = 50
        self.incorporation_type = incorporation_type  # For workflow incorporation
    
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
        """Execute action and return observation, reward, done (aligned with MPO)."""
        observation, reward, done, info = self.env.step([action])
        # MPO 对齐：reward 来自 info['won'][0]
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
            # MPO 对齐：错误处理
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
        
        # Add workflow to observation if needed
        if self.incorporation_type == "observation" and self.task.workflow:
            observation += f"\n\nThis workflow maybe helpful for you to complete the task:\n{self.task.workflow}"
        
        # Add observation to history
        self.state.history.append({
            "role": "user",
            "content": observation,
        })
        
        self.state.steps += 1
        
        # MPO 对齐：超时处理
        if self.state.steps >= self.max_steps or self.bad_steps >= self.max_bad_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward
        
        # MPO 对齐：Success 判断
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
        if self.incorporation_type == "query":
            observation, messages = prompt_with_icl(
                instruction=self.instruction, 
                raw_icl=self.raw_icl[self.task.task_type], 
                cur_task=cur_task, 
                icl_num=1,
                workflow=self.task.workflow if hasattr(self.task, 'workflow') else None,
            )
        else:
            observation, messages = prompt_with_icl(
                instruction=self.instruction, 
                raw_icl=self.raw_icl[self.task.task_type], 
                cur_task=cur_task, 
                icl_num=1,
                workflow=None,
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

