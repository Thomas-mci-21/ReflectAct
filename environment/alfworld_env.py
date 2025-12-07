"""
ALFWorld environment wrapper.
Based on ALFWorld (Shridhar et al., 2021)
https://github.com/alfworld/alfworld
"""
import os
from typing import Tuple, Dict, Any, Optional, List


class ALFWorldEnv:
    """
    Wrapper for ALFWorld text-based environment.
    
    ALFWorld is a text-based game environment for embodied agent tasks.
    It has 134 test tasks across 6 task types:
    - pick_and_place
    - pick_clean_then_place
    - pick_heat_then_place
    - pick_cool_then_place
    - look_at_obj
    - pick_two_obj
    """
    
    def __init__(self, split: str = "eval_out_of_distribution"):
        """
        Initialize ALFWorld environment.
        
        Args:
            split: Dataset split to use ('train', 'eval_in_distribution', 'eval_out_of_distribution')
        """
        self.split = split
        self.env = None
        self.current_task_idx = 0
        self.task_ids = []
        
        self._setup_env()
    
    def _setup_env(self):
        """Setup ALFWorld environment."""
        try:
            # Try the correct ALFWorld import pattern
            from alfworld.agents.environment import get_environment
            import os
            
            # Set ALFWorld data path if not set
            if not os.environ.get('ALFWORLD_DATA'):
                alfworld_data = os.path.expanduser('~/.cache/alfworld')
                os.environ['ALFWORLD_DATA'] = alfworld_data
                print(f"Set ALFWORLD_DATA to: {alfworld_data}")
            
            # Use complete working config - skip load_config() which requires config file
            alfworld_data = os.environ.get('ALFWORLD_DATA', os.path.expanduser('~/.cache/alfworld'))
            config = {
                'env': {
                    'type': 'AlfredTWEnv',
                    'goal_desc_human_anns_prob': 0.0,
                    'clean_debug': True,
                    'domain_randomization': False,
                    'task_types': [1, 2, 3, 4, 5, 6],
                    'expert_timeout_steps': 150,
                    'expert_type': 'handcoded'
                },
                'dataset': {
                    'data_path': alfworld_data,
                    'eval_ood_data_path': alfworld_data,
                    'eval_id_data_path': alfworld_data,
                    'num_train_games': -1,
                    'num_eval_games': -1
                },
                'general': {
                    'save_path': './logs/',
                    'seed': 42,
                    'use_templated_goals': False
                }
            }
            print("Using complete working ALFWorld config")
            
            # Get environment type (text-based)
            env_type = config['env']['type']
            
            # Setup environment
            self.env = get_environment(env_type)(config, train_eval=self.split)
            self.env = self.env.init_env(batch_size=1)
            
            # Get number of tasks
            if hasattr(self.env, 'num_games'):
                self.num_tasks = self.env.num_games
            elif hasattr(self.env, 'gamefiles'):
                self.num_tasks = len(self.env.gamefiles)
            else:
                self.num_tasks = 134  # Default ALFWorld test set size
            
            print(f"✅ ALFWorld environment loaded successfully with {self.num_tasks} tasks")
            
        except ImportError as e:
            print(f"Warning: ALFWorld not installed. Running in mock mode.")
            print(f"Import error: {e}")
            print("Install with: pip install alfworld")
            self.env = None
            self.num_tasks = 134
        except Exception as e:
            print(f"Warning: ALFWorld setup failed. Running in mock mode.")
            print(f"Setup error: {e}")
            print("Try running: export ALFWORLD_DATA=~/.cache/alfworld")
            self.env = None
            self.num_tasks = 134
    
    def reset(self, task_idx: Optional[int] = None) -> Tuple[str, str]:
        """
        Reset environment to a specific task or next task.
        
        Args:
            task_idx: Optional task index. If None, uses next task.
        
        Returns:
            Tuple of (initial_observation, task_description)
        """
        if self.env is None:
            # Mock mode for testing
            return self._mock_reset(task_idx)
        
        if task_idx is not None:
            self.current_task_idx = task_idx
        
        obs, info = self.env.reset()
        
        # Extract observation and task from ALFWorld format
        observation = obs[0] if isinstance(obs, list) else obs
        
        # Parse task description from observation
        # ALFWorld observations typically start with "You are in..."
        # and the task is usually in the 'extra.goal' field
        task_description = ""
        if 'extra.goal' in info:
            task_description = info['extra.goal'][0]
        else:
            # Try to extract from observation
            lines = observation.split('\n')
            for line in lines:
                if 'task is to' in line.lower() or 'your task' in line.lower():
                    task_description = line
                    break
            if not task_description:
                task_description = "Complete the household task."
        
        return observation, task_description
    
    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Text action to execute
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.env is None:
            return self._mock_step(action)
        
        # Execute action
        obs, scores, dones, infos = self.env.step([action])
        
        observation = obs[0] if isinstance(obs, list) else obs
        reward = scores[0] if isinstance(scores, list) else scores
        done = dones[0] if isinstance(dones, list) else dones
        
        return observation, reward, done, infos
    
    def get_task_count(self) -> int:
        """Return total number of tasks."""
        return self.num_tasks
    
    def get_all_tasks(self) -> List[int]:
        """Return list of all task indices."""
        return list(range(self.num_tasks))
    
    # Mock methods for testing without ALFWorld installed
    def _mock_reset(self, task_idx: Optional[int] = None) -> Tuple[str, str]:
        """Mock reset for testing."""
        self.mock_step = 0
        self.mock_done = False
        
        observation = """You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1."""
        
        task_description = "put some spraybottle on toilet"
        
        return observation, task_description
    
    def _mock_step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Mock step for testing."""
        self.mock_step += 1
        
        # Simple mock responses
        mock_responses = {
            "go to cabinet 1": "On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.",
            "go to cabinet 2": "The cabinet 2 is closed.",
            "open cabinet 2": "You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.",
            "take spraybottle 2 from cabinet 2": "You pick up the spraybottle 2 from the cabinet 2.",
            "go to toilet 1": "On the toilet 1, you see a soapbottle 2.",
            "put spraybottle 2 in/on toilet 1": "You put the spraybottle 2 in/on the toilet 1.",
        }
        
        action_lower = action.lower().strip()
        
        # Check for success condition
        if "put" in action_lower and "spraybottle" in action_lower and "toilet" in action_lower:
            self.mock_done = True
            return "You put the spraybottle 2 in/on the toilet 1.", 1.0, True, {}
        
        # Return mock response or default
        for key, response in mock_responses.items():
            if key.lower() in action_lower or action_lower in key.lower():
                return response, 0.0, False, {}
        
        # Default response
        return f"Nothing happens. (Mock response for: {action})", 0.0, False, {}
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except:
                pass

