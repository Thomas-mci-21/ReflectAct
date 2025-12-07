"""
Configuration for ReflAct experiments.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Experiment Configuration
MAX_STEPS = 50  # Maximum steps per task
TEMPERATURE = 0  # For reproducibility

# ALFWorld Configuration
ALFWORLD_DATA = os.getenv("ALFWORLD_DATA", "")

# Logging Configuration
VERBOSE = True  # Print agent actions to terminal
SAVE_RESULTS = True  # Save results to file

# Results directory
RESULTS_DIR = "results"

