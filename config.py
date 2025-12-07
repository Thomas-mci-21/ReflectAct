"""
Configuration for ReflAct experiments.

Environment variables (set in .env file or shell):
    OPENAI_API_KEY: Your OpenAI API key
    OPENAI_BASE_URL: API base URL (default: https://api.chatanywhere.tech/v1 for China relay)
    OPENAI_MODEL: Model to use (default: gpt-4o-mini-ca)
    ALFWORLD_DATA: Path to ALFWorld data directory

Example .env file:
    OPENAI_API_KEY=sk-xxxxxxxxxxxx
    OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
    OPENAI_MODEL=gpt-4o-mini-ca
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration
# 支持中转API，默认使用 chatanywhere
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-ca")

# Experiment Configuration
MAX_STEPS = 50  # Maximum steps per task (ALFWorld default)
TEMPERATURE = 0  # For reproducibility (greedy decoding)

# ALFWorld Configuration
ALFWORLD_DATA = os.getenv("ALFWORLD_DATA", "")

# Logging Configuration
VERBOSE = True  # Print agent actions to terminal
SAVE_RESULTS = True  # Save results to file

# Results directory
RESULTS_DIR = "results"

