#!/usr/bin/env python3
"""
Debug script using official ALFWorld method.
Based on: https://github.com/alfworld/alfworld

Run: python debug_alfworld.py configs/base_config.yaml
"""
import os
import sys

# Set environment variable
os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.cache/alfworld')
print(f"ALFWORLD_DATA: {os.environ['ALFWORLD_DATA']}")

# Check if config file is provided
if len(sys.argv) < 2:
    print("\nUsage: python debug_alfworld.py configs/base_config.yaml")
    print("\nTrying with default config file...")
    config_file = "configs/base_config.yaml"
else:
    config_file = sys.argv[1]

print(f"Config file: {config_file}")

# Test 1: Import
print("\n" + "="*60)
print("Test 1: Import ALFWorld")
print("="*60)
try:
    import numpy as np
    from alfworld.agents.environment import get_environment
    import alfworld.agents.modules.generic as generic
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Load config using official method
print("\n" + "="*60)
print("Test 2: Load config (official method)")
print("="*60)
try:
    # This is the official way - pass config file as command line arg
    # alfworld expects: python script.py config.yaml
    sys.argv = [sys.argv[0], config_file]
    config = generic.load_config()
    print("✅ Config loaded successfully")
    print(f"   env.type: {config['env']['type']}")
    print(f"   env.domain_randomization: {config['env'].get('domain_randomization', 'N/A')}")
except Exception as e:
    print(f"❌ Config load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create environment
print("\n" + "="*60)
print("Test 3: Create environment")
print("="*60)
try:
    env_type = config['env']['type']
    print(f"   Creating {env_type}...")
    
    env = get_environment(env_type)(config, train_eval='eval_out_of_distribution')
    print(f"✅ Environment created")
    print(f"   Number of games: {getattr(env, 'num_games', 'N/A')}")
except Exception as e:
    print(f"❌ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize environment
print("\n" + "="*60)
print("Test 4: Initialize environment (init_env)")
print("="*60)
try:
    env = env.init_env(batch_size=1)
    print("✅ Environment initialized")
except Exception as e:
    print(f"❌ init_env failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Reset and interact
print("\n" + "="*60)
print("Test 5: Reset and interact")
print("="*60)
try:
    obs, info = env.reset()
    print("✅ Reset successful")
    print(f"   Observation preview: {obs[0][:200]}...")
    
    # Get admissible commands
    if 'admissible_commands' in info:
        cmds = info['admissible_commands'][0]
        print(f"   Available commands: {len(cmds)} commands")
        print(f"   First 5: {cmds[:5]}")
    
    # Try a step
    action = ["look"]
    obs2, scores, dones, infos = env.step(action)
    print("✅ Step successful")
    print(f"   New observation: {obs2[0][:200]}...")
except Exception as e:
    print(f"❌ Reset/step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 ALL TESTS PASSED! ALFWorld is working!")
print("="*60)
print("\nYou can now run experiments with:")
print("  python run_experiment.py --agent all --num_tasks 134")
