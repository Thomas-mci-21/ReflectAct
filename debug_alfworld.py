#!/usr/bin/env python3
"""
Debug script to find the exact ALFWorld configuration needed.
Run this in WSL with: python debug_alfworld.py
"""
import os
import traceback

# Set environment variable
os.environ['ALFWORLD_DATA'] = os.path.expanduser('~/.cache/alfworld')
print(f"ALFWORLD_DATA: {os.environ['ALFWORLD_DATA']}")

# Test 1: Basic import
print("\n" + "="*60)
print("Test 1: Import ALFWorld")
print("="*60)
try:
    from alfworld.agents.environment import get_environment
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()
    exit(1)

# Test 2: Create environment class
print("\n" + "="*60)
print("Test 2: Create AlfredTWEnv class")
print("="*60)
try:
    alfworld_data = os.environ.get('ALFWORLD_DATA')
    config = {
        'env': {
            'type': 'AlfredTWEnv',
            'goal_desc_human_anns_prob': 0.0,
        },
        'dataset': {
            'data_path': alfworld_data,
            'eval_ood_data_path': alfworld_data,
            'eval_id_data_path': alfworld_data,
        },
        'general': {
            'save_path': './logs/',
            'training_method': 'dqn',  # Try different values: dqn, dagger, etc.
        }
    }
    
    EnvClass = get_environment('AlfredTWEnv')
    print(f"✅ Got environment class: {EnvClass}")
    
    env_instance = EnvClass(config, train_eval='eval_out_of_distribution')
    print(f"✅ Created environment instance")
    print(f"   Number of games: {getattr(env_instance, 'num_games', 'N/A')}")
    
except Exception as e:
    print(f"❌ Failed to create environment: {e}")
    traceback.print_exc()
    exit(1)

# Test 3: Initialize environment
print("\n" + "="*60)
print("Test 3: Initialize environment (init_env)")
print("="*60)
try:
    env = env_instance.init_env(batch_size=1)
    print(f"✅ Environment initialized!")
    print(f"   Type: {type(env)}")
    
except Exception as e:
    print(f"❌ init_env failed: {e}")
    traceback.print_exc()
    
    # Try to find what's needed
    print("\n" + "-"*40)
    print("Debugging: Looking at config requirements...")
    print("-"*40)
    
    # Check what the env expects
    import inspect
    if hasattr(env_instance, 'init_env'):
        sig = inspect.signature(env_instance.init_env)
        print(f"init_env signature: {sig}")
    
    # Try different training_method values
    for method in ['dqn', 'dagger', 'random', None, '']:
        print(f"\nTrying training_method='{method}'...")
        try:
            config['general']['training_method'] = method
            env_instance2 = EnvClass(config, train_eval='eval_out_of_distribution')
            env2 = env_instance2.init_env(batch_size=1)
            print(f"✅ SUCCESS with training_method='{method}'!")
            
            # Test reset
            obs, info = env2.reset()
            print(f"✅ Reset successful!")
            print(f"   Observation preview: {str(obs[0])[:100]}...")
            break
        except Exception as e2:
            print(f"   ❌ Failed: {e2}")
    
    exit(1)

# Test 4: Reset and step
print("\n" + "="*60)
print("Test 4: Reset and interact")
print("="*60)
try:
    obs, info = env.reset()
    print(f"✅ Reset successful!")
    print(f"   Observation: {obs[0][:200]}...")
    
    # Try a step
    action = ["look"]
    obs2, scores, dones, infos = env.step(action)
    print(f"✅ Step successful!")
    print(f"   New observation: {obs2[0][:200]}...")
    
except Exception as e:
    print(f"❌ Reset/step failed: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("🎉 All tests passed! ALFWorld is working!")
print("="*60)

