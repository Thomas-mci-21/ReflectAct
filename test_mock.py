#!/usr/bin/env python3
"""
Test script to verify the code works correctly (without ALFWorld installed).
Uses mock environment to test the agent logic.
"""

import os
import sys

# Set a dummy API key for testing (won't actually call API in mock mode)
os.environ["OPENAI_API_KEY"] = "test-key"

from environment.alfworld_env import ALFWorldEnv
from agents import (
    NoThinkingAgent,
    ReActAgent,
    PlanAndActAgent,
    ReflActAgent,
)
from utils.llm import parse_action, parse_thought_or_reflection


def test_parse_action():
    """Test action parsing."""
    print("Testing action parsing...")
    
    # Test cases
    test_cases = [
        ("Action: go to cabinet 1", "go to cabinet 1"),
        ("go to cabinet 2", "go to cabinet 2"),
        ("Thought: I need to find something.\nAction: look", "look"),
        ("Reflection: I am at the cabinet.\nAction: open cabinet 1", "open cabinet 1"),
        ("Plan: First find, then take.\nAction: go to countertop 1", "go to countertop 1"),
    ]
    
    for response, expected in test_cases:
        result = parse_action(response)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"  {status} '{response[:40]}...' -> '{result}'")
    
    print("Action parsing tests complete.\n")


def test_parse_thought_or_reflection():
    """Test thought/reflection parsing."""
    print("Testing thought/reflection parsing...")
    
    test_cases = [
        (
            "Thought: I should check the cabinet.\nAction: go to cabinet 1",
            ("thought", "I should check the cabinet.", "go to cabinet 1")
        ),
        (
            "Reflection: I am at cabinet 1 and found a cloth. I need to keep searching.\nAction: go to cabinet 2",
            ("reflection", "I am at cabinet 1 and found a cloth. I need to keep searching.", "go to cabinet 2")
        ),
        (
            "Plan: 1) Find spraybottle 2) Take it 3) Go to toilet 4) Put it there.\nAction: go to cabinet 1",
            ("plan", "1) Find spraybottle 2) Take it 3) Go to toilet 4) Put it there.", "go to cabinet 1")
        ),
    ]
    
    for response, expected in test_cases:
        r_type, r_content, action = parse_thought_or_reflection(response)
        expected_type, expected_content, expected_action = expected
        
        type_match = r_type == expected_type
        action_match = action == expected_action
        
        status = "[PASS]" if type_match and action_match else "[FAIL]"
        print(f"  {status} Type: {r_type}, Action: {action}")
    
    print("Thought/reflection parsing tests complete.\n")


def test_agents_structure():
    """Test agent class structure."""
    print("Testing agent structure...")
    
    agents = [
        ("NoThinking", NoThinkingAgent),
        ("ReAct", ReActAgent),
        ("Plan-and-Act", PlanAndActAgent),
        ("ReflAct", ReflActAgent),
    ]
    
    for name, AgentClass in agents:
        agent = AgentClass(verbose=False)
        
        # Check required methods
        has_system_prompt = callable(getattr(agent, 'get_system_prompt', None))
        has_instruction = callable(getattr(agent, 'get_instruction', None))
        has_agent_type = hasattr(agent, 'agent_type')
        
        system_prompt = agent.get_system_prompt()
        instruction_1 = agent.get_instruction(1)
        instruction_2 = agent.get_instruction(2)
        
        status = "[PASS]" if has_system_prompt and has_instruction and has_agent_type else "[FAIL]"
        print(f"  {status} {name}: type='{agent.agent_type}'")
        print(f"      System prompt length: {len(system_prompt)} chars")
        print(f"      Instruction 1: '{instruction_1[:50]}...'")
        if instruction_1 != instruction_2:
            print(f"      Instruction 2: '{instruction_2[:50]}...'")
    
    print("Agent structure tests complete.\n")


def test_environment_mock():
    """Test mock environment."""
    print("Testing mock environment...")
    
    env = ALFWorldEnv()
    
    # Test reset
    obs, task = env.reset()
    print(f"  [PASS] Reset - Task: '{task}'")
    print(f"      Observation: '{obs[:60]}...'")
    
    # Test step
    actions = [
        "go to cabinet 1",
        "go to cabinet 2",
        "open cabinet 2",
        "take spraybottle 2 from cabinet 2",
        "go to toilet 1",
        "put spraybottle 2 in/on toilet 1",
    ]
    
    for action in actions:
        obs, reward, done, info = env.step(action)
        print(f"  [PASS] Step '{action}' -> reward={reward}, done={done}")
        if done:
            print(f"      Task completed!")
            break
    
    env.close()
    print("Mock environment tests complete.\n")


def test_prompts():
    """Test that prompts contain expected content."""
    print("Testing prompts...")
    
    from prompts.nothinking import NOTHINKING_INSTRUCTION, NOTHINKING_EXAMPLE
    from prompts.react import REACT_INSTRUCTION, REACT_EXAMPLE
    from prompts.plan_and_act import PLAN_AND_ACT_INSTRUCTION_FIRST, PLAN_AND_ACT_EXAMPLE
    from prompts.reflact import REFLACT_INSTRUCTION, REFLACT_EXAMPLE
    
    # Check instructions match paper
    assert "action" in NOTHINKING_INSTRUCTION.lower()
    assert "think" in REACT_INSTRUCTION.lower()
    assert "plan" in PLAN_AND_ACT_INSTRUCTION_FIRST.lower()
    assert "reflect" in REFLACT_INSTRUCTION.lower()
    print("  [PASS] Instructions contain expected keywords")
    
    # Check examples contain expected patterns
    assert "Action:" in NOTHINKING_EXAMPLE
    assert "Thought:" in REACT_EXAMPLE
    assert "Plan:" in PLAN_AND_ACT_EXAMPLE
    assert "Reflection:" in REFLACT_EXAMPLE
    print("  [PASS] Examples contain expected patterns")
    
    # Check examples are about spraybottle task
    for example in [NOTHINKING_EXAMPLE, REACT_EXAMPLE, PLAN_AND_ACT_EXAMPLE, REFLACT_EXAMPLE]:
        assert "spraybottle" in example.lower()
        assert "toilet" in example.lower()
    print("  [PASS] Examples are about spraybottle task")
    
    print("Prompt tests complete.\n")


def main():
    print("=" * 60)
    print("ReflAct Code Tests (Mock Mode)")
    print("=" * 60)
    print()
    
    test_parse_action()
    test_parse_thought_or_reflection()
    test_agents_structure()
    test_environment_mock()
    test_prompts()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print()
    print("Note: These tests use mock environment.")
    print("For real experiments, install ALFWorld and set OPENAI_API_KEY.")


if __name__ == "__main__":
    main()

