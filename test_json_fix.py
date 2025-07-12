#!/usr/bin/env python3
"""
Test script to verify the JSON parsing fixes work correctly.
"""

import json
import re
from agents.planner import PlannerAgent
from utils.ollama_client import OllamaClient

def test_clean_json_string():
    """Test the JSON cleaning function with problematic input."""
    
    # Create a test instance
    ollama_client = OllamaClient()
    planner = PlannerAgent(ollama_client)
    
    # Test cases with problematic JSON
    test_cases = [
        # Case 1: Control characters
        '{"name": "Test\x0b Project", "description": "A test\x0c project"}',
        
        # Case 2: Unescaped newlines
        '{"name": "Test Project", "description": "A project\nwith newlines"}',
        
        # Case 3: Trailing commas
        '{"name": "Test Project", "features": ["feature1", "feature2",], "other": "value",}',
        
        # Case 4: Mixed issues
        '{"name": "Test\x08 Project", "description": "A\ntest\tproject", "features": ["f1",],}',
    ]
    
    print("Testing JSON cleaning function...")
    
    for i, test_json in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Original: {repr(test_json)}")
        
        try:
            # Try parsing original
            json.loads(test_json)
            print("✅ Original JSON is valid")
        except json.JSONDecodeError as e:
            print(f"❌ Original JSON invalid: {e}")
            
            # Try cleaning
            cleaned = planner._clean_json_string(test_json)
            print(f"Cleaned: {repr(cleaned)}")
            
            try:
                result = json.loads(cleaned)
                print("✅ Cleaned JSON is valid")
                print(f"Parsed result: {result}")
            except json.JSONDecodeError as e:
                print(f"❌ Cleaned JSON still invalid: {e}")

if __name__ == "__main__":
    test_clean_json_string()
