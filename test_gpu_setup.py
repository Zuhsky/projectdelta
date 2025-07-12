#!/usr/bin/env python3
"""
Test script to verify the GPU-optimized AI Development Team Orchestrator setup.
"""

import sys
import os
import asyncio

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.core.config import config, RunPodConfig
        print("✅ Core config imported successfully")
        
        from src.utils.ollama_client import RunPodOllamaClient, OllamaResponse
        print("✅ Ollama client imported successfully")
        
        from src.utils.logger import get_logger, setup_logging
        print("✅ Logger imported successfully")
        
        from src.agents.base import GPUOptimizedAgent
        print("✅ Base agent imported successfully")
        
        from src.core.orchestrator import GPUOrchestrator
        print("✅ GPU orchestrator imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration system."""
    print("\n🧪 Testing configuration...")
    
    try:
        from src.core.config import config
        
        print(f"✅ GPU models: {config.default_models}")
        print(f"✅ Output directory: {config.output_directory}")
        print(f"✅ Log level: {config.log_level}")
        
        # Test model config
        model_config = config.get_model_config("llama2:70b-chat")
        print(f"✅ Llama2 70B config: {model_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_gpu_client():
    """Test GPU-optimized Ollama client."""
    print("\n🧪 Testing GPU client...")
    
    try:
        from src.utils.ollama_client import RunPodOllamaClient
        
        client = RunPodOllamaClient()
        
        # Test GPU info
        gpu_info = client.get_gpu_info()
        if gpu_info:
            print(f"✅ GPU detected: {gpu_info.get('name', 'Unknown')}")
            print(f"✅ GPU memory: {gpu_info.get('memory_total', 0)}MB")
        else:
            print("⚠️ No GPU information available")
        
        # Test optimization
        optimized_config = client.optimize_for_gpu("llama2:70b-chat")
        print(f"✅ Optimized config: {optimized_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU client error: {e}")
        return False

async def test_async_functionality():
    """Test async functionality."""
    print("\n🧪 Testing async functionality...")
    
    try:
        from src.core.orchestrator import GPUOrchestrator
        
        orchestrator = GPUOrchestrator()
        
        # Test initialization
        await orchestrator.initialize_agents()
        print("✅ Agents initialized successfully")
        
        # Test welcome display
        orchestrator.display_welcome()
        print("✅ Welcome display working")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing GPU-Optimized AI Development Team Orchestrator")
    print("=" * 60)
    
    # Run synchronous tests
    tests = [
        test_imports,
        test_config,
        test_gpu_client,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # Run async tests
    print("🧪 Testing async functionality...")
    try:
        asyncio.run(test_async_functionality())
        passed += 1
        print("✅ Async functionality working")
    except Exception as e:
        print(f"❌ Async functionality error: {e}")
    
    total += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU-optimized setup is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 