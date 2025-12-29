#!/usr/bin/env python3
"""
Test Phase 1 Trained Model
Tests the trained model on various tasks and evaluates performance
"""

import torch
import sys
import os
import time
from typing import Dict, Any

# Add project root to path (go up from orchestai/tests/ to project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system


def load_trained_model(checkpoint_path=None):
    """Load the trained Phase 1 model"""
    if checkpoint_path is None:
        # Get project root and construct checkpoint path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        checkpoint_path = os.path.join(project_root, "checkpoints", "phase1_best_model.pth")
    """Load the trained Phase 1 model"""
    print("="*60)
    print("Loading Phase 1 Trained Model")
    print("="*60)
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Setup system
    print("Setting up system...")
    orchestrator = setup_system(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model weights
    orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])
    orchestrator.planner.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"   Training Loss: ~2.04")
    print()
    
    return orchestrator


def test_task(orchestrator, test_name: str, instruction: str, input_data: Dict[str, Any]):
    """Test a single task"""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    print(f"Instruction: {instruction}")
    print(f"Input: {input_data}")
    print()
    
    start_time = time.time()
    
    try:
        result = orchestrator.execute(
            instruction=instruction,
            input_data=input_data
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Execution completed in {elapsed_time:.2f}s")
        print(f"\n📊 Results:")
        print(f"   Success: {result.success}")
        print(f"   Cost: ${result.total_cost:.4f}")
        print(f"   Latency: {result.total_latency_ms:.2f} ms")
        print(f"   Retry Count: {result.retry_count}")
        print(f"   Outputs: {len(result.outputs)} outputs")
        
        if result.workflow_graph:
            print(f"   Workflow Graph: Generated")
        
        if result.task_metrics:
            print(f"   Task Metrics: {len(result.task_metrics)} tasks executed")
            for task_id, metrics in list(result.task_metrics.items())[:3]:
                print(f"      Task {task_id}: cost=${metrics.get('cost', 0):.4f}, "
                      f"latency={metrics.get('latency_ms', 0):.2f}ms")
        
        if result.error:
            print(f"   ⚠️  Error: {result.error}")
        
        if result.outputs:
            print(f"\n📝 Output Preview:")
            for key, value in list(result.outputs.items())[:2]:
                output_preview = str(value)[:200] if value else "None"
                print(f"   Output {key}: {output_preview}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_test_suite():
    """Run a comprehensive test suite"""
    print("\n" + "="*60)
    print("Phase 1 Model Test Suite")
    print("="*60)
    
    # Load model
    orchestrator = load_trained_model()
    if orchestrator is None:
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Text Summarization",
            "instruction": "Summarize this text: Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "input_data": {"text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."}
        },
        {
            "name": "Translation Task",
            "instruction": "Translate this text to French: Hello, how are you today?",
            "input_data": {"text": "Hello, how are you today?"}
        },
        {
            "name": "Complex Multi-Step Task",
            "instruction": "Summarize this document and create a brief presentation outline",
            "input_data": {"text": "Artificial intelligence is transforming industries. Machine learning enables computers to learn from data. Deep learning uses neural networks for complex pattern recognition."}
        },
        {
            "name": "Text Analysis",
            "instruction": "Analyze this text and extract key points",
            "input_data": {"text": "Climate change is one of the most pressing issues of our time. It requires immediate action from governments, businesses, and individuals worldwide."}
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*60}")
        print(f"Test Case {i}/{len(test_cases)}")
        print(f"{'#'*60}")
        
        result = test_task(
            orchestrator,
            test_case["name"],
            test_case["instruction"],
            test_case["input_data"]
        )
        
        results.append({
            "test": test_case["name"],
            "success": result.success if result else False,
            "cost": result.total_cost if result else 0.0,
            "latency": result.total_latency_ms if result else 0.0,
        })
        
        # Small delay between tests
        time.sleep(1)
    
    # Summary
    print("\n\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    successful = sum(1 for r in results if r["success"])
    total_cost = sum(r["cost"] for r in results)
    avg_latency = sum(r["latency"] for r in results) / len(results) if results else 0
    
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"Total Cost: ${total_cost:.4f}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)


def test_model_capabilities():
    """Test specific model capabilities"""
    print("\n" + "="*60)
    print("Testing Model Capabilities")
    print("="*60)
    
    orchestrator = load_trained_model()
    if orchestrator is None:
        return
    
    # Test 1: Check if model can decompose tasks
    print("\n1. Testing Task Decomposition...")
    test_task(
        orchestrator,
        "Task Decomposition Test",
        "Break down this task: Create a presentation about AI",
        {"topic": "AI"}
    )
    
    # Test 2: Check workflow graph generation
    print("\n2. Testing Workflow Graph Generation...")
    result = test_task(
        orchestrator,
        "Workflow Graph Test",
        "Process this document: summarize and translate",
        {"text": "Sample document text"}
    )
    
    if result and result.workflow_graph:
        print("   ✅ Workflow graph generated successfully")
    else:
        print("   ⚠️  Workflow graph not returned")
    
    # Test 3: Check model selection
    print("\n3. Testing Model Selection...")
    if result and result.task_metrics:
        print(f"   ✅ Model selected workers for {len(result.task_metrics)} tasks")
        for task_id, metrics in result.task_metrics.items():
            print(f"      Task {task_id}: Worker {metrics.get('worker_id', 'N/A')}")
    else:
        print("   ⚠️  Model selection not visible in results")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Phase 1 Trained Model")
    parser.add_argument("--capabilities", action="store_true", 
                       help="Test specific model capabilities")
    parser.add_argument("--checkpoint", type=str, 
                       default=None,
                       help="Path to checkpoint file (default: checkpoints/phase1_best_model.pth)")
    
    args = parser.parse_args()
    
    if args.capabilities:
        test_model_capabilities()
    else:
        run_test_suite()

