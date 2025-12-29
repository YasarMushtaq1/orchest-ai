#!/usr/bin/env python3
"""
Load and test Phase 1 trained model
"""

import torch
import sys
import os

# Add project root to path (go up from orchestai/tests/ to project root)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system


def load_phase1_model(checkpoint_path=None):
    """
    Load Phase 1 trained model.
    
    Args:
        checkpoint_path: Path to checkpoint file (default: checkpoints/phase1_best_model.pth)
        
    Returns:
        OrchestrationSystem with trained planner
    """
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set default checkpoint path if not provided
    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, "checkpoints", "phase1_best_model.pth")
    
    # Load configuration
    config_path = os.path.join(project_root, "config.yaml")
    config = load_config(config_path)
    
    # Setup system
    orchestrator = setup_system(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model weights
    orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])
    orchestrator.planner.eval()
    
    print(f"✅ Loaded Phase 1 model")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    
    return orchestrator


def test_model(orchestrator, instruction, input_data):
    """
    Test the trained model with a task.
    
    Args:
        orchestrator: OrchestrationSystem instance
        instruction: Task instruction string
        input_data: Input data dictionary
    """
    print(f"\n📝 Testing with instruction: {instruction[:50]}...")
    
    result = orchestrator.execute(
        instruction=instruction,
        input_data=input_data
    )
    
    print(f"\n📊 Execution Result:")
    print(f"   Success: {result.success}")
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Latency: {result.total_latency_ms:.2f} ms")
    print(f"   Outputs: {len(result.outputs)} outputs")
    
    if result.error:
        print(f"   Error: {result.error}")
    
    return result


if __name__ == "__main__":
    # Load model
    orchestrator = load_phase1_model()
    
    # Test with sample tasks
    test_cases = [
        {
            "instruction": "Summarize this text: Machine learning is a subset of artificial intelligence.",
            "input_data": {"text": "Machine learning is a subset of artificial intelligence."}
        },
        {
            "instruction": "Translate this to French: Hello, how are you?",
            "input_data": {"text": "Hello, how are you?"}
        }
    ]
    
    print("\n" + "="*60)
    print("Testing Phase 1 Model")
    print("="*60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        test_model(
            orchestrator,
            test_case["instruction"],
            test_case["input_data"]
        )
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)

