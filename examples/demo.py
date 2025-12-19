"""
Demo script showing basic usage of OrchestAI
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system


def main():
    print("=== OrchestAI Demo ===\n")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    
    # Setup system
    print("Initializing OrchestAI system...")
    orchestrator = setup_system(config)
    print("System initialized!\n")
    
    # Example tasks
    tasks = [
        "Generate a presentation about artificial intelligence",
        "Analyze this dataset and create visualizations",
        "Summarize this research paper",
    ]
    
    # Execute each task
    for i, instruction in enumerate(tasks, 1):
        print(f"Task {i}: {instruction}")
        print("-" * 60)
        
        result = orchestrator.execute(instruction)
        
        print(f"Success: {result.success}")
        print(f"Cost: ${result.total_cost:.4f}")
        print(f"Latency: {result.total_latency_ms:.2f} ms")
        
        if result.error:
            print(f"Error: {result.error}")
        
        print()


if __name__ == "__main__":
    main()

