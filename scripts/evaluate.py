"""
Evaluation script for OrchestAI
"""

import torch
import argparse
import json
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system
from orchestai.evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate OrchestAI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, default=None, help="Path to benchmark tasks JSON")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup system
    orchestrator = setup_system(config, device=device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Load benchmark tasks
    if args.tasks:
        with open(args.tasks, "r") as f:
            benchmark_tasks = json.load(f)
    else:
        # Default benchmark tasks
        benchmark_tasks = [
            {
                "instruction": "Generate a presentation about machine learning",
                "input_data": None,
            },
            {
                "instruction": "Analyze this dataset and create visualizations",
                "input_data": {"data": "sample_data.csv"},
            },
            {
                "instruction": "Summarize this research paper",
                "input_data": {"paper": "sample_paper.pdf"},
            },
        ]
    
    # Initialize evaluator
    evaluator = Evaluator(
        orchestrator=orchestrator,
        benchmark_tasks=benchmark_tasks,
    )
    
    # Evaluate
    print("Evaluating OrchestAI...")
    results = evaluator.evaluate(return_detailed=True)
    
    # Print results
    print("\n=== Evaluation Results ===")
    metrics = results["metrics"]
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

