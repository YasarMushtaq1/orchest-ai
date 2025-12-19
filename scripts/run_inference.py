"""
Inference script for running OrchestAI on custom tasks
"""

import torch
import argparse
import json
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system


def main():
    parser = argparse.ArgumentParser(description="Run OrchestAI inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--instruction", type=str, required=True, help="Task instruction")
    parser.add_argument("--input", type=str, default=None, help="Path to input data file")
    parser.add_argument("--output", type=str, default=None, help="Path to output file")
    parser.add_argument("--return-graph", action="store_true", help="Return workflow graph")
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
    
    # Load input data if provided
    input_data = None
    if args.input:
        with open(args.input, "r") as f:
            input_data = json.load(f)
    
    # Execute
    print(f"Executing instruction: {args.instruction}")
    result = orchestrator.execute(
        instruction=args.instruction,
        input_data=input_data,
        return_graph=args.return_graph,
    )
    
    # Print results
    print("\n=== Execution Results ===")
    print(f"Success: {result.success}")
    print(f"Total Cost: ${result.total_cost:.4f}")
    print(f"Total Latency: {result.total_latency_ms:.2f} ms")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.outputs:
        print(f"\nOutputs:")
        for task_id, output in result.outputs.items():
            print(f"  Task {task_id}: {str(output)[:100]}...")
    
    # Save results if output path provided
    if args.output:
        output_dict = {
            "success": result.success,
            "total_cost": result.total_cost,
            "total_latency_ms": result.total_latency_ms,
            "outputs": {str(k): str(v) for k, v in result.outputs.items()},
            "error": result.error,
        }
        with open(args.output, "w") as f:
            json.dump(output_dict, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

