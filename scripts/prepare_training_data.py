#!/usr/bin/env python3
"""
Prepare training data from execution logs
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_logs_to_training_format(log_file: str, output_file: str):
    """
    Convert execution logs to training dataset format.
    
    Args:
        log_file: Path to execution log file (JSONL)
        output_file: Path to output training data file
    """
    workflows = []
    
    # Read logs
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                log_entry = json.loads(line)
                
                # Only use successful executions
                if not log_entry.get("success", False):
                    continue
                
                # Extract workflow structure
                planner_outputs = log_entry.get("planner_outputs", {})
                
                # Get model selections
                model_selections = planner_outputs.get("model_selections", [])
                if isinstance(model_selections, list) and len(model_selections) > 0:
                    if isinstance(model_selections[0], list):
                        model_selections = model_selections[0]
                
                # Get dependencies from graph
                dependencies = planner_outputs.get("workflow_graph", {}).get("adjacency", [])
                
                # Create workflow entry
                workflow = {
                    "instruction": log_entry["instruction"],
                    "subtasks": [],
                }
                
                # Create subtasks
                num_subtasks = len(model_selections)
                for i in range(num_subtasks):
                    # Get dependencies for this task
                    task_deps = []
                    if isinstance(dependencies, list) and len(dependencies) > 0:
                        if isinstance(dependencies[0], list):
                            deps_matrix = dependencies[0]
                            for j in range(len(deps_matrix)):
                                if isinstance(deps_matrix[j], list) and deps_matrix[j][i] > 0.5:
                                    task_deps.append(j)
                    
                    # Get task type (simplified - would need actual task type from decomposer)
                    task_type = 0  # Default
                    
                    workflow["subtasks"].append({
                        "id": i,
                        "task_type": task_type,
                        "dependencies": task_deps,
                        "model_selection": model_selections[i] if i < len(model_selections) else 0,
                    })
                
                workflows.append(workflow)
    
    # Save training data
    with open(output_file, "w") as f:
        json.dump(workflows, f, indent=2)
    
    print(f"✅ Converted {len(workflows)} workflows to training format")
    print(f"   Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data from execution logs")
    parser.add_argument("--log-file", type=str, required=True, help="Path to execution log file")
    parser.add_argument("--output", type=str, default="training_data.json", help="Output file path")
    args = parser.parse_args()
    
    convert_logs_to_training_format(args.log_file, args.output)


if __name__ == "__main__":
    main()

