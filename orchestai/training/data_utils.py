"""
Data Utilities: Improved data handling for training
"""

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_workflow_dataset(
    data_path: str,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Load workflow dataset from file.
    
    Args:
        data_path: Path to dataset file (JSON or JSONL)
        split: Dataset split (train/val/test)
        
    Returns:
        List of workflow examples
    """
    data_file = Path(data_path)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    workflows = []
    
    if data_file.suffix == ".jsonl":
        with open(data_file, "r") as f:
            for line in f:
                workflows.append(json.loads(line))
    else:
        with open(data_file, "r") as f:
            data = json.load(f)
            workflows = data.get(split, data)
    
    return workflows


def augment_workflow(
    workflow: Dict[str, Any],
    augmentation_type: str = "paraphrase",
) -> Dict[str, Any]:
    """
    Augment a workflow example.
    
    Args:
        workflow: Original workflow
        augmentation_type: Type of augmentation
        
    Returns:
        Augmented workflow
    """
    augmented = workflow.copy()
    
    if augmentation_type == "paraphrase":
        # Paraphrase instruction (would use LLM in practice)
        instruction = workflow.get("instruction", "")
        # Simple augmentation: add variations
        augmented["instruction"] = f"Please {instruction.lower()}"
    
    elif augmentation_type == "dependency_permutation":
        # Permute dependencies while maintaining validity
        subtasks = workflow.get("subtasks", [])
        if len(subtasks) > 1:
            # Swap non-dependent tasks
            for i in range(len(subtasks) - 1):
                if random.random() < 0.3:  # 30% chance to swap
                    if not _has_dependency(subtasks, i, i + 1):
                        subtasks[i], subtasks[i + 1] = subtasks[i + 1], subtasks[i]
            augmented["subtasks"] = subtasks
    
    elif augmentation_type == "model_substitution":
        # Substitute models with similar capabilities
        subtasks = workflow.get("subtasks", [])
        model_substitutions = {
            "gpt-4": "gpt-3.5-turbo",
            "gpt-3.5-turbo": "llama-3-8b",
            "clip-vit-base": "clip-vit-large",
        }
        
        for subtask in subtasks:
            model = subtask.get("model_selection", "")
            if model in model_substitutions:
                subtask["model_selection"] = model_substitutions[model]
        
        augmented["subtasks"] = subtasks
    
    return augmented


def _has_dependency(subtasks: List[Dict], task_id: int, dep_id: int) -> bool:
    """Check if task_id depends on dep_id"""
    deps = subtasks[task_id].get("dependencies", [])
    return dep_id in deps


def create_synthetic_workflows(
    base_workflows: List[Dict[str, Any]],
    num_synthetic: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Create synthetic workflows from base workflows.
    
    Args:
        base_workflows: Base workflow examples
        num_synthetic: Number of synthetic workflows to generate
        
    Returns:
        List of synthetic workflows
    """
    synthetic = []
    augmentation_types = ["paraphrase", "dependency_permutation", "model_substitution"]
    
    for _ in range(num_synthetic):
        base = random.choice(base_workflows)
        aug_type = random.choice(augmentation_types)
        synthetic.append(augment_workflow(base, aug_type))
    
    return synthetic


def validate_workflow(workflow: Dict[str, Any]) -> bool:
    """
    Validate workflow structure.
    
    Args:
        workflow: Workflow to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["instruction", "subtasks"]
    if not all(key in workflow for key in required_keys):
        return False
    
    subtasks = workflow["subtasks"]
    if not isinstance(subtasks, list) or len(subtasks) == 0:
        return False
    
    # Check for valid dependencies (no cycles)
    num_tasks = len(subtasks)
    visited = [False] * num_tasks
    rec_stack = [False] * num_tasks
    
    def has_cycle(node):
        visited[node] = True
        rec_stack[node] = True
        
        deps = subtasks[node].get("dependencies", [])
        for dep in deps:
            if isinstance(dep, int) and 0 <= dep < num_tasks:
                if not visited[dep]:
                    if has_cycle(dep):
                        return True
                elif rec_stack[dep]:
                    return True
        
        rec_stack[node] = False
        return False
    
    for i in range(num_tasks):
        if not visited[i]:
            if has_cycle(i):
                return False
    
    return True

