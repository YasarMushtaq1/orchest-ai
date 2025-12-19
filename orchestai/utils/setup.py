"""
System setup utilities
"""

import torch
from typing import Dict, Any, Optional
from orchestai.planner.planner_model import PlannerModel
from orchestai.worker.worker_layer import WorkerModelLayer
from orchestai.orchestrator import OrchestrationSystem


def setup_system(config: Dict[str, Any], device: Optional[torch.device] = None) -> OrchestrationSystem:
    """
    Set up complete OrchestAI system from configuration.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device (default: auto-detect)
        
    Returns:
        Initialized OrchestrationSystem
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract configurations
    planner_config = config["planner"]
    worker_configs = config["worker_models"]
    system_config = config.get("system", {})
    
    # Initialize Planner Model
    planner = PlannerModel(
        instruction_encoder_config=planner_config["instruction_encoder"],
        task_decomposer_config=planner_config["task_decomposer"],
        graph_generator_config=planner_config["workflow_graph_generator"],
        model_selector_config=planner_config["model_selector"],
    )
    planner.to(device)
    
    # Initialize Worker Model Layer
    worker_layer = WorkerModelLayer(worker_configs)
    
    # Initialize Orchestration System
    orchestrator = OrchestrationSystem(
        planner=planner,
        worker_layer=worker_layer,
        max_workflow_depth=system_config.get("max_workflow_depth", 10),
        max_parallel_tasks=system_config.get("max_parallel_tasks", 5),
        timeout_seconds=system_config.get("timeout_seconds", 300),
    )
    
    return orchestrator

