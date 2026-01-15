"""
Worker Model Layer: Manages all worker models and routes tasks to appropriate workers
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from orchestai.worker.base_worker import BaseWorker, WorkerConfig, WorkerOutput
from orchestai.worker.llm_worker import LLMWorker
from orchestai.worker.vision_worker import VisionWorker
from orchestai.worker.audio_worker import AudioWorker
from orchestai.worker.model_discovery import ModelDiscovery


@dataclass
class WorkerSelection:
    """Resolved worker selection from planner action."""
    worker_type: str
    level: int
    model_names: List[str]
    worker_ids: List[int]


class WorkerModelLayer:
    """
    Manages all worker models and provides unified interface for task execution.
    """
    
    def __init__(
        self,
        worker_configs: List[Dict[str, Any]],
        enable_discovery: bool = False,
        discovery_config: Optional[Dict[str, Any]] = None,
        routing_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize worker layer with configurations.
        
        Args:
            worker_configs: List of worker configuration dictionaries
            enable_discovery: Whether to enable dynamic model discovery
            discovery_config: Configuration for model discovery
        """
        self.workers: Dict[str, BaseWorker] = {}
        self.worker_ids: List[int] = []
        self.routing_config = routing_config or {}
        self.routing_mode = self.routing_config.get("routing_mode", "type_only")
        self.routing_levels = int(self.routing_config.get("levels", 5))
        self.worker_types = list(self.routing_config.get("types", []))
        self.routing_map = self.routing_config.get("mapping", {}) or {}
        
        # Initialize workers
        for idx, config_dict in enumerate(worker_configs):
            config = WorkerConfig(**config_dict)
            
            # Create appropriate worker type
            if config.model_type == "llm":
                worker = LLMWorker(config)
            elif config.model_type == "vision":
                worker = VisionWorker(config)
            elif config.model_type == "audio":
                worker = AudioWorker(config)
            else:
                # Default to LLM worker for unknown types
                worker = LLMWorker(config)
            
            self.workers[config.name] = worker
            self.worker_ids.append(idx)
        
        # Create mapping from model selection index to worker
        self.id_to_worker = {
            idx: worker for idx, (name, worker) in enumerate(self.workers.items())
        }
        self.worker_to_id = {
            name: idx for idx, name in enumerate(self.workers.keys())
        }
        self.worker_type_to_ids = self._build_worker_type_map()
        
        # Model discovery (optional)
        self.model_discovery: Optional[ModelDiscovery] = None
        if enable_discovery:
            self.model_discovery = ModelDiscovery(
                huggingface_token=discovery_config.get("huggingface_token") if discovery_config else None,
                local_model_endpoint=discovery_config.get("local_model_endpoint") if discovery_config else None,
            )

    def _build_worker_type_map(self) -> Dict[str, List[int]]:
        type_map: Dict[str, List[int]] = {}
        for idx, worker in self.id_to_worker.items():
            type_map.setdefault(worker.model_type, []).append(idx)
        return type_map

    def _complexity_to_level(self, complexity: Optional[float]) -> int:
        if complexity is None:
            return 1
        complexity = max(0.0, min(1.0, float(complexity)))
        return min(self.routing_levels, max(1, int(round(complexity * (self.routing_levels - 1))) + 1))

    def decode_action(
        self,
        action_index: int,
        complexity: Optional[float] = None,
    ) -> WorkerSelection:
        """
        Decode planner action into worker_type + level and resolve to worker IDs.
        """
        if not self.worker_types:
            # Fallback: treat action as direct worker ID
            worker = self.get_worker(action_index)
            if worker is None:
                return WorkerSelection("unknown", 1, [], [])
            return WorkerSelection(worker.model_type, 1, [worker.name], [action_index])

        if self.routing_mode == "type_level":
            levels = self.routing_levels
            type_idx = action_index // levels
            level = (action_index % levels) + 1
            worker_type = self.worker_types[type_idx] if type_idx < len(self.worker_types) else "unknown"
        else:
            worker_type = self.worker_types[action_index] if action_index < len(self.worker_types) else "unknown"
            level = self._complexity_to_level(complexity)

        model_names = []
        worker_ids = []
        type_map = self.routing_map.get(worker_type, {})
        level_map = type_map.get(level, [])

        for name in level_map:
            if name in self.worker_to_id:
                worker_ids.append(self.worker_to_id[name])
                model_names.append(name)

        # Fallback: if no mapping, try by model_type
        if not worker_ids:
            fallback_ids = self.worker_type_to_ids.get(worker_type, [])
            worker_ids.extend(fallback_ids)
            model_names.extend([self.id_to_worker[idx].name for idx in fallback_ids])

        return WorkerSelection(worker_type, level, model_names, worker_ids)
    
    def get_worker(self, worker_id: int) -> Optional[BaseWorker]:
        """
        Get worker by ID.
        
        Args:
            worker_id: Worker index
            
        Returns:
            Worker instance or None if not found
        """
        return self.id_to_worker.get(worker_id)
    
    def get_worker_by_name(self, name: str) -> Optional[BaseWorker]:
        """
        Get worker by name.
        
        Args:
            name: Worker name
            
        Returns:
            Worker instance or None if not found
        """
        return self.workers.get(name)
    
    def execute_task(
        self,
        worker_id: int,
        task: str,
        data: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> WorkerOutput:
        """
        Execute a task using the specified worker.
        
        Args:
            worker_id: Index of worker to use
            task: Task description
            data: Input data
            parameters: Optional task parameters
            
        Returns:
            WorkerOutput from the worker
        """
        worker = self.get_worker(worker_id)
        if worker is None:
            return WorkerOutput(
                content=None,
                metadata={},
                cost=0.0,
                latency_ms=0.0,
                success=False,
                error=f"Worker {worker_id} not found",
            )
        
        input_data = {
            "task": task,
            "data": data,
            "parameters": parameters or {},
        }
        
        return worker.process(input_data)

    def execute_task_for_selection(
        self,
        selection: WorkerSelection,
        task: str,
        data: Any,
        parameters: Optional[Dict[str, Any]] = None,
        multi_model_strategy: str = "first",
    ) -> WorkerOutput:
        """
        Execute task for a worker selection. Supports multiple models per level.
        """
        if not selection.worker_ids:
            return WorkerOutput(
                content=None,
                metadata={
                    "worker_type": selection.worker_type,
                    "level": selection.level,
                    "models": selection.model_names,
                },
                cost=0.0,
                latency_ms=0.0,
                success=False,
                error="No workers mapped to selection",
            )

        if multi_model_strategy == "first":
            return self.execute_task(selection.worker_ids[0], task, data, parameters)

        # Execute all mapped models (sequential)
        outputs: Dict[str, Any] = {}
        total_cost = 0.0
        max_latency = 0.0
        all_success = True
        errors = []

        for worker_id in selection.worker_ids:
            worker = self.get_worker(worker_id)
            if worker is None:
                all_success = False
                errors.append(f"Worker {worker_id} not found")
                continue
            result = self.execute_task(worker_id, task, data, parameters)
            outputs[worker.name] = result.content
            total_cost += result.cost
            max_latency = max(max_latency, result.latency_ms)
            if not result.success:
                all_success = False
                if result.error:
                    errors.append(result.error)

        return WorkerOutput(
            content=outputs,
            metadata={
                "worker_type": selection.worker_type,
                "level": selection.level,
                "models": selection.model_names,
                "multi_model_strategy": multi_model_strategy,
            },
            cost=total_cost,
            latency_ms=max_latency,
            success=all_success,
            error="; ".join(errors) if errors else None,
        )
    
    def get_worker_info(self, worker_id: int) -> Optional[Dict[str, Any]]:
        """
        Get information about a worker.
        
        Args:
            worker_id: Worker index
            
        Returns:
            Dictionary with worker information
        """
        worker = self.get_worker(worker_id)
        if worker is None:
            return None
        
        return {
            "id": worker_id,
            "name": worker.name,
            "type": worker.model_type,
            "cost_per_token": worker.config.cost_per_token,
            "latency_ms": worker.config.latency_ms,
        }
    
    def list_workers(self) -> List[Dict[str, Any]]:
        """
        List all available workers.
        
        Returns:
            List of worker information dictionaries
        """
        return [
            self.get_worker_info(idx) for idx in self.worker_ids
        ]
    
    def estimate_total_cost(
        self,
        worker_selections: List[int],
        task_sizes: List[int],
    ) -> float:
        """
        Estimate total cost for a set of tasks.
        
        Args:
            worker_selections: List of worker IDs for each task
            task_sizes: List of estimated task sizes (token counts)
            
        Returns:
            Total estimated cost
        """
        total_cost = 0.0
        for worker_id, size in zip(worker_selections, task_sizes):
            worker = self.get_worker(worker_id)
            if worker:
                total_cost += size * worker.config.cost_per_token
        return total_cost

