"""
Worker Model Layer: Manages all worker models and routes tasks to appropriate workers
"""

from typing import Dict, List, Optional, Any
from orchestai.worker.base_worker import BaseWorker, WorkerConfig, WorkerOutput
from orchestai.worker.llm_worker import LLMWorker
from orchestai.worker.vision_worker import VisionWorker
from orchestai.worker.audio_worker import AudioWorker


class WorkerModelLayer:
    """
    Manages all worker models and provides unified interface for task execution.
    """
    
    def __init__(self, worker_configs: List[Dict[str, Any]]):
        """
        Initialize worker layer with configurations.
        
        Args:
            worker_configs: List of worker configuration dictionaries
        """
        self.workers: Dict[str, BaseWorker] = {}
        self.worker_ids: List[int] = []
        
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

