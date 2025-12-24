"""
Execution Logger: Collect execution data for training
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class ExecutionLogger:
    """
    Logs execution data for training dataset creation.
    """
    
    def __init__(self, log_dir: str = "execution_logs"):
        """
        Initialize execution logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logs: List[Dict[str, Any]] = []
    
    def log_execution(
        self,
        instruction: str,
        planner_outputs: Dict[str, Any],
        execution_result: Any,
        success: bool,
    ):
        """
        Log an execution for training data collection.
        
        Args:
            instruction: User instruction
            planner_outputs: Outputs from planner
            execution_result: ExecutionResult object
            success: Whether execution succeeded
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction,
            "success": success,
            "cost": execution_result.total_cost,
            "latency_ms": execution_result.total_latency_ms,
            "planner_outputs": self._serialize_planner_outputs(planner_outputs),
            "execution_result": self._serialize_execution_result(execution_result),
        }
        
        self.logs.append(log_entry)
        
        # Save to file (append mode)
        log_file = self.log_dir / f"executions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _serialize_planner_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize planner outputs (convert tensors to lists)"""
        import torch
        
        def convert_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor(v) for v in obj]
            else:
                return obj
        
        return convert_tensor(outputs)
    
    def _serialize_execution_result(self, result: Any) -> Dict[str, Any]:
        """Serialize execution result"""
        return {
            "success": result.success,
            "total_cost": result.total_cost,
            "total_latency_ms": result.total_latency_ms,
            "outputs": {str(k): str(v) for k, v in result.outputs.items()},
            "error": result.error,
            "retry_count": getattr(result, 'retry_count', 0),
        }
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logged executions"""
        return self.logs
    
    def clear_logs(self):
        """Clear in-memory logs"""
        self.logs = []

