"""
Cost Optimization Utilities: Track and optimize API costs
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class CostMetrics:
    """Cost metrics for tracking"""
    total_cost: float = 0.0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_task_type: Dict[str, float] = field(default_factory=dict)
    num_calls: int = 0
    num_calls_by_model: Dict[str, int] = field(default_factory=dict)
    avg_cost_per_call: float = 0.0
    timestamp: float = field(default_factory=time.time)


class CostOptimizer:
    """
    Tracks and optimizes API costs across worker models.
    """
    
    def __init__(self):
        self.metrics = CostMetrics()
        self.history: List[Dict[str, Any]] = []
        self.budget: Optional[float] = None
        self.budget_warning_threshold: float = 0.8  # Warn at 80% of budget
    
    def record_cost(
        self,
        model_name: str,
        task_type: str,
        cost: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a cost event.
        
        Args:
            model_name: Name of the model used
            task_type: Type of task
            cost: Cost incurred
            metadata: Optional metadata
        """
        self.metrics.total_cost += cost
        self.metrics.cost_by_model[model_name] = (
            self.metrics.cost_by_model.get(model_name, 0.0) + cost
        )
        self.metrics.cost_by_task_type[task_type] = (
            self.metrics.cost_by_task_type.get(task_type, 0.0) + cost
        )
        self.metrics.num_calls += 1
        self.metrics.num_calls_by_model[model_name] = (
            self.metrics.num_calls_by_model.get(model_name, 0) + 1
        )
        self.metrics.avg_cost_per_call = (
            self.metrics.total_cost / self.metrics.num_calls
            if self.metrics.num_calls > 0 else 0.0
        )
        
        # Record in history
        self.history.append({
            "model_name": model_name,
            "task_type": task_type,
            "cost": cost,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        
        # Check budget
        if self.budget is not None:
            if self.metrics.total_cost >= self.budget:
                raise ValueError(f"Budget exceeded: ${self.metrics.total_cost:.4f} >= ${self.budget:.4f}")
            elif self.metrics.total_cost >= self.budget * self.budget_warning_threshold:
                print(f"Warning: Budget usage at {self.metrics.total_cost / self.budget * 100:.1f}%")
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get summary of costs"""
        return {
            "total_cost": self.metrics.total_cost,
            "num_calls": self.metrics.num_calls,
            "avg_cost_per_call": self.metrics.avg_cost_per_call,
            "cost_by_model": dict(self.metrics.cost_by_model),
            "cost_by_task_type": dict(self.metrics.cost_by_task_type),
            "num_calls_by_model": dict(self.metrics.num_calls_by_model),
            "budget_remaining": (
                self.budget - self.metrics.total_cost
                if self.budget is not None else None
            ),
        }
    
    def suggest_optimization(
        self,
        task_type: str,
        available_models: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Suggest a cheaper model for a task type based on historical data.
        
        Args:
            task_type: Type of task
            available_models: List of available models with cost info
            
        Returns:
            Suggested model name or None
        """
        if not available_models:
            return None
        
        # Find models that can handle this task type
        suitable_models = [
            m for m in available_models
            if task_type in m.get("capabilities", [])
        ]
        
        if not suitable_models:
            return None
        
        # Sort by cost
        suitable_models.sort(key=lambda x: x.get("cost_per_token", float("inf")))
        
        # Return cheapest suitable model
        return suitable_models[0].get("name")
    
    def set_budget(self, budget: float):
        """Set a budget limit"""
        self.budget = budget
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = CostMetrics()
        self.history = []

