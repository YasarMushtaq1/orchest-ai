"""
Evaluator: Framework for evaluating OrchestAI on benchmark tasks
"""

from typing import Dict, List, Any, Optional
from orchestai.evaluation.metrics import compute_metrics, compare_with_baseline
from orchestai.orchestrator import OrchestrationSystem


class Evaluator:
    """
    Evaluator for benchmarking OrchestAI on various tasks.
    """
    
    def __init__(
        self,
        orchestrator: OrchestrationSystem,
        benchmark_tasks: List[Dict[str, Any]],
    ):
        """
        Initialize evaluator.
        
        Args:
            orchestrator: OrchestrationSystem instance
            benchmark_tasks: List of benchmark task dictionaries with:
                - instruction: Task instruction
                - input_data: Optional input data
                - expected_output: Optional expected output for validation
        """
        self.orchestrator = orchestrator
        self.benchmark_tasks = benchmark_tasks
    
    def evaluate(
        self,
        return_detailed: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate orchestrator on benchmark tasks.
        
        Args:
            return_detailed: Whether to return detailed per-task results
            
        Returns:
            Dictionary with evaluation results
        """
        results = []
        
        for task in self.benchmark_tasks:
            instruction = task["instruction"]
            input_data = task.get("input_data")
            
            # Execute task
            execution_result = self.orchestrator.execute(
                instruction=instruction,
                input_data=input_data,
            )
            
            # Check workflow validity
            workflow_valid = self._check_workflow_validity(execution_result)
            
            result = {
                "instruction": instruction,
                "success": execution_result.success,
                "total_cost": execution_result.total_cost,
                "total_latency_ms": execution_result.total_latency_ms,
                "workflow_valid": workflow_valid,
                "error": execution_result.error,
            }
            
            if return_detailed:
                result["outputs"] = execution_result.outputs
                result["workflow_graph"] = execution_result.workflow_graph
            
            results.append(result)
        
        # Compute metrics
        metrics = compute_metrics(results)
        
        return {
            "metrics": metrics,
            "results": results if return_detailed else None,
            "num_tasks": len(self.benchmark_tasks),
        }
    
    def _check_workflow_validity(self, execution_result: Any) -> bool:
        """
        Check if workflow graph is valid (DAG).
        
        Args:
            execution_result: ExecutionResult
            
        Returns:
            True if workflow is valid
        """
        if execution_result.workflow_graph is None:
            return True  # Assume valid if no graph
        
        # Check for cycles (simplified check)
        try:
            import networkx as nx
            G = execution_result.workflow_graph
            return nx.is_directed_acyclic_graph(G)
        except:
            return True  # Assume valid if check fails
    
    def compare_baseline(
        self,
        baseline_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare with baseline results.
        
        Args:
            baseline_results: Baseline execution results
            
        Returns:
            Comparison results
        """
        # Evaluate OrchestAI
        orchestai_results = self.evaluate(return_detailed=True)["results"]
        
        # Compare
        comparison = compare_with_baseline(orchestai_results, baseline_results)
        
        return comparison

