"""
Evaluation metrics for OrchestAI
"""

from typing import Dict, List, Any
import numpy as np


def compute_metrics(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute evaluation metrics from execution results.
    
    Args:
        results: List of execution result dictionaries
        
    Returns:
        Dictionary of metric values
    """
    if not results:
        return {}
    
    # Task success rate
    successes = [r.get("success", False) for r in results]
    success_rate = np.mean(successes)
    
    # Cost efficiency (average cost per successful task)
    costs = [r.get("total_cost", 0.0) for r in results]
    successful_costs = [r.get("total_cost", 0.0) for r in results if r.get("success", False)]
    avg_cost = np.mean(costs)
    avg_cost_successful = np.mean(successful_costs) if successful_costs else 0.0
    
    # Latency
    latencies = [r.get("total_latency_ms", 0.0) for r in results]
    avg_latency = np.mean(latencies)
    successful_latencies = [r.get("total_latency_ms", 0.0) for r in results if r.get("success", False)]
    avg_latency_successful = np.mean(successful_latencies) if successful_latencies else 0.0
    
    # Workflow validity (DAG validity)
    valid_workflows = sum(1 for r in results if r.get("workflow_valid", True))
    workflow_validity = valid_workflows / len(results) if results else 0.0
    
    # Cost per success
    cost_per_success = avg_cost / success_rate if success_rate > 0 else float("inf")
    
    return {
        "task_success_rate": success_rate,
        "avg_cost": avg_cost,
        "avg_cost_successful": avg_cost_successful,
        "avg_latency_ms": avg_latency,
        "avg_latency_successful_ms": avg_latency_successful,
        "workflow_validity": workflow_validity,
        "cost_per_success": cost_per_success,
    }


def compare_with_baseline(
    orchestai_results: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Compare OrchestAI results with baseline.
    
    Args:
        orchestai_results: OrchestAI execution results
        baseline_results: Baseline execution results
        
    Returns:
        Dictionary with comparison metrics
    """
    orchestai_metrics = compute_metrics(orchestai_results)
    baseline_metrics = compute_metrics(baseline_results)
    
    # Compute improvements
    improvements = {}
    for key in orchestai_metrics:
        if key in baseline_metrics and baseline_metrics[key] > 0:
            improvement = (
                (orchestai_metrics[key] - baseline_metrics[key]) / baseline_metrics[key] * 100
            )
            improvements[key] = improvement
    
    return {
        "orchestai": orchestai_metrics,
        "baseline": baseline_metrics,
        "improvements": improvements,
    }

