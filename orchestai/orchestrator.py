"""
Orchestration System: Main system that coordinates Planner and Worker models
"""

import torch
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

from orchestai.planner.planner_model import PlannerModel
from orchestai.worker.worker_layer import WorkerModelLayer
from orchestai.worker.base_worker import WorkerOutput


@dataclass
class ExecutionResult:
    """Result of executing a workflow"""
    success: bool
    outputs: Dict[int, Any]  # sub-task ID -> output
    total_cost: float
    total_latency_ms: float
    workflow_graph: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    task_metrics: Optional[Dict[int, Dict[str, Any]]] = None  # Per-task metrics


class OrchestrationSystem:
    """
    Main orchestration system that coordinates the Planner Model and Worker Model Layer
    to execute complex multi-step tasks.
    """
    
    def __init__(
        self,
        planner: PlannerModel,
        worker_layer: WorkerModelLayer,
        max_workflow_depth: int = 10,
        max_parallel_tasks: int = 5,
        timeout_seconds: int = 300,
    ):
        """
        Initialize orchestration system.
        
        Args:
            planner: PlannerModel instance
            worker_layer: WorkerModelLayer instance
            max_workflow_depth: Maximum depth of workflow graph
            max_parallel_tasks: Maximum number of parallel task executions
            timeout_seconds: Maximum execution time in seconds
        """
        self.planner = planner
        self.worker_layer = worker_layer
        self.max_workflow_depth = max_workflow_depth
        self.max_parallel_tasks = max_parallel_tasks
        self.timeout_seconds = timeout_seconds
        
        # Execution history for learning
        self.execution_history = deque(maxlen=1000)
        
        # Execution logger for training data collection (lazy import to avoid circular dependency)
        self.execution_logger = None
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_tasks)
    
    def execute(
        self,
        instruction: str,
        input_data: Optional[Any] = None,
        return_graph: bool = False,
    ) -> ExecutionResult:
        """
        Execute a complex task by orchestrating multiple models.
        
        Args:
            instruction: Natural language instruction
            input_data: Optional input data for the task
            return_graph: Whether to return workflow graph
            
        Returns:
            ExecutionResult with outputs and metadata
        """
        start_time = time.time()
        
        try:
            # 1. Plan: Decompose task and generate workflow
            with torch.no_grad():
                planner_outputs = self.planner(
                    [instruction],
                    return_graph=return_graph,
                )
            
            # Extract workflow information
            model_selections = planner_outputs["model_selections"][0]
            adjacency = planner_outputs["workflow_graph"]["adjacency"][0]
            num_subtasks = len(model_selections)
            
            # 2. Execute workflow following topological order with parallel execution
            execution_order = self._topological_sort(adjacency, num_subtasks)
            
            # Track execution state
            task_outputs: Dict[int, Any] = {}
            task_results: Dict[int, WorkerOutput] = {}
            task_metrics: Dict[int, Dict[str, Any]] = {}
            total_cost = 0.0
            total_latency = 0.0
            retry_count = 0
            
            # Execute tasks with parallel execution where possible
            completed_tasks = set()
            ready_tasks = set(execution_order[:1] if execution_order else [])  # Start with first task
            
            while ready_tasks or len(completed_tasks) < num_subtasks:
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    return ExecutionResult(
                        success=False,
                        outputs=task_outputs,
                        total_cost=total_cost,
                        total_latency_ms=total_latency,
                        error="Execution timeout",
                        retry_count=retry_count,
                        task_metrics=task_metrics,
                    )
                
                # Execute ready tasks in parallel (up to max_parallel_tasks)
                tasks_to_execute = list(ready_tasks)[:self.max_parallel_tasks]
                execution_results = {}
                
                # Execute tasks in parallel
                futures = {}
                for task_id in tasks_to_execute:
                    future = self.executor.submit(
                        self._execute_single_task,
                        task_id,
                        model_selections[task_id],
                        input_data,
                        task_results,
                        adjacency,
                    )
                    futures[future] = task_id
                
                # Collect results
                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        worker_output, task_metric = future.result()
                        execution_results[task_id] = (worker_output, task_metric)
                    except Exception as e:
                        # Task failed, try retry
                        worker_output = WorkerOutput(
                            content=None,
                            metadata={},
                            cost=0.0,
                            latency_ms=0.0,
                            success=False,
                            error=str(e),
                        )
                        execution_results[task_id] = (worker_output, {"error": str(e)})
                
                # Process results
                for task_id, (worker_output, task_metric) in execution_results.items():
                    task_metrics[task_id] = task_metric
                    
                    # Retry logic for failed tasks
                    if not worker_output.success and retry_count < self.max_retries:
                        retry_count += 1
                        time.sleep(self.retry_delay)
                        # Retry the task
                        worker_output, task_metric = self._execute_single_task(
                            task_id,
                            model_selections[task_id],
                            input_data,
                            task_results,
                            adjacency,
                        )
                        task_metrics[task_id] = task_metric
                    
                    # Store results
                    task_results[task_id] = worker_output
                    task_outputs[task_id] = worker_output.content
                    completed_tasks.add(task_id)
                    
                    # Accumulate metrics
                    total_cost += worker_output.cost
                    total_latency += worker_output.latency_ms
                    
                    # Check for failures after retry
                    if not worker_output.success:
                        return ExecutionResult(
                            success=False,
                            outputs=task_outputs,
                            total_cost=total_cost,
                            total_latency_ms=total_latency,
                            error=worker_output.error or f"Task {task_id} execution failed after retries",
                            retry_count=retry_count,
                            task_metrics=task_metrics,
                        )
                
                # Update ready tasks for next iteration
                ready_tasks = self._get_ready_tasks(
                    execution_order,
                    completed_tasks,
                    adjacency,
                    num_subtasks,
                )
            
            # 3. Combine outputs (could be customized based on task type)
            final_output = self._combine_outputs(task_outputs, execution_order)
            
            # Record execution for learning
            final_result = ExecutionResult(
                success=True,
                outputs=task_outputs,
                total_cost=total_cost,
                total_latency_ms=total_latency,
                workflow_graph=planner_outputs.get("graphs", [None])[0] if return_graph else None,
                retry_count=retry_count,
                task_metrics=task_metrics,
            )
            
            self._record_execution(
                instruction=instruction,
                planner_outputs=planner_outputs,
                execution_result=final_result,
            )
            
            # Log execution for training data (lazy initialization)
            if self.execution_logger is None:
                from orchestai.utils.execution_logger import ExecutionLogger
                self.execution_logger = ExecutionLogger()
            
            self.execution_logger.log_execution(
                instruction=instruction,
                planner_outputs=planner_outputs,
                execution_result=final_result,
                success=True,
            )
            
            return ExecutionResult(
                success=True,
                outputs={-1: final_output},  # Final combined output
                total_cost=total_cost,
                total_latency_ms=total_latency,
                workflow_graph=planner_outputs.get("graphs", [None])[0] if return_graph else None,
                retry_count=retry_count,
                task_metrics=task_metrics,
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                outputs={},
                total_cost=0.0,
                total_latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    def _topological_sort(
        self,
        adjacency: torch.Tensor,
        num_nodes: int,
    ) -> List[int]:
        """
        Perform topological sort to determine execution order.
        
        Args:
            adjacency: [num_nodes, num_nodes] adjacency matrix
            num_nodes: Number of nodes
            
        Returns:
            List of node IDs in topological order
        """
        # Build in-degree map
        in_degree = [0] * num_nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency[j, i].item() > 0.5:  # Edge from j to i
                    in_degree[i] += 1
        
        # Kahn's algorithm
        queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Remove edges from this node
            for neighbor in range(num_nodes):
                if adjacency[node, neighbor].item() > 0.5:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return result
    
    def _prepare_task_input(
        self,
        task_id: int,
        initial_input: Any,
        task_results: Dict[int, WorkerOutput],
        adjacency: torch.Tensor,
    ) -> Any:
        """
        Prepare input for a task based on its dependencies.
        
        Args:
            task_id: Current task ID
            initial_input: Initial input data
            task_results: Results from completed tasks
            adjacency: Workflow adjacency matrix
            
        Returns:
            Prepared input data for the task
        """
        # Collect outputs from dependencies
        dependency_outputs = []
        num_nodes = adjacency.size(0)
        
        for dep_id in range(num_nodes):
            if adjacency[dep_id, task_id].item() > 0.5 and dep_id in task_results:
                dependency_outputs.append(task_results[dep_id].content)
        
        # Combine dependency outputs with initial input
        if dependency_outputs:
            # If we have dependencies, use the first dependency output as data
            # (workers expect string data)
            if len(dependency_outputs) == 1:
                data = dependency_outputs[0]
                # If data is a dict with "text" key, extract it
                if isinstance(data, dict) and "text" in data:
                    return data["text"]
                elif isinstance(data, str):
                    return data
                else:
                    return str(data)
            else:
                # Multiple dependencies - combine them
                combined = " ".join(str(dep) for dep in dependency_outputs)
                return combined
        else:
            # No dependencies - use initial input
            if initial_input is None:
                return ""
            # If initial_input is a dict, extract text field
            if isinstance(initial_input, dict):
                return initial_input.get("text", str(initial_input))
            return str(initial_input)
    
    def _combine_outputs(
        self,
        task_outputs: Dict[int, Any],
        execution_order: List[int],
    ) -> Any:
        """
        Combine outputs from all sub-tasks into final result.
        
        Args:
            task_outputs: Dictionary of task outputs
            execution_order: Order of task execution
            
        Returns:
            Combined output
        """
        # Simple combination: return the last task's output
        # In production, this could be more sophisticated (e.g., structured combination)
        if execution_order:
            last_task = execution_order[-1]
            return task_outputs.get(last_task, None)
        return None
    
    def _execute_single_task(
        self,
        task_id: int,
        worker_id: int,
        input_data: Any,
        task_results: Dict[int, WorkerOutput],
        adjacency: torch.Tensor,
    ) -> Tuple[WorkerOutput, Dict[str, Any]]:
        """
        Execute a single task with error handling.
        
        Returns:
            Tuple of (WorkerOutput, task_metrics)
        """
        task_start_time = time.time()
        
        try:
            # Prepare task input
            task_input = self._prepare_task_input(
                task_id,
                input_data,
                task_results,
                adjacency,
            )
            
            # Execute task
            task_description = f"subtask_{task_id}"
            worker_output = self.worker_layer.execute_task(
                worker_id=worker_id,
                task=task_description,
                data=task_input,
            )
            
            task_metric = {
                "task_id": task_id,
                "worker_id": worker_id,
                "latency_ms": worker_output.latency_ms,
                "cost": worker_output.cost,
                "success": worker_output.success,
                "timestamp": task_start_time,
            }
            
            return worker_output, task_metric
            
        except Exception as e:
            return WorkerOutput(
                content=None,
                metadata={},
                cost=0.0,
                latency_ms=(time.time() - task_start_time) * 1000,
                success=False,
                error=str(e),
            ), {
                "task_id": task_id,
                "worker_id": worker_id,
                "error": str(e),
                "timestamp": task_start_time,
            }
    
    def _get_ready_tasks(
        self,
        execution_order: List[int],
        completed_tasks: set,
        adjacency: torch.Tensor,
        num_subtasks: int,
    ) -> set:
        """
        Get tasks that are ready to execute (all dependencies completed).
        
        Args:
            execution_order: Topological order of tasks
            completed_tasks: Set of completed task IDs
            adjacency: Workflow adjacency matrix
            num_subtasks: Total number of subtasks
            
        Returns:
            Set of ready task IDs
        """
        ready = set()
        
        for task_id in execution_order:
            if task_id in completed_tasks:
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = True
            for dep_id in range(num_subtasks):
                if adjacency[dep_id, task_id].item() > 0.5:  # dep_id is dependency of task_id
                    if dep_id not in completed_tasks:
                        deps_satisfied = False
                        break
            
            if deps_satisfied:
                ready.add(task_id)
        
        return ready
    
    def _record_execution(
        self,
        instruction: str,
        planner_outputs: Dict[str, Any],
        execution_result: ExecutionResult,
    ):
        """Record execution for continuous learning"""
        self.execution_history.append({
            "instruction": instruction,
            "planner_outputs": planner_outputs,
            "result": execution_result,
            "timestamp": time.time(),
        })
    
    def __del__(self):
        """Cleanup thread pool"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

