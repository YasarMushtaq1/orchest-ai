"""
Hybrid Planner: Combines LLM-based planning with learned components for flexibility
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import json

from orchestai.planner.planner_model import PlannerModel


class HybridPlanner(nn.Module):
    """
    Hybrid planner that combines:
    - LLM-based planning for novel/unseen tasks (flexibility)
    - Learned planning for common patterns (optimization)
    
    This addresses the limitation of requiring training data while maintaining
    the benefits of learned optimization.
    """
    
    def __init__(
        self,
        learned_planner: PlannerModel,
        llm_client: Optional[Any] = None,  # LLM client for fallback
        use_llm_threshold: float = 0.5,  # Confidence threshold for using LLM
    ):
        super().__init__()
        self.learned_planner = learned_planner
        self.llm_client = llm_client
        self.use_llm_threshold = use_llm_threshold
        
    def forward(
        self,
        instructions: List[str],
        return_graph: bool = False,
        force_llm: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with hybrid planning.
        
        Args:
            instructions: List of instructions
            return_graph: Whether to return graph
            force_llm: Force LLM-based planning (for novel tasks)
            
        Returns:
            Planner outputs (same format as learned planner)
        """
        if force_llm or self.llm_client is not None:
            # Try learned planner first
            try:
                with torch.no_grad():
                    learned_outputs = self.learned_planner(instructions, return_graph=return_graph)
                
                # Check confidence (using model selection probabilities)
                avg_confidence = learned_outputs["model_selection_probs"].max(dim=-1)[0].mean().item()
                
                if avg_confidence >= self.use_llm_threshold and not force_llm:
                    # Use learned planner
                    return learned_outputs
            except Exception as e:
                # Fallback to LLM if learned planner fails
                pass
        
        # Use LLM-based planning
        if self.llm_client is not None:
            return self._llm_plan(instructions, return_graph=return_graph)
        else:
            # No LLM client, use learned planner anyway
            return self.learned_planner(instructions, return_graph=return_graph)
    
    def _llm_plan(
        self,
        instructions: List[str],
        return_graph: bool = False,
    ) -> Dict[str, Any]:
        """
        LLM-based planning (similar to HuggingGPT approach).
        
        Args:
            instructions: List of instructions
            return_graph: Whether to return graph
            
        Returns:
            Planner outputs in same format
        """
        # LLM prompt for task planning
        prompt = self._create_planning_prompt(instructions[0])
        
        # Call LLM
        response = self.llm_client.generate(prompt)
        
        # Parse LLM response into task structure
        tasks = self._parse_llm_response(response)
        
        # Convert to learned planner format
        return self._convert_to_planner_format(tasks, instructions, return_graph)
    
    def _create_planning_prompt(self, instruction: str) -> str:
        """Create prompt for LLM-based planning"""
        return f"""Parse the following user request into a structured task plan.

User Request: {instruction}

Generate a JSON list of tasks, where each task has:
- "id": task ID (integer, starting from 0)
- "task": task type (e.g., "text-extraction", "summarization", "visualization")
- "description": brief description of what this task does
- "dependencies": list of task IDs this depends on (empty if no dependencies)
- "model_type": suggested model type (e.g., "llm", "vision", "audio")

Example:
[
  {{"id": 0, "task": "text-extraction", "description": "Extract text from document", "dependencies": [], "model_type": "document"}},
  {{"id": 1, "task": "summarization", "description": "Summarize extracted text", "dependencies": [0], "model_type": "llm"}},
  {{"id": 2, "task": "visualization", "description": "Create visualization", "dependencies": [1], "model_type": "vision"}}
]

Tasks:"""
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into task list"""
        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except Exception:
            pass
        
        # Fallback: return single task
        return [{
            "id": 0,
            "task": "general",
            "description": "Process request",
            "dependencies": [],
            "model_type": "llm",
        }]
    
    def _convert_to_planner_format(
        self,
        tasks: List[Dict[str, Any]],
        instructions: List[str],
        return_graph: bool,
    ) -> Dict[str, Any]:
        """Convert LLM-generated tasks to learned planner format"""
        # This is a simplified conversion
        # In practice, would need to encode tasks into embeddings
        
        num_subtasks = len(tasks)
        batch_size = len(instructions)
        
        # Create mock embeddings (would use instruction encoder in practice)
        instruction_embeddings = torch.zeros(batch_size, 768)
        subtask_embeddings = torch.zeros(batch_size, num_subtasks, 512)
        
        # Create dependency matrix
        dependencies = torch.zeros(batch_size, num_subtasks, num_subtasks)
        for i, task in enumerate(tasks):
            for dep_id in task.get("dependencies", []):
                if dep_id < num_subtasks:
                    dependencies[0, i, dep_id] = 1.0
        
        # Create adjacency matrix
        adjacency = torch.zeros(batch_size, num_subtasks, num_subtasks)
        for i, task in enumerate(tasks):
            for dep_id in task.get("dependencies", []):
                if dep_id < num_subtasks:
                    adjacency[0, dep_id, i] = 1.0  # dep_id -> i
        
        # Model selections (would use model selector in practice)
        model_selections = [[0] * num_subtasks for _ in range(batch_size)]
        
        return {
            "instruction_embeddings": instruction_embeddings,
            "decomposition": {
                "subtask_embeddings": subtask_embeddings,
                "dependencies": dependencies,
            },
            "workflow_graph": {
                "node_embeddings": subtask_embeddings,
                "adjacency": adjacency,
                "edge_probs": adjacency.float(),
            },
            "model_selections": model_selections,
            "model_selection_probs": torch.ones(batch_size, num_subtasks, 8) / 8,
            "model_selection_values": torch.zeros(batch_size, num_subtasks, 1),
        }

