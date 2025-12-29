"""
Planner Model: Main orchestrator combining all planner components
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from orchestai.planner.instruction_encoder import InstructionEncoder
from orchestai.planner.task_decomposer import TaskDecomposer
from orchestai.planner.graph_generator import WorkflowGraphGenerator
from orchestai.planner.model_selector import RLModelSelector


class PlannerModel(nn.Module):
    """
    Main Planner Model that integrates all components:
    - Instruction Encoder
    - Task Decomposer
    - Workflow Graph Generator
    - RL-Based Model Selector
    """
    
    def __init__(
        self,
        instruction_encoder_config: Dict,
        task_decomposer_config: Dict,
        graph_generator_config: Dict,
        model_selector_config: Dict,
    ):
        super().__init__()
        
        # Initialize components
        self.instruction_encoder = InstructionEncoder(**instruction_encoder_config)
        
        # Task decomposer takes instruction encoder output
        task_decomposer_config["input_dim"] = instruction_encoder_config["hidden_size"]
        self.task_decomposer = TaskDecomposer(**task_decomposer_config)
        
        # Graph generator takes task decomposer output
        graph_generator_config["input_dim"] = task_decomposer_config["hidden_size"]
        self.graph_generator = WorkflowGraphGenerator(**graph_generator_config)
        
        # Model selector takes combined state (instruction + graph node embeddings)
        model_selector_config["state_dim"] = (
            instruction_encoder_config["hidden_size"] +
            graph_generator_config["output_dim"]
        )
        self.model_selector = RLModelSelector(**model_selector_config)
        
    def forward(
        self,
        instructions: List[str],
        return_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through planner.
        
        Args:
            instructions: List of natural language instruction strings
            return_graph: Whether to return NetworkX graph objects
            
        Returns:
            Dictionary containing all intermediate outputs and final predictions
        """
        # 1. Encode instructions
        instruction_outputs = self.instruction_encoder(instructions)
        instruction_embeddings = instruction_outputs["embeddings"]  # [batch_size, hidden_size]
        
        # 2. Decompose tasks
        decomposition_outputs = self.task_decomposer(instruction_embeddings)
        
        # 3. Generate workflow graph
        graph_outputs = self.graph_generator(
            decomposition_outputs["subtask_embeddings"],
            decomposition_outputs["dependencies"],
            return_graph=return_graph,
        )
        
        # 4. Select models for each sub-task
        batch_size = instruction_embeddings.size(0)
        num_subtasks = decomposition_outputs["subtask_embeddings"].size(1)
        
        # Combine instruction embedding with each node embedding for model selection
        model_selections = []
        model_selection_probs = []
        model_selection_values = []
        
        for b in range(batch_size):
            batch_selections = []
            batch_probs = []
            batch_values = []
            
            for i in range(num_subtasks):
                # Combine instruction embedding with node embedding
                node_emb = graph_outputs["node_embeddings"][b, i]  # [output_dim]
                inst_emb = instruction_embeddings[b]  # [hidden_size]
                combined_state = torch.cat([inst_emb, node_emb])  # [hidden_size + output_dim]
                
                # Select model for this sub-task
                selector_outputs = self.model_selector(combined_state.unsqueeze(0), return_value=True)
                action, log_prob, entropy = self.model_selector.select_action(combined_state)
                
                batch_selections.append(action.item())
                batch_probs.append(selector_outputs["action_probs"].squeeze(0))
                batch_values.append(selector_outputs["value"].squeeze(0))
            
            model_selections.append(batch_selections)
            model_selection_probs.append(torch.stack(batch_probs))
            model_selection_values.append(torch.stack(batch_values))
        
        # Combine all outputs
        result = {
            "instruction_embeddings": instruction_embeddings,
            "decomposition": decomposition_outputs,
            "workflow_graph": graph_outputs,
            "model_selections": model_selections,
            "model_selection_probs": torch.stack(model_selection_probs),
            "model_selection_values": torch.stack(model_selection_values),
        }
        
        if return_graph and "graphs" in graph_outputs:
            result["graphs"] = graph_outputs["graphs"]
        
        return result
    
    def compute_supervised_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        lambda_dag: float = 0.3,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised learning loss for Phase 1 training.
        
        Args:
            outputs: Output dictionary from forward()
            targets: Dictionary containing target values:
                - subtask_sequence: Target sub-task sequence
                - task_types: Target task types
                - dependencies: Target dependencies
                - model_selections: Target model selections
            lambda_dag: Weight for DAG validity loss
            
        Returns:
            Dictionary containing:
                - total_loss: Total loss
                - ce_loss: Cross-entropy loss
                - dag_loss: DAG validity loss
        """
        # Cross-entropy loss for next sub-task prediction
        ce_loss = 0.0
        
        # Task type prediction loss
        if "task_types" in targets:
            pred_task_types = outputs["decomposition"]["task_types"]  # [batch_size, num_subtasks, num_task_types]
            target_task_types = targets["task_types"]  # [batch_size, num_subtasks]
            
            # Ensure shapes match
            if pred_task_types.size(0) != target_task_types.size(0) or pred_task_types.size(1) != target_task_types.size(1):
                # Handle shape mismatch by taking minimum
                min_batch = min(pred_task_types.size(0), target_task_types.size(0))
                min_subtasks = min(pred_task_types.size(1), target_task_types.size(1))
                pred_task_types = pred_task_types[:min_batch, :min_subtasks, :]
                target_task_types = target_task_types[:min_batch, :min_subtasks]
            
            # Flatten for loss computation
            pred_flat = pred_task_types.view(-1, pred_task_types.size(-1))  # [batch*num_subtasks, num_task_types]
            target_flat = target_task_types.view(-1)  # [batch*num_subtasks]
            
            # Mask out invalid targets (-1 values, which indicate padding or invalid)
            valid_mask = target_flat >= 0
            if valid_mask.any():
                pred_valid = pred_flat[valid_mask]
                target_valid = target_flat[valid_mask]
                ce_loss += nn.functional.cross_entropy(
                    pred_valid,
                    target_valid,
                )
        
        # Model selection loss
        if "model_selections" in targets:
            pred_probs = outputs["model_selection_probs"]  # [batch_size, num_subtasks, action_dim]
            target_selections = targets["model_selections"]  # [batch_size, num_subtasks]
            
            # Ensure shapes match
            if pred_probs.size(0) != target_selections.size(0) or pred_probs.size(1) != target_selections.size(1):
                # Handle shape mismatch by taking minimum
                min_batch = min(pred_probs.size(0), target_selections.size(0))
                min_subtasks = min(pred_probs.size(1), target_selections.size(1))
                pred_probs = pred_probs[:min_batch, :min_subtasks, :]
                target_selections = target_selections[:min_batch, :min_subtasks]
            
            # Flatten for loss computation
            pred_flat = pred_probs.view(-1, pred_probs.size(-1))  # [batch*num_subtasks, action_dim]
            target_flat = target_selections.view(-1)  # [batch*num_subtasks]
            
            # Mask out invalid targets (-1 values, which indicate padding or invalid)
            valid_mask = target_flat >= 0
            if valid_mask.any():
                pred_valid = pred_flat[valid_mask]
                target_valid = target_flat[valid_mask]
                ce_loss += nn.functional.cross_entropy(
                    pred_valid,
                    target_valid,
                )
        
        # DAG validity loss (topological ordering constraint)
        dag_loss = self._compute_dag_loss(outputs["workflow_graph"]["adjacency"])
        
        total_loss = ce_loss + lambda_dag * dag_loss
        
        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "dag_loss": dag_loss,
        }
    
    def _compute_dag_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Compute loss that encourages valid DAG structure.
        
        Args:
            adjacency: [batch_size, num_subtasks, num_subtasks] adjacency matrices
            
        Returns:
            Scalar DAG validity loss
        """
        batch_size = adjacency.size(0)
        dag_loss = 0.0
        
        for b in range(batch_size):
            adj = adjacency[b]
            num_nodes = adj.size(0)
            
            # Penalize self-loops
            self_loop_penalty = torch.diagonal(adj).sum()
            
            # Penalize cycles (simplified: penalize edges that violate topological order)
            # For a valid DAG, if there's an edge from i to j, then i < j in topological order
            cycle_penalty = 0.0
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    # If edge from j to i exists (j > i), this violates topological order
                    if adj[j, i] > 0.5:
                        cycle_penalty += adj[j, i]
            
            dag_loss += self_loop_penalty + cycle_penalty
        
        return dag_loss / batch_size

