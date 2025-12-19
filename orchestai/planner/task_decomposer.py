"""
Task Decomposer: Learned module for breaking complex tasks into sub-tasks with dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SubTask:
    """Represents a single sub-task in the decomposition"""
    id: int
    description: str
    task_type: str  # e.g., "text_extraction", "summarization", "visualization"
    dependencies: List[int]  # IDs of prerequisite sub-tasks
    estimated_complexity: float  # 0.0 to 1.0


class TaskDecomposer(nn.Module):
    """
    Learned module that decomposes complex tasks into sub-tasks with dependencies.
    Outputs a sequence of sub-tasks that can form a DAG.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_size: int = 512,
        num_layers: int = 3,
        max_subtasks: int = 20,
        num_task_types: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.max_subtasks = max_subtasks
        self.num_task_types = num_task_types
        
        # Encoder for instruction embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM for sequential sub-task generation
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Sub-task description decoder
        self.description_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),  # Output embedding for description
        )
        
        # Task type classifier
        self.task_type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_task_types),
        )
        
        # Dependency predictor (predicts which previous tasks this depends on)
        self.dependency_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, max_subtasks),  # Binary classification for each possible dependency
        )
        
        # Complexity estimator
        self.complexity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),  # Output between 0 and 1
        )
        
        # Stop token predictor (whether to continue generating sub-tasks)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        instruction_embedding: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose instruction into sub-tasks.
        
        Args:
            instruction_embedding: [batch_size, input_dim] instruction representation
            max_steps: Maximum number of sub-tasks to generate (default: self.max_subtasks)
            
        Returns:
            Dictionary containing:
                - subtask_embeddings: [batch_size, num_subtasks, hidden_size]
                - task_types: [batch_size, num_subtasks, num_task_types] logits
                - dependencies: [batch_size, num_subtasks, max_subtasks] dependency logits
                - complexities: [batch_size, num_subtasks, 1]
                - stop_probs: [batch_size, num_subtasks, 1]
        """
        batch_size = instruction_embedding.size(0)
        max_steps = max_steps or self.max_subtasks
        
        # Encode instruction
        encoded = self.encoder(instruction_embedding)  # [batch_size, hidden_size]
        
        # Initialize LSTM state
        h0 = encoded.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)  # [num_layers, batch_size, hidden_size]
        c0 = torch.zeros_like(h0)
        
        # Generate sub-tasks sequentially
        subtask_embeddings = []
        task_types = []
        dependencies = []
        complexities = []
        stop_probs = []
        
        # Use instruction as initial input
        lstm_input = encoded.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        for step in range(max_steps):
            # LSTM forward pass
            lstm_out, (h0, c0) = self.lstm(lstm_input, (h0, c0))
            hidden = lstm_out.squeeze(1)  # [batch_size, hidden_size]
            
            # Generate sub-task components
            subtask_emb = self.description_decoder(hidden)
            task_type_logits = self.task_type_head(hidden)
            dependency_logits = self.dependency_head(hidden)
            complexity = self.complexity_head(hidden)
            stop_prob = self.stop_head(hidden)
            
            subtask_embeddings.append(subtask_emb)
            task_types.append(task_type_logits)
            dependencies.append(dependency_logits)
            complexities.append(complexity)
            stop_probs.append(stop_prob)
            
            # Use current hidden state as next input
            lstm_input = hidden.unsqueeze(1)
        
        # Stack all outputs
        subtask_embeddings = torch.stack(subtask_embeddings, dim=1)  # [batch_size, max_steps, hidden_size]
        task_types = torch.stack(task_types, dim=1)  # [batch_size, max_steps, num_task_types]
        dependencies = torch.stack(dependencies, dim=1)  # [batch_size, max_steps, max_subtasks]
        complexities = torch.stack(complexities, dim=1)  # [batch_size, max_steps, 1]
        stop_probs = torch.stack(stop_probs, dim=1)  # [batch_size, max_steps, 1]
        
        return {
            "subtask_embeddings": subtask_embeddings,
            "task_types": task_types,
            "dependencies": dependencies,
            "complexities": complexities,
            "stop_probs": stop_probs,
        }
    
    def decode_subtasks(
        self,
        outputs: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[List[SubTask]]:
        """
        Decode model outputs into SubTask objects.
        
        Args:
            outputs: Output dictionary from forward()
            threshold: Threshold for dependency prediction
            
        Returns:
            List of SubTask lists (one per batch item)
        """
        batch_size = outputs["subtask_embeddings"].size(0)
        num_subtasks = outputs["subtask_embeddings"].size(1)
        
        # Get predictions
        task_type_probs = F.softmax(outputs["task_types"], dim=-1)
        dependency_probs = torch.sigmoid(outputs["dependencies"])
        complexities = outputs["complexities"].squeeze(-1)
        stop_probs = outputs["stop_probs"].squeeze(-1)
        
        all_subtasks = []
        
        for b in range(batch_size):
            subtasks = []
            for i in range(num_subtasks):
                # Check if we should stop (stop probability > threshold)
                if stop_probs[b, i].item() > threshold and i > 0:
                    break
                
                # Get task type
                task_type_idx = task_type_probs[b, i].argmax().item()
                
                # Get dependencies (previous tasks with probability > threshold)
                deps = [
                    j for j in range(i)
                    if dependency_probs[b, i, j].item() > threshold
                ]
                
                subtask = SubTask(
                    id=i,
                    description=f"subtask_{i}",  # Would need a decoder for actual descriptions
                    task_type=f"type_{task_type_idx}",
                    dependencies=deps,
                    estimated_complexity=complexities[b, i].item(),
                )
                subtasks.append(subtask)
            
            all_subtasks.append(subtasks)
        
        return all_subtasks

