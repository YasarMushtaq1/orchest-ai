"""
RL-Based Model Selector: Policy network for optimizing model routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class RLModelSelector(nn.Module):
    """
    Reinforcement Learning-based model selector that optimizes routing
    of sub-tasks to worker models based on success probability and efficiency.
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 8,  # number of available worker models
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network (actor)
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_network = nn.Sequential(*layers)
        
        # Value network (critic) for PPO
        value_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_network = nn.Sequential(*value_layers)
        
    def forward(
        self,
        state: torch.Tensor,
        return_value: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy and value networks.
        
        Args:
            state: [batch_size, state_dim] state representation
            return_value: Whether to compute value estimate
            
        Returns:
            Dictionary containing:
                - action_logits: [batch_size, action_dim] unnormalized logits
                - action_probs: [batch_size, action_dim] action probabilities
                - value: [batch_size, 1] state value estimate (if return_value=True)
        """
        # Policy network
        action_logits = self.policy_network(state)  # [batch_size, action_dim]
        action_probs = F.softmax(action_logits, dim=-1)
        
        result = {
            "action_logits": action_logits,
            "action_probs": action_probs,
        }
        
        if return_value:
            # Value network
            value = self.value_network(state)  # [batch_size, 1]
            result["value"] = value
        
        return result
    
    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using policy network.
        
        Args:
            state: [batch_size, state_dim] or [state_dim] state representation
            deterministic: If True, select argmax; if False, sample from distribution
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Tuple of:
                - action: [batch_size] or scalar selected action indices
                - log_prob: [batch_size] or scalar log probabilities
                - entropy: [batch_size] or scalar action distribution entropy
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Get action distribution
        outputs = self.forward(state, return_value=False)
        action_probs = outputs["action_probs"]
        action_logits = outputs["action_logits"]
        
        # Create distribution
        dist = torch.distributions.Categorical(logits=action_logits / temperature)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
            log_prob = dist.log_prob(action)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # Compute entropy
        entropy = dist.entropy()
        
        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
        
        return action, log_prob, entropy
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss.
        
        Args:
            states: [batch_size, state_dim] state representations
            actions: [batch_size] selected actions
            old_log_probs: [batch_size] log probabilities from old policy
            advantages: [batch_size] advantage estimates
            returns: [batch_size] discounted returns
            clip_epsilon: PPO clipping parameter
            value_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus
            
        Returns:
            Dictionary containing:
                - total_loss: Scalar total loss
                - policy_loss: Scalar policy loss
                - value_loss: Scalar value loss
                - entropy: Scalar entropy
        """
        # Get current policy outputs
        outputs = self.forward(states, return_value=True)
        action_probs = outputs["action_probs"]
        values = outputs["value"].squeeze(-1)  # [batch_size]
        
        # Create distribution
        dist = torch.distributions.Categorical(logits=outputs["action_logits"])
        
        # Compute new log probabilities
        new_log_probs = dist.log_prob(actions)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

