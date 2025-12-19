"""
Reinforcement Learning Fine-Tuning (Phase 2) using PPO
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from tqdm import tqdm
import wandb


class RLTrainer:
    """
    Trainer for Phase 2: Reinforcement Learning Fine-Tuning using PPO
    """
    
    def __init__(
        self,
        planner: nn.Module,
        orchestrator: Any,  # OrchestrationSystem
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
    ):
        """
        Initialize RL trainer.
        
        Args:
            planner: PlannerModel instance
            orchestrator: OrchestrationSystem instance
            config: Training configuration dictionary
            device: PyTorch device
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.planner = planner
        self.orchestrator = orchestrator
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.planner.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.planner.parameters(),
            lr=config.get("learning_rate", 3e-5),
            weight_decay=1e-5,
        )
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.2)
        self.alpha = config.get("alpha", 10.0)  # success reward weight
        self.beta = config.get("beta", 0.1)  # cost penalty weight
        self.gamma_latency = config.get("gamma_latency", 0.01)  # latency penalty weight
        self.shaped_reward = config.get("shaped_reward", 0.1)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="orchestai-rl", config=config)
    
    def train(
        self,
        task_pool: List[str],
        num_episodes: int = 1000,
        max_steps_per_episode: int = 50,
        update_frequency: int = 10,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train using PPO.
        
        Args:
            task_pool: List of task instructions for training
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            update_frequency: Update policy every N episodes
            batch_size: Batch size for policy updates
            
        Returns:
            Dictionary with training history
        """
        history = {
            "episode_rewards": [],
            "episode_success": [],
            "episode_cost": [],
            "episode_latency": [],
            "policy_loss": [],
            "value_loss": [],
        }
        
        for episode in tqdm(range(num_episodes), desc="RL Training"):
            # Sample task
            task = np.random.choice(task_pool)
            
            # Run episode
            episode_data = self._run_episode(task, max_steps_per_episode)
            
            # Store experience
            self.experience_buffer.extend(episode_data["experiences"])
            
            # Record metrics
            history["episode_rewards"].append(episode_data["total_reward"])
            history["episode_success"].append(1.0 if episode_data["success"] else 0.0)
            history["episode_cost"].append(episode_data["total_cost"])
            history["episode_latency"].append(episode_data["total_latency"])
            
            # Update policy
            if len(self.experience_buffer) >= batch_size and episode % update_frequency == 0:
                update_metrics = self._update_policy(batch_size)
                history["policy_loss"].append(update_metrics["policy_loss"])
                history["value_loss"].append(update_metrics["value_loss"])
            
            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(history["episode_rewards"][-10:])
                avg_success = np.mean(history["episode_success"][-10:])
                print(f"Episode {episode}: Reward={avg_reward:.2f}, Success={avg_success:.2f}")
                
                if self.use_wandb:
                    wandb.log({
                        "episode": episode,
                        "avg_reward": avg_reward,
                        "avg_success": avg_success,
                        "avg_cost": np.mean(history["episode_cost"][-10:]),
                        "avg_latency": np.mean(history["episode_latency"][-10:]),
                    })
        
        return history
    
    def _run_episode(
        self,
        instruction: str,
        max_steps: int,
    ) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            instruction: Task instruction
            max_steps: Maximum steps
            
        Returns:
            Episode data including experiences and metrics
        """
        self.planner.eval()
        
        # Generate plan
        with torch.no_grad():
            planner_outputs = self.planner([instruction])
        
        # Execute workflow
        execution_result = self.orchestrator.execute(instruction)
        
        # Compute reward
        reward = self._compute_reward(execution_result)
        
        # Create experiences (simplified - in practice, would track per-step)
        experiences = []
        
        # Extract states, actions, rewards from planner outputs
        model_selections = planner_outputs["model_selections"][0]
        model_probs = planner_outputs["model_selection_probs"][0]
        
        for i, (action, prob) in enumerate(zip(model_selections, model_probs)):
            # Get state (instruction embedding + node embedding)
            inst_emb = planner_outputs["instruction_embeddings"][0]
            node_emb = planner_outputs["workflow_graph"]["node_embeddings"][0, i]
            state = torch.cat([inst_emb, node_emb])
            
            # Compute log probability
            log_prob = torch.log(prob[action] + 1e-8)
            
            experiences.append({
                "state": state,
                "action": action,
                "log_prob": log_prob,
                "reward": reward,  # Shared reward for all actions in episode
            })
        
        return {
            "experiences": experiences,
            "total_reward": reward,
            "success": execution_result.success,
            "total_cost": execution_result.total_cost,
            "total_latency": execution_result.total_latency_ms,
        }
    
    def _compute_reward(self, execution_result: Any) -> float:
        """
        Compute reward based on execution result.
        
        Args:
            execution_result: ExecutionResult from orchestrator
            
        Returns:
            Scalar reward value
        """
        # Success reward
        success_reward = self.alpha * (1.0 if execution_result.success else 0.0)
        
        # Cost penalty
        cost_penalty = self.beta * execution_result.total_cost
        
        # Latency penalty
        latency_penalty = self.gamma_latency * execution_result.total_latency_ms
        
        # Shaped rewards (per successful sub-task)
        shaped = 0.0
        if execution_result.success and execution_result.outputs:
            num_subtasks = len(execution_result.outputs)
            shaped = self.shaped_reward * num_subtasks
        
        total_reward = success_reward - cost_penalty - latency_penalty + shaped
        
        return total_reward
    
    def _update_policy(self, batch_size: int) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary with loss metrics
        """
        if len(self.experience_buffer) < batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        
        # Sample batch
        batch = np.random.choice(self.experience_buffer, size=batch_size, replace=False)
        
        # Prepare data
        states = torch.stack([exp["state"] for exp in batch]).to(self.device)
        actions = torch.tensor([exp["action"] for exp in batch], dtype=torch.long).to(self.device)
        old_log_probs = torch.stack([exp["log_prob"] for exp in batch]).to(self.device)
        rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float).to(self.device)
        
        # Compute advantages (simplified - in practice would use GAE)
        returns = rewards
        with torch.no_grad():
            outputs = self.planner.model_selector(states, return_value=True)
            values = outputs["value"].squeeze(-1)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute PPO loss
        loss_dict = self.planner.model_selector.compute_loss(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            clip_epsilon=self.epsilon,
        )
        
        # Update
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.planner.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "policy_loss": loss_dict["policy_loss"].item(),
            "value_loss": loss_dict["value_loss"].item(),
        }

