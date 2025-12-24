"""
Supervised Pre-Training (Phase 1) for Planner Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WorkflowDataset(Dataset):
    """
    Dataset for supervised training of Planner Model.
    Contains expert-annotated workflows.
    """
    
    def __init__(
        self,
        instructions: List[str],
        subtask_sequences: List[List[Dict[str, Any]]],
        task_types: List[List[int]],
        dependencies: List[List[List[int]]],
        model_selections: List[List[int]],
    ):
        """
        Initialize dataset.
        
        Args:
            instructions: List of instruction strings
            subtask_sequences: List of sub-task sequences (each is list of sub-task dicts)
            task_types: List of task type labels for each sub-task
            dependencies: List of dependency lists for each sub-task
            model_selections: List of model selection labels for each sub-task
        """
        self.instructions = instructions
        self.subtask_sequences = subtask_sequences
        self.task_types = task_types
        self.dependencies = dependencies
        self.model_selections = model_selections
        
        assert len(instructions) == len(subtask_sequences) == len(task_types) == len(dependencies) == len(model_selections)
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        return {
            "instruction": self.instructions[idx],
            "subtask_sequence": self.subtask_sequences[idx],
            "task_types": torch.tensor(self.task_types[idx], dtype=torch.long),
            "dependencies": torch.tensor(self.dependencies[idx], dtype=torch.float),
            "model_selections": torch.tensor(self.model_selections[idx], dtype=torch.long),
        }


class SupervisedTrainer:
    """
    Trainer for Phase 1: Supervised Pre-Training
    """
    
    def __init__(
        self,
        planner: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
    ):
        """
        Initialize supervised trainer.
        
        Args:
            planner: PlannerModel instance
            config: Training configuration dictionary
            device: PyTorch device
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.planner = planner
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.planner.to(self.device)
        
        # Optimizer
        learning_rate = float(config.get("learning_rate", 1e-4))
        self.optimizer = optim.AdamW(
            self.planner.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(project="orchestai", config=config)
        elif use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not installed. Continuing without wandb logging.")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Train the planner model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary with training history
        """
        history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_dag_loss": [],
            "val_loss": [],
        }
        
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["train_ce_loss"].append(train_metrics["ce_loss"])
            history["train_dag_loss"].append(train_metrics["dag_loss"])
            
            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                
                # Learning rate scheduling
                self.scheduler.step(val_metrics["loss"])
                
                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    self._save_checkpoint(epoch, val_metrics["loss"], is_best=True)
                
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "val_loss": val_metrics["loss"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    })
            else:
                self.scheduler.step(train_metrics["loss"])
                if self.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    })
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.planner.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_dag_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Move batch to device
            instructions = batch["instruction"]
            
            # Forward pass
            outputs = self.planner(instructions)
            
            # Prepare targets
            targets = {
                "task_types": batch["task_types"].to(self.device),
                "model_selections": batch["model_selections"].to(self.device),
            }
            
            # Compute loss
            loss_dict = self.planner.compute_supervised_loss(
                outputs,
                targets,
                lambda_dag=self.config.get("lambda_dag", 0.3),
            )
            
            loss = loss_dict["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.planner.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_ce_loss += loss_dict["ce_loss"].item()
            total_dag_loss += loss_dict["dag_loss"].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": loss.item(),
                "ce": loss_dict["ce_loss"].item(),
                "dag": loss_dict["dag_loss"].item(),
            })
        
        return {
            "loss": total_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "dag_loss": total_dag_loss / num_batches,
        }
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.planner.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                instructions = batch["instruction"]
                outputs = self.planner(instructions)
                
                targets = {
                    "task_types": batch["task_types"].to(self.device),
                    "model_selections": batch["model_selections"].to(self.device),
                }
                
                loss_dict = self.planner.compute_supervised_loss(
                    outputs,
                    targets,
                    lambda_dag=self.config.get("lambda_dag", 0.3),
                )
                
                total_loss += loss_dict["total_loss"].item()
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.planner.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        
        filename = "checkpoint_best.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)
        
        if is_best:
            print(f"Saved best model with val_loss={val_loss:.4f}")

