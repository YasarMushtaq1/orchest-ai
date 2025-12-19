"""
Training script for Phase 1: Supervised Pre-Training
"""

import torch
import argparse
from torch.utils.data import DataLoader
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system
from orchestai.planner.planner_model import PlannerModel
from orchestai.training.supervised_trainer import SupervisedTrainer, WorkflowDataset


def create_dummy_dataset(num_samples: int = 100):
    """Create dummy dataset for testing"""
    instructions = [f"Task {i}: Generate a presentation about topic {i}" for i in range(num_samples)]
    subtask_sequences = [[{"id": j, "description": f"subtask_{j}"} for j in range(3)] for _ in range(num_samples)]
    task_types = [[0, 1, 2] for _ in range(num_samples)]
    dependencies = [[[0], [0, 1], [1, 2]] for _ in range(num_samples)]
    model_selections = [[0, 1, 2] for _ in range(num_samples)]
    
    return WorkflowDataset(
        instructions=instructions,
        subtask_sequences=subtask_sequences,
        task_types=task_types,
        dependencies=dependencies,
        model_selections=model_selections,
    )


def main():
    parser = argparse.ArgumentParser(description="Train OrchestAI Planner (Phase 1: Supervised)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    train_config = config["training"]["phase1_supervised"]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize planner
    planner_config = config["planner"]
    planner = PlannerModel(
        instruction_encoder_config=planner_config["instruction_encoder"],
        task_decomposer_config=planner_config["task_decomposer"],
        graph_generator_config=planner_config["workflow_graph_generator"],
        model_selector_config=planner_config["model_selector"],
    )
    
    # Create dataset
    if args.data:
        # Load from file (implement data loading)
        dataset = create_dummy_dataset()
    else:
        dataset = create_dummy_dataset(num_samples=train_config.get("train_data_size", 300))
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size or train_config.get("batch_size", 32),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size or train_config.get("batch_size", 32),
        shuffle=False,
    )
    
    # Initialize trainer
    trainer = SupervisedTrainer(
        planner=planner,
        config=train_config,
        device=device,
        use_wandb=args.use_wandb,
    )
    
    # Train
    print("Starting Phase 1: Supervised Pre-Training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs or train_config.get("num_epochs", 50),
    )
    
    print("Training completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history["val_loss"]:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()

