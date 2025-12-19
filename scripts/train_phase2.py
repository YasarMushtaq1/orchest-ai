"""
Training script for Phase 2: Reinforcement Learning Fine-Tuning
"""

import torch
import argparse
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system
from orchestai.training.rl_trainer import RLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train OrchestAI Planner (Phase 2: RL)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Phase 1 checkpoint")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    rl_config = config["training"]["phase2_rl"]
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup system
    orchestrator = setup_system(config, device=device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create task pool
    task_pool = [
        "Generate a presentation about artificial intelligence",
        "Analyze this dataset and create visualizations",
        "Summarize this research paper",
        "Create a document from these notes",
        "Process this image and generate a description",
    ]
    
    # Initialize RL trainer
    trainer = RLTrainer(
        planner=orchestrator.planner,
        orchestrator=orchestrator,
        config=rl_config,
        device=device,
        use_wandb=args.use_wandb,
    )
    
    # Train
    print("Starting Phase 2: Reinforcement Learning Fine-Tuning...")
    history = trainer.train(
        task_pool=task_pool,
        num_episodes=args.episodes or rl_config.get("num_episodes", 1000),
        max_steps_per_episode=rl_config.get("max_steps_per_episode", 50),
    )
    
    print("Training completed!")
    print(f"Average reward: {sum(history['episode_rewards'][-100:]) / 100:.2f}")
    print(f"Success rate: {sum(history['episode_success'][-100:]) / 100:.2f}")


if __name__ == "__main__":
    main()

