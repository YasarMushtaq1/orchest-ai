#!/usr/bin/env python3
"""
Test the best checkpoint from 1000 data training
Tests model performance and investigates DAG loss issue
"""

import torch
import sys
import os
import numpy as np
from typing import Dict, Any, List

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from orchestai.utils.config_loader import load_config
from orchestai.planner.planner_model import PlannerModel
from orchestai.training.supervised_trainer import WorkflowDataset
from torch.utils.data import DataLoader
import json


def load_checkpoint_info(checkpoint_path: str):
    """Load and display checkpoint information"""
    print("="*60)
    print("Checkpoint Information")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"Keys: {list(checkpoint.keys())}")
    print()
    
    return checkpoint


def load_model(checkpoint_path: str):
    """Load the trained model"""
    print("="*60)
    print("Loading Model")
    print("="*60)
    
    # Load config
    config = load_config("config.yaml")
    planner_config = config["planner"]
    
    # Create model
    planner = PlannerModel(
        instruction_encoder_config=planner_config["instruction_encoder"],
        task_decomposer_config=planner_config["task_decomposer"],
        graph_generator_config=planner_config["workflow_graph_generator"],
        model_selector_config=planner_config["model_selector"],
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    planner.load_state_dict(checkpoint["model_state_dict"])
    planner.eval()
    
    print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    print()
    
    return planner, checkpoint


def test_model_predictions(planner: PlannerModel, num_samples: int = 5):
    """Test model predictions on sample data"""
    print("="*60)
    print("Testing Model Predictions")
    print("="*60)
    
    # Load sample data
    with open("training_data.json", "r") as f:
        data = json.load(f)
    
    sample_data = data[:num_samples]
    
    results = {
        "task_decomposition": [],
        "dag_validity": [],
        "model_selection": [],
        "dag_losses": [],
    }
    
    for i, sample in enumerate(sample_data):
        instruction = sample["instruction"]
        
        # Forward pass
        with torch.no_grad():
            outputs = planner([instruction])
        
        # Check outputs
        num_subtasks = outputs["decomposition"]["subtask_embeddings"].size(1)
        adjacency = outputs["workflow_graph"]["adjacency"][0]  # First batch item
        
        # Compute DAG loss for this sample
        dag_loss = compute_dag_loss_single(adjacency)
        results["dag_losses"].append(dag_loss)
        
        # Check for cycles
        has_cycles = check_cycles(adjacency)
        results["dag_validity"].append(not has_cycles)
        
        # Model selections
        model_selections = outputs["model_selections"][0]
        results["model_selection"].append(len(model_selections))
        
        print(f"\nSample {i+1}: {instruction[:60]}...")
        print(f"  Subtasks generated: {num_subtasks}")
        print(f"  Model selections: {len(model_selections)}")
        print(f"  DAG loss: {dag_loss:.4f}")
        print(f"  Has cycles: {has_cycles}")
        print(f"  Valid DAG: {not has_cycles}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Valid DAGs: {sum(results['dag_validity'])}/{num_samples} ({sum(results['dag_validity'])/num_samples*100:.1f}%)")
    print(f"Average DAG loss: {np.mean(results['dag_losses']):.4f}")
    print(f"Min DAG loss: {np.min(results['dag_losses']):.4f}")
    print(f"Max DAG loss: {np.max(results['dag_losses']):.4f}")
    
    return results


def compute_dag_loss_single(adjacency: torch.Tensor) -> float:
    """Compute DAG loss for a single adjacency matrix"""
    num_nodes = adjacency.size(0)
    
    # Penalize self-loops
    self_loop_penalty = torch.diagonal(adjacency).sum().item()
    
    # Penalize cycles (edges that violate topological order)
    cycle_penalty = 0.0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # If edge from j to i exists (j > i), this violates topological order
            if adjacency[j, i].item() > 0.5:
                cycle_penalty += adjacency[j, i].item()
    
    return self_loop_penalty + cycle_penalty


def check_cycles(adjacency: torch.Tensor) -> bool:
    """Check if adjacency matrix has cycles using DFS"""
    num_nodes = adjacency.size(0)
    adj_list = {}
    
    # Build adjacency list
    for i in range(num_nodes):
        adj_list[i] = []
        for j in range(num_nodes):
            if adjacency[i, j].item() > 0.5:
                adj_list[i].append(j)
    
    # DFS to detect cycles
    visited = [False] * num_nodes
    rec_stack = [False] * num_nodes
    
    def has_cycle(node):
        visited[node] = True
        rec_stack[node] = True
        
        for neighbor in adj_list.get(node, []):
            if not visited[neighbor]:
                if has_cycle(neighbor):
                    return True
            elif rec_stack[neighbor]:
                return True
        
        rec_stack[node] = False
        return False
    
    for node in range(num_nodes):
        if not visited[node]:
            if has_cycle(node):
                return True
    
    return False


def investigate_dag_loss(planner: PlannerModel):
    """Investigate DAG loss issue in detail"""
    print("="*60)
    print("Investigating DAG Loss Issue")
    print("="*60)
    
    # Load validation data
    with open("training_data.json", "r") as f:
        data = json.load(f)
    
    # Use validation set (last 200 samples)
    val_data = data[800:1000]
    
    print(f"Testing on {len(val_data)} validation samples...")
    
    dag_losses = []
    cycle_counts = []
    self_loop_counts = []
    edge_counts = []
    
    for i, sample in enumerate(val_data[:20]):  # Test first 20
        instruction = sample["instruction"]
        
        with torch.no_grad():
            outputs = planner([instruction])
        
        adjacency = outputs["workflow_graph"]["adjacency"][0]
        num_nodes = adjacency.size(0)
        
        # Count edges
        edges = (adjacency > 0.5).sum().item()
        edge_counts.append(edges)
        
        # Count self-loops
        self_loops = torch.diagonal(adjacency > 0.5).sum().item()
        self_loop_counts.append(self_loops)
        
        # Count cycles (backward edges)
        cycles = 0
        for i_node in range(num_nodes):
            for j_node in range(i_node + 1, num_nodes):
                if adjacency[j_node, i_node].item() > 0.5:
                    cycles += 1
        cycle_counts.append(cycles)
        
        # Compute DAG loss
        dag_loss = compute_dag_loss_single(adjacency)
        dag_losses.append(dag_loss)
    
    # Statistics
    print("\n" + "="*60)
    print("DAG Loss Analysis")
    print("="*60)
    print(f"Average DAG loss: {np.mean(dag_losses):.4f}")
    print(f"Min DAG loss: {np.min(dag_losses):.4f}")
    print(f"Max DAG loss: {np.max(dag_losses):.4f}")
    print(f"Std DAG loss: {np.std(dag_losses):.4f}")
    print()
    print(f"Average edges per graph: {np.mean(edge_counts):.2f}")
    print(f"Average self-loops: {np.mean(self_loop_counts):.2f}")
    print(f"Average backward edges (cycles): {np.mean(cycle_counts):.2f}")
    print()
    print(f"Graphs with cycles: {sum(1 for c in cycle_counts if c > 0)}/{len(cycle_counts)}")
    print(f"Graphs with self-loops: {sum(1 for s in self_loop_counts if s > 0)}/{len(self_loop_counts)}")
    
    # Check adjacency matrix values
    print("\n" + "="*60)
    print("Adjacency Matrix Value Analysis")
    print("="*60)
    
    sample_adj = outputs["workflow_graph"]["adjacency"][0]
    print(f"Adjacency matrix shape: {sample_adj.shape}")
    print(f"Min value: {sample_adj.min().item():.4f}")
    print(f"Max value: {sample_adj.max().item():.4f}")
    print(f"Mean value: {sample_adj.mean().item():.4f}")
    print(f"Values > 0.5: {(sample_adj > 0.5).sum().item()}")
    print(f"Values > 0.1: {(sample_adj > 0.1).sum().item()}")
    print(f"Values > 0.01: {(sample_adj > 0.01).sum().item()}")
    
    return {
        "dag_losses": dag_losses,
        "cycle_counts": cycle_counts,
        "self_loop_counts": self_loop_counts,
        "edge_counts": edge_counts,
    }


def compare_with_178_model():
    """Compare with previous 178 data model if available"""
    print("="*60)
    print("Comparison with 178 Data Model")
    print("="*60)
    
    # Check if old checkpoint exists
    old_checkpoint = "checkpoints/phase1_best_model.pth"
    new_checkpoint = "checkpoint_best.pth"
    
    if os.path.exists(old_checkpoint):
        print("Found old checkpoint (178 data model)")
        old_ckpt = torch.load(old_checkpoint, map_location="cpu")
        print(f"  Epoch: {old_ckpt['epoch']}")
        print(f"  Val Loss: {old_ckpt['val_loss']:.4f}")
    
    if os.path.exists(new_checkpoint):
        print("\nNew checkpoint (1000 data model)")
        new_ckpt = torch.load(new_checkpoint, map_location="cpu")
        print(f"  Epoch: {new_ckpt['epoch']}")
        print(f"  Val Loss: {new_ckpt['val_loss']:.4f}")
        print(f"  Improvement: {((old_ckpt['val_loss'] - new_ckpt['val_loss']) / old_ckpt['val_loss'] * 100):.1f}%")
    else:
        print("New checkpoint not found")


def main():
    """Main test function"""
    checkpoint_path = "checkpoint_best.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load checkpoint info
    checkpoint = load_checkpoint_info(checkpoint_path)
    
    # Load model
    planner, _ = load_model(checkpoint_path)
    
    # Test predictions
    test_results = test_model_predictions(planner, num_samples=5)
    
    # Investigate DAG loss
    dag_analysis = investigate_dag_loss(planner)
    
    # Compare with old model
    compare_with_178_model()
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)


if __name__ == "__main__":
    main()

