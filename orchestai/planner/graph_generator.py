"""
Workflow Graph Generator: GNN-based module for constructing DAGs representing execution plans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import networkx as nx


class WorkflowGraphGenerator(nn.Module):
    """
    Graph Neural Network module that constructs DAGs representing execution plans
    from decomposed sub-tasks.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = "GCN",  # GCN, GAT, or GraphSAGE
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        current_dim = hidden_dim
        for i in range(num_layers):
            if gnn_type == "GCN":
                self.gnn_layers.append(
                    GCNConv(
                        current_dim,
                        hidden_dim,
                    )
                )
                current_dim = hidden_dim
            elif gnn_type == "GAT":
                num_heads = 4
                is_last = (i == num_layers - 1)
                self.gnn_layers.append(
                    GATConv(
                        current_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                        concat=not is_last,
                    )
                )
                # Update current_dim based on whether heads are concatenated
                current_dim = hidden_dim * num_heads if not is_last else hidden_dim
            elif gnn_type == "GraphSAGE":
                self.gnn_layers.append(
                    SAGEConv(
                        current_dim,
                        hidden_dim,
                    )
                )
                current_dim = hidden_dim
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Output projection (use current_dim which accounts for GAT multi-head)
        self.output_proj = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Edge predictor (predicts if there should be an edge between two nodes)
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        subtask_embeddings: torch.Tensor,
        dependencies: torch.Tensor,
        return_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate workflow graph from sub-task embeddings and dependencies.
        
        Args:
            subtask_embeddings: [batch_size, num_subtasks, input_dim]
            dependencies: [batch_size, num_subtasks, max_subtasks] dependency logits
            return_graph: Whether to return NetworkX graph objects
            
        Returns:
            Dictionary containing:
                - node_embeddings: [batch_size, num_subtasks, output_dim]
                - edge_probs: [batch_size, num_subtasks, num_subtasks]
                - adjacency: [batch_size, num_subtasks, num_subtasks] binary adjacency
                - graphs: Optional list of NetworkX graphs if return_graph=True
        """
        batch_size, num_subtasks, _ = subtask_embeddings.shape
        
        # Project input embeddings
        node_features = self.input_proj(subtask_embeddings)  # [batch_size, num_subtasks, hidden_dim]
        
        all_node_embeddings = []
        all_edge_probs = []
        all_adjacencies = []
        graphs = [] if return_graph else None
        
        for b in range(batch_size):
            # Create initial graph structure from dependencies
            # Convert dependency logits to binary edges
            dep_probs = torch.sigmoid(dependencies[b])  # [num_subtasks, max_subtasks]
            edge_index = []
            
            # Build edge list from dependencies
            for i in range(num_subtasks):
                for j in range(min(i, num_subtasks)):
                    # Edge from j to i if j is a dependency of i
                    if dep_probs[i, j].item() > 0.5:
                        edge_index.append([j, i])
            
            # If no edges, create a simple chain to ensure connectivity
            if len(edge_index) == 0:
                edge_index = [[i, i+1] for i in range(num_subtasks - 1)]
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=node_features[b],  # [num_subtasks, hidden_dim]
                edge_index=edge_index,
            )
            
            # Apply GNN layers
            x = data.x
            for i, gnn_layer in enumerate(self.gnn_layers):
                x = gnn_layer(x, data.edge_index)
                if i < len(self.gnn_layers) - 1:
                    x = F.relu(x)
                    x = self.dropout(x)
            
            # Project to output dimension
            node_emb = self.output_proj(x)  # [num_subtasks, output_dim]
            
            # Predict edge probabilities for all pairs
            edge_probs = torch.zeros(num_subtasks, num_subtasks)
            for i in range(num_subtasks):
                for j in range(num_subtasks):
                    if i != j:
                        # Concatenate node embeddings
                        pair_emb = torch.cat([node_emb[i], node_emb[j]])
                        edge_probs[i, j] = self.edge_predictor(pair_emb).squeeze()
            
            # Create adjacency matrix (ensure DAG: no cycles)
            adjacency = self._create_dag_adjacency(edge_probs, dependencies[b])
            
            all_node_embeddings.append(node_emb)
            all_edge_probs.append(edge_probs)
            all_adjacencies.append(adjacency)
            
            if return_graph:
                # Create NetworkX graph
                G = nx.DiGraph()
                for i in range(num_subtasks):
                    G.add_node(i, embedding=node_emb[i].detach().cpu().numpy())
                for i in range(num_subtasks):
                    for j in range(num_subtasks):
                        if adjacency[i, j].item() > 0.5:
                            G.add_edge(i, j, weight=edge_probs[i, j].item())
                graphs.append(G)
        
        # Stack batch outputs
        node_embeddings = torch.stack(all_node_embeddings, dim=0)  # [batch_size, num_subtasks, output_dim]
        edge_probs = torch.stack(all_edge_probs, dim=0)  # [batch_size, num_subtasks, num_subtasks]
        adjacencies = torch.stack(all_adjacencies, dim=0)  # [batch_size, num_subtasks, num_subtasks]
        
        result = {
            "node_embeddings": node_embeddings,
            "edge_probs": edge_probs,
            "adjacency": adjacencies,
        }
        
        if return_graph:
            result["graphs"] = graphs
        
        return result
    
    def _create_dag_adjacency(
        self,
        edge_probs: torch.Tensor,
        dependencies: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a valid DAG adjacency matrix ensuring topological ordering.
        
        Args:
            edge_probs: [num_subtasks, num_subtasks] edge probabilities
            dependencies: [num_subtasks, max_subtasks] dependency logits
            
        Returns:
            [num_subtasks, num_subtasks] binary adjacency matrix (DAG)
        """
        num_subtasks = edge_probs.size(0)
        adjacency = torch.zeros_like(edge_probs)
        
        # Use dependencies to create initial edges (respecting topological order)
        dep_probs = torch.sigmoid(dependencies)
        for i in range(num_subtasks):
            for j in range(min(i, num_subtasks)):
                if dep_probs[i, j].item() > 0.5:
                    # Edge from j to i (j must complete before i)
                    adjacency[j, i] = 1.0
        
        # Add additional high-probability edges if they don't create cycles
        for i in range(num_subtasks):
            for j in range(num_subtasks):
                if i != j and adjacency[i, j] == 0 and edge_probs[i, j].item() > 0.7:
                    # Check if adding this edge would create a cycle
                    test_adj = adjacency.clone()
                    test_adj[i, j] = 1.0
                    if self._is_dag(test_adj):
                        adjacency[i, j] = 1.0
        
        return adjacency
    
    def _is_dag(self, adjacency: torch.Tensor) -> bool:
        """
        Check if adjacency matrix represents a valid DAG (no cycles).
        
        Args:
            adjacency: [num_subtasks, num_subtasks] adjacency matrix
            
        Returns:
            True if DAG, False if contains cycles
        """
        num_subtasks = adjacency.size(0)
        visited = [False] * num_subtasks
        rec_stack = [False] * num_subtasks
        
        def has_cycle(node):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in range(num_subtasks):
                if adjacency[node, neighbor].item() > 0.5:
                    if not visited[neighbor]:
                        if has_cycle(neighbor):
                            return True
                    elif rec_stack[neighbor]:
                        return True
            
            rec_stack[node] = False
            return False
        
        for node in range(num_subtasks):
            if not visited[node]:
                if has_cycle(node):
                    return False
        
        return True

