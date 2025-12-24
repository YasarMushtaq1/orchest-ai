# Configuration Guide

This document explains how to configure OrchestAI for your use case.

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Planner Configuration](#planner-configuration)
3. [Worker Configuration](#worker-configuration)
4. [Training Configuration](#training-configuration)
5. [System Configuration](#system-configuration)
6. [Environment Variables](#environment-variables)

---

## Configuration File Structure

OrchestAI uses YAML files for configuration. The main configuration file is `config.yaml`.

**Example:**
```yaml
# config.yaml
planner:
  instruction_encoder: {...}
  task_decomposer: {...}
  workflow_graph_generator: {...}
  model_selector: {...}

worker_models: [...]
training: {...}
evaluation: {...}
system: {...}
```

---

## Planner Configuration

### Instruction Encoder

**Purpose:** Configures the instruction encoding component.

**Options:**
```yaml
instruction_encoder:
  model_name: "t5-base"        # or "bert-base-uncased"
  hidden_size: 768             # 768 for BERT, 512 for T5
  max_length: 512              # Maximum token length
```

**Choices:**
- **t5-base**: Better for generation tasks, 512 hidden size
- **bert-base-uncased**: Better for understanding, 768 hidden size

### Task Decomposer

**Purpose:** Configures task decomposition component.

**Options:**
```yaml
task_decomposer:
  hidden_size: 512             # Hidden dimension
  num_layers: 3                # LSTM layers
  dropout: 0.1                 # Dropout rate
  max_subtasks: 20             # Maximum subtasks
  num_task_types: 10           # Number of task type classes
```

**Recommendations:**
- **hidden_size**: 512-768 (match instruction encoder)
- **num_layers**: 2-4 (more layers = more capacity, but slower)
- **dropout**: 0.1-0.3 (prevent overfitting)

### Workflow Graph Generator

**Purpose:** Configures graph generation component.

**Options:**
```yaml
workflow_graph_generator:
  gnn_type: "GCN"              # "GCN", "GAT", or "GraphSAGE"
  num_layers: 3                # GNN layers
  hidden_dim: 256              # Hidden dimension
  output_dim: 128              # Output embedding dimension
  dropout: 0.1                 # Dropout rate
```

**GNN Types:**
- **GCN**: Standard graph convolution, fastest
- **GAT**: Attention-based, better for complex dependencies
- **GraphSAGE**: Sampling-based, good for large graphs

**Recommendations:**
- **GAT** for complex workflows with many dependencies
- **GCN** for simpler workflows (faster)
- **GraphSAGE** for very large graphs

### Model Selector

**Purpose:** Configures RL-based model selector.

**Options:**
```yaml
model_selector:
  state_dim: 512               # State dimension (instruction + node embedding)
  action_dim: 8                # Number of worker models
  hidden_dims: [256, 128]      # Hidden layer dimensions
  dropout: 0.1                 # Dropout rate
```

**Note:** `state_dim` should be `instruction_encoder.hidden_size + workflow_graph_generator.output_dim`

---

## Worker Configuration

### Worker Models

**Purpose:** Defines available worker models.

**Format:**
```yaml
worker_models:
  - name: "gpt-4"
    model_type: "llm"
    cost_per_token: 0.03
    latency_ms: 500
    
  - name: "gpt-3.5-turbo"
    model_type: "llm"
    cost_per_token: 0.002
    latency_ms: 200
```

**Required Fields:**
- **name**: Unique worker identifier
- **model_type**: "llm", "vision", "audio", "code", "document", "data"
- **cost_per_token**: Cost per token (for cost optimization)
- **latency_ms**: Expected latency in milliseconds

**Model Types:**
- **llm**: Language models (GPT, LLaMA, etc.)
- **vision**: Vision models (CLIP, etc.)
- **audio**: Audio models (Whisper, etc.)
- **code**: Code models (CodeLlama, etc.)
- **document**: Document processing models
- **data**: Data analysis models

**Cost and Latency:**
- Used by RL model selector for optimization
- Should reflect actual API costs and latencies
- Can be estimated from historical data

---

## Training Configuration

### Phase 1: Supervised Pre-Training

**Purpose:** Configures supervised learning phase.

**Options:**
```yaml
training:
  phase1_supervised:
    batch_size: 32             # Training batch size
    learning_rate: 1e-4        # Learning rate
    num_epochs: 50             # Number of epochs
    lambda_dag: 0.3            # DAG validity loss weight
    train_data_size: 300       # Training examples
    synthetic_augmentations: 1000  # Synthetic data augmentations
```

**Recommendations:**
- **batch_size**: 16-64 (larger = more stable, but slower)
- **learning_rate**: 1e-4 to 1e-3 (start high, reduce if unstable)
- **lambda_dag**: 0.1-0.5 (higher = more emphasis on valid DAGs)

### Phase 2: RL Fine-Tuning

**Purpose:** Configures reinforcement learning phase.

**Options:**
```yaml
training:
  phase2_rl:
    algorithm: "PPO"           # RL algorithm
    learning_rate: 3e-5        # Learning rate (lower than Phase 1)
    num_episodes: 1000         # Number of training episodes
    max_steps_per_episode: 50   # Maximum steps per episode
    gamma: 0.99                # Discount factor
    epsilon: 0.2               # PPO clipping epsilon
    alpha: 10.0                # Success reward weight
    beta: 0.1                  # Cost penalty weight
    gamma_latency: 0.01        # Latency penalty weight
    shaped_reward: 0.1         # Reward per successful subtask
```

**Reward Weights:**
- **alpha**: Higher = more emphasis on success (default: 10.0)
- **beta**: Higher = more emphasis on cost (default: 0.1)
- **gamma_latency**: Higher = more emphasis on latency (default: 0.01)
- **shaped_reward**: Encourages partial success (default: 0.1)

**Recommendations:**
- **learning_rate**: 1e-5 to 1e-4 (lower than Phase 1)
- **gamma**: 0.95-0.99 (higher = more long-term thinking)
- **epsilon**: 0.1-0.3 (PPO clipping parameter)

### Phase 3: Continuous Learning

**Purpose:** Configures online learning from execution.

**Options:**
```yaml
training:
  phase3_continuous:
    online_learning: true       # Enable online learning
    update_frequency: 100       # Update every N executions
    learning_rate: 1e-5         # Learning rate (very low)
```

---

## System Configuration

**Purpose:** Configures system-level settings.

**Options:**
```yaml
system:
  max_workflow_depth: 10       # Maximum workflow depth
  max_parallel_tasks: 5         # Maximum parallel tasks
  timeout_seconds: 300          # Execution timeout
  output_schema: "json"         # Output format
  log_level: "INFO"            # Logging level
```

**Recommendations:**
- **max_workflow_depth**: 5-20 (higher = more complex workflows)
- **max_parallel_tasks**: 3-10 (higher = faster but more resource usage)
- **timeout_seconds**: 60-600 (depends on task complexity)

---

## Environment Variables

### API Keys

**OpenAI API Key:**
```bash
# .env file
OPENAI_API_KEY=sk-proj-...
```

**HuggingFace Token (optional):**
```bash
# .env file
HUGGINGFACE_TOKEN=hf_...
```

### Loading Environment Variables

**Using python-dotenv:**
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file

import os
api_key = os.getenv("OPENAI_API_KEY")
```

---

## Configuration Loading

### Loading Configuration

**From File:**
```python
from orchestai.utils.config_loader import load_config

config = load_config("config.yaml")
```

**Programmatic:**
```python
config = {
    "planner": {...},
    "worker_models": [...],
    ...
}
```

### System Setup

**Using Configuration:**
```python
from orchestai.utils.setup import setup_system

orchestrator = setup_system(config)
```

---

## Configuration Examples

### Minimal Configuration

```yaml
planner:
  instruction_encoder:
    model_name: "t5-base"
    hidden_size: 512
    max_length: 512
  task_decomposer:
    hidden_size: 512
    num_layers: 2
    dropout: 0.1
  workflow_graph_generator:
    gnn_type: "GCN"
    num_layers: 2
    hidden_dim: 256
    output_dim: 128
    dropout: 0.1
  model_selector:
    state_dim: 640  # 512 + 128
    action_dim: 3
    hidden_dims: [128]
    dropout: 0.1

worker_models:
  - name: "gpt-3.5-turbo"
    model_type: "llm"
    cost_per_token: 0.002
    latency_ms: 200

system:
  max_workflow_depth: 5
  max_parallel_tasks: 3
  timeout_seconds: 60
```

### Production Configuration

```yaml
planner:
  instruction_encoder:
    model_name: "bert-base-uncased"
    hidden_size: 768
    max_length: 512
  task_decomposer:
    hidden_size: 768
    num_layers: 4
    dropout: 0.2
  workflow_graph_generator:
    gnn_type: "GAT"
    num_layers: 4
    hidden_dim: 512
    output_dim: 256
    dropout: 0.2
  model_selector:
    state_dim: 1024  # 768 + 256
    action_dim: 8
    hidden_dims: [512, 256, 128]
    dropout: 0.2

worker_models:
  - name: "gpt-4"
    model_type: "llm"
    cost_per_token: 0.03
    latency_ms: 500
  - name: "gpt-3.5-turbo"
    model_type: "llm"
    cost_per_token: 0.002
    latency_ms: 200
  # ... more workers

system:
  max_workflow_depth: 15
  max_parallel_tasks: 10
  timeout_seconds: 600
```

---

## Validation

### Configuration Validation

**Check Configuration:**
```python
from orchestai.utils.config_loader import load_config, validate_config

config = load_config("config.yaml")
errors = validate_config(config)
if errors:
    print("Configuration errors:", errors)
```

**Common Issues:**
- **state_dim mismatch**: Should be `instruction_encoder.hidden_size + workflow_graph_generator.output_dim`
- **action_dim mismatch**: Should match number of worker models
- **Missing required fields**: Check all required fields are present

---

## Next Steps

- See `07_API_INTEGRATION.md` for API integration details
- See `REAL_WORLD_TESTING_GUIDE.md` for testing guide
- See `TESTING_GUIDE.md` for testing documentation

