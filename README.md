# OrchestAI: Autonomous Multi-Model Orchestration via Learned Task Planning

**OrchestAI** is a research project that implements an intelligent Planner Model capable of autonomously orchestrating multiple AI foundation models to solve complex, multi-step tasks. Unlike existing frameworks that rely on fixed pipelines and manual routing, OrchestAI uses Graph Neural Networks (GNNs) and Reinforcement Learning (RL) to dynamically decompose tasks and route sub-tasks to specialized models.

## Table of Contents

- [Overview](#overview)
- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
  - [System Architecture](#system-architecture)
  - [Core Components](#core-components)
  - [Training Pipeline](#training-pipeline)
  - [Execution Flow](#execution-flow)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Research Background](#research-background)
- [Contributing](#contributing)
- [License](#license)

## Overview

OrchestAI addresses a critical gap in AI systems: the lack of intelligent, autonomous orchestration that enables foundation models to collaborate on complex tasks. The system learns to:

1. **Decompose** complex instructions into sub-tasks with dependencies
2. **Generate** valid workflow graphs (DAGs) representing execution plans
3. **Route** each sub-task to the optimal worker model
4. **Optimize** for success rate, cost, and latency through reinforcement learning

## What It Does

### Problem Statement

Large Language Models excel at many tasks but struggle with complex, multi-step problems requiring:
- Cross-modal integration (text, vision, audio)
- Specialized processing (code generation, data analysis, document processing)
- Sequential dependencies (task A must complete before task B)

**Example Task**: "Generate a presentation from a research paper"
- Requires: text extraction → summarization → slide structuring → layout design → visual generation
- Each step optimally handled by different specialized models

### Solution

OrchestAI provides an **autonomous Planner** that:
- Understands natural language instructions
- Automatically breaks down complex tasks into sub-tasks
- Creates execution workflows (DAGs) respecting dependencies
- Selects optimal models for each sub-task
- Learns and improves from experience

### Key Advantages

- **No Manual Engineering**: No need to pre-define workflows or manually select models
- **Adaptive**: Learns optimal routing strategies through RL
- **Cost-Efficient**: Optimizes for cost while maintaining quality
- **Generalizable**: Works on unseen tasks without retraining

## How It Works

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Instruction                          │
│          "Generate a presentation about AI"                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Planner Model                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Instruction  │  │   Task       │  │   Workflow   │     │
│  │   Encoder    │→ │ Decomposer   │→ │   Graph      │     │
│  │  (T5/BERT)   │  │   (LSTM)     │  │  Generator   │     │
│  └──────────────┘  └──────────────┘  │    (GNN)     │     │
│                                      └──────┬───────┘     │
│                                             │              │
│                                      ┌──────▼───────┐     │
│                                      │   Model      │     │
│                                      │  Selector    │     │
│                                      │     (RL)     │     │
│                                      └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Workflow Execution Plan                        │
│  Task 1 (LLM) → Task 2 (Vision) → Task 3 (LLM)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Worker Model Layer                           │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐          │
│  │ GPT-4│  │Llama │  │ CLIP │  │Whisper│  │ Code │  ...    │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Output                              │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Instruction Encoder

**What it does**: Converts natural language instructions into dense vector representations.

**How it works**:
- Uses transformer-based models (T5 or BERT) pre-trained on large text corpora
- Processes instructions through the encoder to extract semantic meaning
- Outputs fixed-size embeddings (768 dimensions) that capture task intent

**Implementation** (`orchestai/planner/instruction_encoder.py`):
```python
# Encodes instruction: "Generate a presentation about AI"
instruction_embedding = encoder("Generate a presentation about AI")
# Returns: [768-dim vector]
```

**Key Features**:
- Supports T5 and BERT architectures
- Handles variable-length inputs (up to 512 tokens)
- Provides pooled and sequence-level embeddings

#### 2. Task Decomposer

**What it does**: Breaks down complex instructions into a sequence of sub-tasks with dependencies.

**How it works**:
- Uses LSTM (Long Short-Term Memory) networks to generate sub-tasks sequentially
- For each sub-task, predicts:
  - **Description**: What the sub-task does
  - **Task Type**: Category (text_extraction, summarization, visualization, etc.)
  - **Dependencies**: Which previous sub-tasks must complete first
  - **Complexity**: Estimated difficulty (0.0 to 1.0)
- Ensures valid dependency structure (no cycles)

**Implementation** (`orchestai/planner/task_decomposer.py`):
```python
# Input: Instruction embedding
# Output: List of SubTask objects
subtasks = [
    SubTask(id=0, description="extract_text", dependencies=[], ...),
    SubTask(id=1, description="summarize", dependencies=[0], ...),
    SubTask(id=2, description="create_slides", dependencies=[1], ...),
]
```

**Architecture**:
- **Encoder**: Projects instruction embedding to hidden space
- **LSTM**: Generates sub-tasks sequentially (up to 20 sub-tasks)
- **Heads**: 
  - Description decoder
  - Task type classifier
  - Dependency predictor (binary classification for each previous task)
  - Complexity estimator
  - Stop token predictor

**Loss Function**:
- Cross-entropy for task type and dependency prediction
- DAG validity constraint (penalizes cycles)

#### 3. Workflow Graph Generator

**What it does**: Constructs Directed Acyclic Graphs (DAGs) representing execution plans from decomposed sub-tasks.

**How it works**:
- Uses Graph Neural Networks (GNNs) to model relationships between sub-tasks
- Takes sub-task embeddings and dependency information
- Applies GNN layers (GCN, GAT, or GraphSAGE) to:
  - Refine node (sub-task) embeddings based on graph structure
  - Predict edge probabilities between sub-tasks
- Ensures output is a valid DAG (no cycles) using topological ordering

**Implementation** (`orchestai/planner/graph_generator.py`):
```python
# Input: Sub-task embeddings + dependencies
# Output: Workflow graph (DAG)
graph = {
    "node_embeddings": [num_subtasks, 128],  # Refined embeddings
    "adjacency": [num_subtasks, num_subtasks],  # Binary adjacency matrix
    "edge_probs": [num_subtasks, num_subtasks],  # Edge probabilities
}
```

**GNN Types Supported**:
- **GCN (Graph Convolutional Network)**: Standard message passing
- **GAT (Graph Attention Network)**: Attention-based aggregation (4 heads)
- **GraphSAGE**: Sampling and aggregation

**DAG Validation**:
- Topological sort to determine execution order
- Cycle detection to ensure validity
- Penalizes invalid structures during training

#### 4. RL-Based Model Selector

**What it does**: Selects the optimal worker model for each sub-task using Reinforcement Learning.

**How it works**:
- Uses a policy network (actor) to select models based on:
  - Instruction embedding (global context)
  - Sub-task node embedding (local context)
  - Combined state representation
- Uses a value network (critic) to estimate expected returns
- Optimized using PPO (Proximal Policy Optimization) to:
  - Maximize task success rate
  - Minimize cost (API calls × model prices)
  - Minimize latency (execution time)

**Implementation** (`orchestai/planner/model_selector.py`):
```python
# Input: Combined state [instruction_emb + node_emb]
# Output: Model selection (index 0-7)
state = concat([instruction_embedding, node_embedding])
action, log_prob, entropy = selector.select_action(state)
# action: 0 (GPT-4), 1 (GPT-3.5), 2 (Llama), etc.
```

**Reward Function**:
```
R = α·Success - β·Cost - γ·Latency + shaped_rewards
```
Where:
- `α = 10.0`: Success reward weight
- `β = 0.1`: Cost penalty weight
- `γ = 0.01`: Latency penalty weight
- `shaped_rewards`: +0.1 per successful sub-task

**PPO Algorithm**:
- Clipped objective to prevent large policy updates
- Value function learning for advantage estimation
- Entropy bonus for exploration

#### 5. Worker Model Layer

**What it does**: Manages and executes tasks on specialized AI models.

**How it works**:
- Provides unified interface for different model types:
  - **LLM Workers**: GPT-4, GPT-3.5, Llama, Code-Llama
  - **Vision Workers**: CLIP, image generation models
  - **Audio Workers**: Whisper (ASR), TTS models
  - **Document Workers**: PDF processing, document generation
  - **Data Workers**: Data analysis, visualization
- Standardizes I/O format (JSON-based)
- Tracks cost and latency per execution
- Handles errors gracefully

**Implementation** (`orchestai/worker/`):
```python
# Execute task on selected worker
output = worker_layer.execute_task(
    worker_id=0,  # GPT-4
    task="summarize",
    data="Long text to summarize...",
)
# Returns: WorkerOutput(content, cost, latency, success, error)
```

**Worker Interface**:
- `process()`: Execute task and return standardized output
- `validate_input()`: Validate input format
- `estimate_cost()`: Estimate processing cost
- `estimate_latency()`: Estimate processing time

#### 6. Orchestration System

**What it does**: Coordinates the Planner and Worker models to execute complete workflows.

**How it works**:
1. **Planning Phase**:
   - Takes user instruction
   - Runs through Planner Model to generate workflow
   - Extracts execution plan (sub-tasks, dependencies, model selections)

2. **Execution Phase**:
   - Performs topological sort to determine execution order
   - Executes sub-tasks sequentially (respecting dependencies)
   - For each sub-task:
     - Gets selected worker model
     - Prepares input (combines initial input + dependency outputs)
     - Executes task
     - Stores result
   - Combines all outputs into final result

3. **Learning Phase**:
   - Records execution history
   - Computes rewards (success, cost, latency)
   - Updates Planner Model via RL

**Implementation** (`orchestai/orchestrator.py`):
```python
# Execute complete workflow
result = orchestrator.execute(
    instruction="Generate a presentation about AI",
    input_data={"paper": "research_paper.pdf"},
)
# Returns: ExecutionResult(success, outputs, cost, latency, error)
```

**Topological Sort**:
- Uses Kahn's algorithm to order sub-tasks
- Ensures dependencies are satisfied
- Handles parallel execution (future enhancement)

### Training Pipeline

#### Phase 1: Supervised Pre-Training (Months 7-9)

**Objective**: Learn valid task decompositions and reasonable model selections.

**Data**:
- 200-300 expert-annotated workflows
- 1000 synthetic augmentations (generated via LLM)

**Training Process**:
1. **Forward Pass**:
   - Encode instruction
   - Decompose into sub-tasks
   - Generate workflow graph
   - Select models

2. **Loss Computation**:
   ```
   L = L_CE + λ·L_DAG
   ```
   - `L_CE`: Cross-entropy for task type and model selection prediction
   - `L_DAG`: DAG validity loss (penalizes cycles)
   - `λ = 0.3`: DAG loss weight

3. **Optimization**:
   - AdamW optimizer (lr=1e-4)
   - Learning rate scheduling (ReduceLROnPlateau)
   - Gradient clipping (max_norm=1.0)

**Expected Outcome**:
- Planner generates syntactically correct workflows (valid DAGs)
- 60-70% task success rate on validation set

**Script**: `scripts/train_phase1.py`

#### Phase 2: Reinforcement Learning Fine-Tuning (Months 10-13)

**Objective**: Optimize for task completion success while minimizing cost.

**Algorithm**: Proximal Policy Optimization (PPO)

**Training Process**:
1. **Episode Execution**:
   - Sample task from task pool
   - Generate plan using current policy
   - Execute workflow on worker models
   - Compute reward based on success, cost, latency

2. **Reward Computation**:
   ```
   R = α·Success - β·Cost - γ·Latency + shaped_rewards
   ```
   - Success: Binary (1 if task completes successfully, 0 otherwise)
   - Cost: Sum of (API calls × model prices)
   - Latency: Total execution time in milliseconds
   - Shaped rewards: +0.1 per successful sub-task

3. **Policy Update**:
   - Collect experiences (states, actions, rewards)
   - Compute advantages using value function
   - Update policy using PPO clipped objective
   - Update value function to minimize TD error

**Addressing Challenges**:
- **Sparse Rewards**: Shaped rewards provide intermediate feedback
- **Exploration**: ε-greedy policy (ε=0.2) ensures trying diverse models
- **Cost Collapse**: Cost penalty prevents always using expensive models
- **Curriculum Learning**: Start with easy tasks, gradually increase difficulty

**Expected Outcome**:
- Improved success rate (target: 75-85%)
- Reduced cost (15-20% improvement vs. static pipeline)
- Better generalization to unseen tasks

**Script**: `scripts/train_phase2.py`

#### Phase 3: Continuous Improvement (Months 14-20)

**Objective**: Online learning from execution logs during benchmark evaluation.

**Process**:
- Collect execution logs from real-world usage
- Periodically update Planner Model (every 100 executions)
- Fine-tune on successful workflows
- Adapt to new task patterns

### Execution Flow

#### Step-by-Step Example

**Input**: "Generate a presentation from this research paper"

1. **Instruction Encoding**:
   ```
   Instruction → T5 Encoder → [768-dim embedding]
   ```

2. **Task Decomposition**:
   ```
   Embedding → LSTM → Sub-tasks:
   - Task 0: Extract text from PDF (dependencies: [])
   - Task 1: Summarize text (dependencies: [0])
   - Task 2: Structure into slides (dependencies: [1])
   - Task 3: Generate visuals (dependencies: [2])
   ```

3. **Graph Generation**:
   ```
   Sub-tasks → GNN → Workflow Graph (DAG):
   0 → 1 → 2 → 3
   ```

4. **Model Selection**:
   ```
   For each sub-task:
   - Task 0: Select Document Worker (model_id=6)
   - Task 1: Select GPT-3.5 (model_id=1) [cheaper than GPT-4]
   - Task 2: Select GPT-4 (model_id=0) [needs quality]
   - Task 3: Select CLIP (model_id=3)
   ```

5. **Execution**:
   ```
   Topological order: [0, 1, 2, 3]
   
   Step 1: Execute Task 0 (Document Worker)
   - Input: research_paper.pdf
   - Output: extracted_text.txt
   
   Step 2: Execute Task 1 (GPT-3.5)
   - Input: extracted_text.txt
   - Output: summary.txt
   
   Step 3: Execute Task 2 (GPT-4)
   - Input: summary.txt
   - Output: slide_structure.json
   
   Step 4: Execute Task 3 (CLIP)
   - Input: slide_structure.json
   - Output: presentation_with_visuals.pptx
   ```

6. **Result**:
   ```
   Success: True
   Total Cost: $0.15
   Total Latency: 2.3 seconds
   Output: presentation_with_visuals.pptx
   ```

## Key Features

### 1. Autonomous Planning
- No manual workflow engineering required
- Automatically decomposes complex tasks
- Generates valid execution plans

### 2. Learned Optimization
- RL-based model selection
- Optimizes for success, cost, and latency
- Adapts to new task patterns

### 3. Multi-Model Coordination
- Seamlessly integrates different model types
- Handles cross-modal dependencies
- Standardized I/O interface

### 4. Cost Efficiency
- Selects cheaper models when sufficient
- Balances quality and cost
- Tracks and optimizes total cost

### 5. Robust Execution
- Handles errors gracefully
- Validates workflow structure (DAG)
- Provides detailed execution logs

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YasarMushtaq1/orchest-ai.git
   cd orchest-ai
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package** (optional):
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```python
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

# Load configuration
config = load_config("config.yaml")

# Setup system
orchestrator = setup_system(config)

# Execute task
result = orchestrator.execute(
    instruction="Generate a presentation about machine learning",
    input_data={"topic": "machine learning"},
)

print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost:.4f}")
print(f"Output: {result.outputs}")
```

### Training Phase 1 (Supervised)

```bash
python scripts/train_phase1.py \
    --config config.yaml \
    --epochs 50 \
    --batch-size 32 \
    --use-wandb
```

### Training Phase 2 (RL)

```bash
python scripts/train_phase2.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --episodes 1000 \
    --use-wandb
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --tasks benchmark_tasks.json \
    --output results.json
```

### Inference

```bash
python scripts/run_inference.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --instruction "Analyze this dataset and create visualizations" \
    --input input_data.json \
    --output result.json
```

## Project Structure

```
orchest-ai/
├── orchestai/                 # Main package
│   ├── planner/               # Planner Model components
│   │   ├── instruction_encoder.py
│   │   ├── task_decomposer.py
│   │   ├── graph_generator.py
│   │   ├── model_selector.py
│   │   └── planner_model.py
│   ├── worker/                # Worker Model Layer
│   │   ├── base_worker.py
│   │   ├── llm_worker.py
│   │   ├── vision_worker.py
│   │   ├── audio_worker.py
│   │   └── worker_layer.py
│   ├── training/              # Training infrastructure
│   │   ├── supervised_trainer.py
│   │   └── rl_trainer.py
│   ├── evaluation/            # Evaluation framework
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── utils/                 # Utilities
│   │   ├── config_loader.py
│   │   └── setup.py
│   └── orchestrator.py        # Main orchestration system
├── scripts/                   # Training and evaluation scripts
│   ├── train_phase1.py
│   ├── train_phase2.py
│   ├── evaluate.py
│   └── run_inference.py
├── examples/                  # Example usage
│   └── demo.py
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## Configuration

The `config.yaml` file contains all system configuration:

### Planner Configuration

```yaml
planner:
  instruction_encoder:
    model_name: "t5-base"  # or "bert-base-uncased"
    hidden_size: 768
    max_length: 512
    
  task_decomposer:
    hidden_size: 512
    num_layers: 3
    dropout: 0.1
    
  workflow_graph_generator:
    gnn_type: "GCN"  # GCN, GAT, or GraphSAGE
    num_layers: 3
    hidden_dim: 256
    output_dim: 128
    
  model_selector:
    state_dim: 512
    action_dim: 8  # number of worker models
    hidden_dims: [256, 128]
```

### Worker Models

```yaml
worker_models:
  - name: "gpt-4"
    type: "llm"
    cost_per_token: 0.03
    latency_ms: 500
  # ... more models
```

### Training Configuration

```yaml
training:
  phase1_supervised:
    batch_size: 32
    learning_rate: 1e-4
    num_epochs: 50
    lambda_dag: 0.3
    
  phase2_rl:
    algorithm: "PPO"
    learning_rate: 3e-5
    num_episodes: 1000
    alpha: 10.0  # success reward weight
    beta: 0.1   # cost penalty weight
    gamma_latency: 0.01
```

## Research Background

### Motivation

Existing multi-model orchestration systems (HuggingGPT, AutoGen) suffer from:
- **Static Pipelines**: Require manual workflow definition
- **Rule-Based Routing**: Cannot adapt to novel tasks
- **No Learning**: Don't improve from experience

### Novel Contributions

1. **Learned Task Decomposition**: Uses LSTM to automatically break down tasks
2. **GNN-Based Workflow Generation**: Constructs valid DAGs using graph neural networks
3. **RL-Optimized Routing**: Learns optimal model selection via PPO
4. **End-to-End Learning**: All components trained jointly

### Comparison with Existing Systems

| System | Decomposition | Routing | Learning | Limitation |
|--------|--------------|---------|----------|------------|
| HuggingGPT | LLM-generated | Deterministic | None | Fixed model mappings |
| AutoGen | Rule-based | Manual config | None | Requires expertise |
| ReAct | LLM reasoning | LLM-selected | Via prompts | No workflow planning |
| **OrchestAI** | **GNN-learned** | **RL-optimized** | **Yes** | **(addresses all)** |

## Technical Details

### Model Sizes

- **Instruction Encoder**: T5-Base (220M parameters)
- **Task Decomposer**: ~50M parameters (3-layer LSTM)
- **Graph Generator**: ~20M parameters (3-layer GNN)
- **Model Selector**: ~5M parameters (2-layer MLP)
- **Total Planner**: ~295M parameters

### Training Data

- **Phase 1**: 200-300 expert-annotated workflows + 1000 synthetic
- **Phase 2**: Online RL with task pool (1000+ episodes)

### Computational Requirements

- **Training**: GPU with 16GB+ VRAM (RTX 3090, A100)
- **Inference**: CPU or GPU (CPU sufficient for small workloads)
- **Memory**: 16GB+ RAM

## Contributing

This is a research project. Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use OrchestAI in your research, please cite:

```bibtex
@misc{orchestai,
  title={OrchestAI: Autonomous Multi-Model Orchestration via Learned Task Planning},
  author={Mushtaq, Sayed Yasar Ahmad},
  year={2024},
  url={https://github.com/YasarMushtaq1/orchest-ai}
}
```

## Contact

- **Author**: Sayed Yasar Ahmad Mushtaq
- **Email**: 271mushtaq@gmail.com
- **GitHub**: [github.com/YasarMushtaq1/orchest-ai](https://github.com/YasarMushtaq1/orchest-ai)

---

**Note**: This is a research prototype. For production use, additional testing, optimization, and security considerations are required.

