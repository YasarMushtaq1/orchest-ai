# Core Components Documentation

This document provides detailed documentation for each core component of OrchestAI, explaining what it does, why it exists, and how it works.

## Table of Contents

1. [Planner Model Components](#planner-model-components)
2. [Worker Model Layer](#worker-model-layer)
3. [Orchestration System](#orchestration-system)
4. [Training Components](#training-components)
5. [Utility Components](#utility-components)

---

## Planner Model Components

### Instruction Encoder (`orchestai/planner/instruction_encoder.py`)

**What it does:**
- Encodes natural language instructions into dense vector representations
- Uses pre-trained language models (T5 or BERT) for semantic understanding
- Provides fixed-size embeddings for downstream components

**Why it exists:**
- Neural networks require numerical inputs, not text
- Pre-trained models capture semantic meaning better than random embeddings
- Fixed-size embeddings enable batch processing

**How it works:**
```python
# Pseudocode
instruction = "Summarize this document and create a presentation"
embedding = encoder(instruction)  # [768] or [512] dimensional vector
```

**Key Features:**
- Supports T5-base and BERT-base-uncased
- Handles variable-length inputs (truncates/pads to max_length)
- Returns pooled embeddings suitable for downstream tasks

**Configuration:**
- `model_name`: "t5-base" or "bert-base-uncased"
- `hidden_size`: 768 (BERT) or 512 (T5)
- `max_length`: 512 tokens

---

### Task Decomposer (`orchestai/planner/task_decomposer.py`)

**What it does:**
- Breaks complex instructions into a sequence of sub-tasks
- Predicts dependencies between sub-tasks
- Estimates complexity for each sub-task
- Determines task types (text_extraction, summarization, etc.)

**Why it exists:**
- Complex tasks require multiple steps
- Need to understand task relationships (dependencies)
- Different sub-tasks may need different models

**How it works:**
```python
# Architecture
Instruction Embedding [batch, hidden_size]
    │
    ▼
Encoder (Linear + LayerNorm + ReLU)
    │
    ▼
LSTM (Sequential Generation)
    │
    ├─── Description Decoder → Subtask Embeddings
    ├─── Task Type Head → Task Type Classification
    ├─── Dependency Head → Dependency Prediction
    ├─── Complexity Head → Complexity Estimation
    └─── Stop Head → Continue/Stop Decision
```

**Key Components:**
1. **LSTM Encoder**: Generates sub-tasks sequentially
2. **Description Decoder**: Creates embeddings for each sub-task
3. **Task Type Classifier**: Predicts task type (10 predefined types)
4. **Dependency Predictor**: Predicts which previous tasks are dependencies
5. **Complexity Estimator**: Estimates task complexity (0.0 to 1.0)
6. **Stop Predictor**: Decides when to stop generating sub-tasks

**Output Format:**
```python
{
    "subtask_embeddings": [batch, num_subtasks, hidden_size],
    "task_types": [batch, num_subtasks, num_task_types],
    "dependencies": [batch, num_subtasks, max_subtasks],
    "complexities": [batch, num_subtasks, 1],
    "stop_probs": [batch, num_subtasks, 1]
}
```

**Design Rationale:**
- **LSTM**: Sequential generation allows modeling task order
- **Dependency Prediction**: Binary classification for each possible dependency
- **Complexity Estimation**: Helps with model selection and resource allocation
- **Stop Prediction**: Prevents generating too many sub-tasks

---

### Workflow Graph Generator (`orchestai/planner/graph_generator.py`)

**What it does:**
- Constructs Directed Acyclic Graphs (DAGs) from sub-task embeddings
- Uses Graph Neural Networks to learn node relationships
- Ensures valid DAG structure (no cycles)
- Generates adjacency matrices for execution

**Why it exists:**
- Workflows are graph-structured (not linear)
- Need to represent dependencies as edges
- GNNs can learn complex dependency patterns
- DAG structure ensures valid execution order

**How it works:**
```python
# Architecture
Subtask Embeddings [batch, num_subtasks, input_dim]
    │
    ▼
Input Projection (Linear)
    │
    ▼
GNN Layers (GCN/GAT/GraphSAGE)
    │
    ├─── Node Embeddings → [batch, num_subtasks, output_dim]
    └─── Edge Predictor → Edge Probabilities
    │
    ▼
DAG Construction → Adjacency Matrix
```

**GNN Types Supported:**
1. **GCN (Graph Convolutional Network)**: Standard graph convolution
2. **GAT (Graph Attention Network)**: Attention-based aggregation
3. **GraphSAGE**: Sampling and aggregation approach

**Key Features:**
- **DAG Validation**: Ensures no cycles in generated graphs
- **Edge Prediction**: Predicts edges between all node pairs
- **Topological Ordering**: Respects dependency constraints
- **NetworkX Integration**: Can export to NetworkX graphs for visualization

**DAG Construction Algorithm:**
1. Start with dependencies from Task Decomposer
2. Add high-probability edges if they don't create cycles
3. Verify DAG property using cycle detection
4. Return binary adjacency matrix

**Bug Fix Note:**
- Fixed GAT hidden dimension issue: GAT with 4 heads concatenates to `hidden_dim * 4`, but only on non-final layers
- Final layer uses `concat=False` to output `hidden_dim` size

---

### RL Model Selector (`orchestai/planner/model_selector.py`)

**What it does:**
- Selects optimal worker model for each sub-task
- Uses Reinforcement Learning (PPO) to optimize selection
- Considers task characteristics and model capabilities
- Optimizes for success rate, cost, and latency

**Why it exists:**
- Different models have different strengths
- Need to balance cost, latency, and quality
- RL can learn optimal routing from execution feedback
- Sequential decision problem suited for RL

**How it works:**
```python
# Architecture
State = [instruction_embedding, node_embedding]
    │
    ├─── Policy Network (Actor) → Action Probabilities
    └─── Value Network (Critic) → State Value Estimate
    │
    ▼
Action = Selected Worker ID
```

**PPO Algorithm:**
1. **Policy Network**: Outputs probability distribution over workers
2. **Value Network**: Estimates expected return from current state
3. **Action Selection**: Sample from policy (or argmax for deterministic)
4. **Loss Computation**: PPO clipped objective + value loss + entropy bonus

**Reward Structure:**
- **Success Reward**: +10.0 if task succeeds
- **Cost Penalty**: -0.1 * total_cost
- **Latency Penalty**: -0.01 * total_latency_ms
- **Shaped Reward**: +0.1 per successful sub-task

**Key Features:**
- **Stochastic Policy**: Explores different model selections
- **Value Estimation**: Reduces variance in policy updates
- **Entropy Bonus**: Encourages exploration
- **PPO Clipping**: Prevents large policy updates

---

### Hybrid Planner (`orchestai/planner/hybrid_planner.py`)

**What it does:**
- Combines learned planning with LLM fallback
- Tries learned planner first, falls back to LLM if needed
- Provides flexibility for novel tasks

**Why it exists:**
- Learned planner may fail on unseen task types
- LLM can handle edge cases
- Best of both worlds: efficiency + flexibility

**How it works:**
1. Try learned planner
2. If confidence is low or task is novel, use LLM
3. Combine results appropriately

---

## Worker Model Layer

### Base Worker (`orchestai/worker/base_worker.py`)

**What it does:**
- Defines interface for all worker models
- Provides common functionality (cost tracking, latency measurement)
- Standardizes worker output format

**Why it exists:**
- Need consistent interface across different model types
- Common functionality (cost, latency) should be shared
- Makes it easy to add new worker types

**Interface:**
```python
class BaseWorker:
    def process(self, input_data: Dict) -> WorkerOutput:
        # Execute task and return result
        pass
```

**WorkerOutput Format:**
```python
@dataclass
class WorkerOutput:
    content: Any  # Task output
    metadata: Dict[str, Any]  # Additional info
    cost: float  # Execution cost
    latency_ms: float  # Execution time
    success: bool  # Whether task succeeded
    error: Optional[str]  # Error message if failed
```

---

### LLM Worker (`orchestai/worker/llm_worker.py`)

**What it does:**
- Executes text-based tasks using Large Language Models
- Supports OpenAI API integration
- Falls back to mock responses if API unavailable

**Why it exists:**
- Most tasks are text-based
- LLMs are versatile (summarization, generation, analysis)
- OpenAI API provides reliable service

**How it works:**
1. Receives task description and input text
2. Calls OpenAI API (if key available) or uses mock
3. Tracks cost and latency
4. Returns formatted output

**API Integration:**
- Uses `openai.OpenAI()` client (v1.0+)
- Loads API key from environment variable `OPENAI_API_KEY`
- Supports `.env` file via `python-dotenv`

**Cost Tracking:**
- Estimates cost based on input/output tokens
- Uses `cost_per_token` from configuration
- Accumulates total cost per execution

---

### Vision Worker (`orchestai/worker/vision_worker.py`)

**What it does:**
- Handles image-based tasks
- Uses CLIP or similar vision models
- Processes image inputs

**Why it exists:**
- Some tasks require vision capabilities
- Different from text processing
- Need specialized worker for images

**Status**: Placeholder implementation (can be extended with real vision models)

---

### Audio Worker (`orchestai/worker/audio_worker.py`)

**What it does:**
- Handles audio-based tasks
- Uses Whisper or similar audio models
- Processes audio inputs

**Why it exists:**
- Audio tasks require specialized processing
- Different from text/vision
- Need dedicated worker for audio

**Status**: Placeholder implementation (can be extended with real audio models)

---

### Worker Model Layer (`orchestai/worker/worker_layer.py`)

**What it does:**
- Manages collection of worker models
- Routes tasks to appropriate workers
- Provides unified interface for execution
- Supports dynamic model discovery

**Why it exists:**
- Need centralized management of multiple workers
- Consistent interface for orchestrator
- Easy to add/remove workers

**Key Methods:**
- `execute_task(worker_id, task, data)`: Execute task with specified worker
- `get_worker(worker_id)`: Get worker by ID
- `list_workers()`: List all available workers
- `estimate_total_cost()`: Estimate cost for workflow

**Model Discovery:**
- Can discover models from HuggingFace Hub
- Supports local model endpoints
- Automatically integrates discovered models

---

## Orchestration System

### OrchestrationSystem (`orchestai/orchestrator.py`)

**What it does:**
- Main coordinator for planning and execution
- Manages workflow execution
- Handles parallel execution
- Implements retry mechanisms
- Logs executions for training

**Why it exists:**
- Need central coordinator
- Workflow execution is complex (dependencies, parallelism)
- Error handling and retries improve robustness
- Execution logging enables continuous learning

**Key Methods:**
- `execute(instruction, input_data)`: Main execution method
- `_topological_sort()`: Determine execution order
- `_get_ready_tasks()`: Find tasks ready to execute
- `_execute_single_task()`: Execute one task
- `_execute_task_with_retry()`: Execute with retry logic

**Execution Flow:**
1. **Planning**: Call PlannerModel to generate workflow
2. **Topological Sort**: Determine execution order
3. **Parallel Execution**: Execute ready tasks concurrently
4. **Dependency Resolution**: Wait for dependencies
5. **Output Combination**: Combine sub-task outputs
6. **Logging**: Record execution for training

**Parallel Execution:**
- Uses `ThreadPoolExecutor` for concurrent execution
- Limits parallelism to `max_parallel_tasks`
- Tracks ready tasks (all dependencies completed)
- Executes multiple tasks simultaneously when possible

**Retry Mechanism:**
- Retries failed tasks up to `max_retries` times
- Exponential backoff between retries
- Tracks retry count in ExecutionResult

**Error Handling:**
- Catches exceptions at task level
- Returns WorkerOutput with error information
- Continues execution if possible
- Returns failure if critical task fails

---

## Training Components

### Supervised Trainer (`orchestai/training/supervised_trainer.py`)

**What it does:**
- Trains PlannerModel using supervised learning (Phase 1)
- Uses expert-annotated workflows
- Optimizes for task decomposition and model selection accuracy

**Why it exists:**
- Need initial training before RL
- Supervised learning provides good initialization
- Faster than RL-only training

**Training Process:**
1. Load training data (instructions + expert workflows)
2. Forward pass through PlannerModel
3. Compute supervised loss (CE loss + DAG loss)
4. Backward pass and optimization
5. Validation and checkpointing

**Loss Components:**
- **Cross-Entropy Loss**: Task type and model selection accuracy
- **DAG Loss**: Penalizes invalid graph structures
- **Total Loss**: Weighted combination

**Key Features:**
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping
- Weights & Biases integration
- Checkpoint saving

---

### RL Trainer (`orchestai/training/rl_trainer.py`)

**What it does:**
- Fine-tunes PlannerModel using Reinforcement Learning (Phase 2)
- Uses PPO algorithm
- Optimizes for success rate, cost, and latency

**Why it exists:**
- Supervised learning doesn't optimize for execution outcomes
- RL can learn from execution feedback
- Optimizes for multiple objectives simultaneously

**Training Process:**
1. Collect trajectories (execute workflows)
2. Compute rewards (success, cost, latency)
3. Compute advantages (GAE - Generalized Advantage Estimation)
4. Update policy using PPO
5. Repeat for multiple episodes

**PPO Algorithm:**
- Clipped surrogate objective
- Value function learning
- Entropy bonus for exploration
- Multiple epochs per batch

---

## Utility Components

### Execution Logger (`orchestai/utils/execution_logger.py`)

**What it does:**
- Logs execution results to JSONL files
- Serializes PyTorch tensors to JSON-compatible format
- Enables training data collection

**Why it exists:**
- Need to collect real-world execution data
- Training requires execution logs
- JSONL format is easy to process

**Key Features:**
- Automatic tensor serialization
- Timestamped log files
- Structured log format
- Easy to convert to training data

---

### Cost Optimizer (`orchestai/utils/cost_optimizer.py`)

**What it does:**
- Tracks API costs across executions
- Provides cost estimation utilities
- Suggests cost-optimized model selections

**Why it exists:**
- API costs can be significant
- Need to optimize for cost
- Helps with budget management

---

### Model Discovery (`orchestai/worker/model_discovery.py`)

**What it does:**
- Discovers models from HuggingFace Hub
- Integrates discovered models into worker layer
- Supports local model endpoints

**Why it exists:**
- Don't want to hardcode all models
- New models are released frequently
- Dynamic discovery is more flexible

---

### Setup Utilities (`orchestai/utils/setup.py`)

**What it does:**
- Initializes complete system from configuration
- Creates PlannerModel, WorkerLayer, OrchestrationSystem
- Handles device placement (CPU/GPU)

**Why it exists:**
- Simplifies system initialization
- Ensures correct component setup
- Reduces boilerplate code

**Usage:**
```python
from orchestai.utils.setup import setup_system
from orchestai.utils.config_loader import load_config

config = load_config("config.yaml")
orchestrator = setup_system(config)
```

---

## Component Interactions

### Planning Phase
```
InstructionEncoder → TaskDecomposer → WorkflowGraphGenerator → RLModelSelector
```

### Execution Phase
```
OrchestrationSystem → WorkerModelLayer → BaseWorker (LLM/Vision/Audio)
```

### Training Phase
```
ExecutionLogger → prepare_training_data → SupervisedTrainer/RLTrainer → PlannerModel
```

---

## Next Steps

- See `03_DESIGN_DECISIONS.md` for rationale behind component design
- See `04_DATA_FLOW.md` for detailed data flow documentation
- See `05_TRAINING_PIPELINE.md` for training process details

