# OrchestAI Architecture Overview

## What is OrchestAI?

OrchestAI is an autonomous multi-model orchestration system that uses learned planning to intelligently decompose complex tasks into sub-tasks, generate execution workflows, and route each sub-task to the most appropriate AI model. Unlike rule-based systems, OrchestAI learns optimal task decomposition and model selection through supervised pre-training and reinforcement learning.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OrchestrationSystem                       │
│  (Main Coordinator - Manages Planning & Execution)          │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌─────────▼──────────┐
│  PlannerModel  │            │  WorkerModelLayer   │
│                │            │                     │
│  - Plans       │            │  - Executes        │
│  - Decomposes  │            │  - Routes          │
│  - Routes      │            │  - Manages         │
└───────┬────────┘            └────────────────────┘
        │
        ├─── InstructionEncoder (T5/BERT)
        ├─── TaskDecomposer (LSTM-based)
        ├─── WorkflowGraphGenerator (GNN-based)
        └─── RLModelSelector (PPO-based)
```

## Core Components

### 1. Planner Model (`orchestai/planner/planner_model.py`)

**What it does:**
- Takes natural language instructions as input
- Decomposes complex tasks into sub-tasks with dependencies
- Generates workflow graphs (DAGs) representing execution plans
- Selects optimal worker models for each sub-task

**Why it exists:**
- Traditional rule-based planners are brittle and don't adapt
- Learned planning can discover optimal decompositions from data
- End-to-end learning allows optimization for success, cost, and latency

**How it works:**
1. **Instruction Encoding**: Converts natural language to dense embeddings using T5/BERT
2. **Task Decomposition**: Uses LSTM to sequentially generate sub-tasks with dependencies
3. **Graph Generation**: Uses Graph Neural Networks (GCN/GAT/GraphSAGE) to construct DAGs
4. **Model Selection**: Uses Reinforcement Learning (PPO) to route tasks to optimal workers

### 2. Worker Model Layer (`orchestai/worker/worker_layer.py`)

**What it does:**
- Manages a collection of specialized AI models (LLM, Vision, Audio, etc.)
- Provides unified interface for task execution
- Tracks costs and latencies for each worker
- Supports dynamic model discovery from HuggingFace Hub

**Why it exists:**
- Different tasks require different model capabilities
- Need centralized management of multiple models
- Cost and latency tracking enables optimization

**How it works:**
- Maintains a registry of worker models with their configurations
- Routes execution requests to appropriate workers based on worker_id
- Each worker implements `BaseWorker` interface for consistent execution
- Supports both pre-configured and dynamically discovered models

### 3. Orchestration System (`orchestai/orchestrator.py`)

**What it does:**
- Coordinates planning and execution phases
- Manages workflow execution following topological order
- Handles parallel execution of independent tasks
- Implements retry mechanisms and error handling
- Logs executions for continuous learning

**Why it exists:**
- Need a central coordinator to manage the entire pipeline
- Workflow execution requires dependency management
- Parallel execution improves efficiency
- Retry mechanisms improve robustness

**How it works:**
1. **Planning Phase**: Calls PlannerModel to generate workflow
2. **Topological Sort**: Determines execution order respecting dependencies
3. **Parallel Execution**: Executes independent tasks concurrently (up to `max_parallel_tasks`)
4. **Dependency Resolution**: Ensures tasks only execute when dependencies are ready
5. **Output Combination**: Combines sub-task outputs into final result
6. **Logging**: Records execution for training data collection

## Data Flow

### Execution Flow

```
User Instruction
    │
    ▼
┌─────────────────────┐
│ InstructionEncoder  │ → [batch_size, hidden_size] embeddings
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  TaskDecomposer     │ → Subtasks with dependencies
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ WorkflowGraphGen     │ → DAG with node embeddings
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  RLModelSelector    │ → Worker selection for each subtask
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ OrchestrationSystem │ → Executes workflow
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  WorkerModelLayer   │ → Executes each subtask
└──────────┬──────────┘
           │
           ▼
    Final Output
```

### Training Flow

```
Execution Logs (JSONL)
    │
    ▼
┌─────────────────────┐
│ prepare_training_data│ → Structured training dataset
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ SupervisedTrainer   │ → Phase 1: Pre-training
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   RLTrainer         │ → Phase 2: RL fine-tuning
└──────────┬──────────┘
           │
           ▼
    Trained Planner
```

## Key Design Decisions

### 1. Learned Planning vs. Rule-Based

**Decision**: Use learned planning with neural networks

**Why:**
- Rule-based systems are brittle and require manual engineering
- Learned systems can discover optimal decompositions from data
- Can optimize for multiple objectives (success, cost, latency) simultaneously
- Adapts to new task types without code changes

**Trade-offs:**
- Requires training data (solved with execution logging)
- Less interpretable than rules (mitigated with hybrid planner)
- Training time required (acceptable for research system)

### 2. Graph Neural Networks for Workflow Generation

**Decision**: Use GNNs (GCN/GAT/GraphSAGE) to generate workflow graphs

**Why:**
- GNNs naturally handle graph-structured data
- Can learn complex dependency patterns
- Outputs valid DAGs through learned constraints
- Node embeddings capture task relationships

**Trade-offs:**
- More complex than simple sequential models
- Requires graph construction from dependencies
- But provides better structure for complex workflows

### 3. Reinforcement Learning for Model Selection

**Decision**: Use PPO (Proximal Policy Optimization) for model routing

**Why:**
- Model selection is a sequential decision problem
- Need to optimize for long-term rewards (success, cost, latency)
- RL can learn from execution feedback
- PPO is stable and sample-efficient

**Trade-offs:**
- Requires reward shaping (success, cost, latency)
- More complex than supervised learning
- But enables optimization for multiple objectives

### 4. Hybrid Planning

**Decision**: Combine learned planning with LLM fallback

**Why:**
- Learned planner may fail on novel tasks
- LLM can provide fallback for edge cases
- Best of both worlds: learned efficiency + LLM flexibility

**Implementation**: `HybridPlanner` class that tries learned planner first, falls back to LLM

### 5. Parallel Execution

**Decision**: Execute independent tasks in parallel

**Why:**
- Significantly reduces total latency
- Modern systems have parallel execution capability
- DAG structure naturally identifies independent tasks

**Implementation**: Topological sort identifies ready tasks, `ThreadPoolExecutor` executes them concurrently

## System Capabilities

### What OrchestAI Can Do

1. **Task Decomposition**: Break complex instructions into sub-tasks
2. **Workflow Generation**: Create valid execution plans (DAGs)
3. **Model Selection**: Route tasks to optimal workers
4. **Parallel Execution**: Execute independent tasks concurrently
5. **Cost Optimization**: Minimize API costs through intelligent routing
6. **Error Handling**: Retry failed tasks automatically
7. **Continuous Learning**: Improve from execution logs

### Limitations

1. **Training Data**: Requires execution logs for training (solved with data collection)
2. **Model Availability**: Depends on configured worker models
3. **API Costs**: Real-world execution incurs API costs
4. **Latency**: Planning adds overhead (acceptable for complex tasks)

## Next Steps

- See `02_CORE_COMPONENTS.md` for detailed component documentation
- See `03_DESIGN_DECISIONS.md` for rationale behind architectural choices
- See `04_DATA_FLOW.md` for detailed data flow documentation
- See `05_TRAINING_PIPELINE.md` for training process documentation

