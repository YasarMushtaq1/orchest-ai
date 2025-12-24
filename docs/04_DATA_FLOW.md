# Data Flow Documentation

This document explains how data flows through OrchestAI, from user instructions to final outputs, and through the training pipeline.

## Table of Contents

1. [Execution Flow](#execution-flow)
2. [Training Data Flow](#training-data-flow)
3. [Component Data Formats](#component-data-formats)
4. [Tensor Shapes and Dimensions](#tensor-shapes-and-dimensions)

---

## Execution Flow

### 1. User Input → Instruction Encoding

**Input:**
```python
instruction = "Summarize this document and create a presentation"
input_data = {"text": "Document content here..."}
```

**Processing:**
```python
# InstructionEncoder
instruction_embedding = encoder(instruction)
# Shape: [batch_size=1, hidden_size=768]
```

**Output:**
- Dense vector representation of instruction
- Fixed size for batch processing
- Semantic meaning captured in embedding

---

### 2. Instruction Embedding → Task Decomposition

**Input:**
```python
instruction_embedding  # [1, 768]
```

**Processing:**
```python
# TaskDecomposer
decomposition_outputs = task_decomposer(instruction_embedding)
# Outputs:
#   subtask_embeddings: [1, num_subtasks, 512]
#   task_types: [1, num_subtasks, 10]
#   dependencies: [1, num_subtasks, max_subtasks]
#   complexities: [1, num_subtasks, 1]
#   stop_probs: [1, num_subtasks, 1]
```

**Output:**
- Sequence of sub-task embeddings
- Task type classifications
- Dependency predictions
- Complexity estimates

**Example Decomposition:**
```
Subtask 0: Extract text from document (type: text_extraction, deps: [])
Subtask 1: Summarize text (type: summarization, deps: [0])
Subtask 2: Create presentation (type: generation, deps: [1])
```

---

### 3. Subtask Embeddings → Workflow Graph

**Input:**
```python
subtask_embeddings  # [1, num_subtasks, 512]
dependencies        # [1, num_subtasks, max_subtasks]
```

**Processing:**
```python
# WorkflowGraphGenerator
graph_outputs = graph_generator(subtask_embeddings, dependencies)
# Outputs:
#   node_embeddings: [1, num_subtasks, 128]
#   edge_probs: [1, num_subtasks, num_subtasks]
#   adjacency: [1, num_subtasks, num_subtasks]  # Binary DAG
```

**Output:**
- Node embeddings (learned representations of tasks)
- Edge probabilities (likelihood of dependencies)
- Adjacency matrix (binary DAG structure)

**Graph Structure:**
```
    0 → 1 → 2
    (Extract → Summarize → Create)
```

---

### 4. Graph + Instruction → Model Selection

**Input:**
```python
instruction_embedding  # [1, 768]
node_embeddings        # [1, num_subtasks, 128]
```

**Processing:**
```python
# RLModelSelector
for each subtask:
    state = concat(instruction_embedding, node_embedding)
    # [1, 768 + 128 = 896]
    
    action, log_prob, entropy = model_selector.select_action(state)
    # action: worker_id (0-7)
```

**Output:**
- Worker selection for each sub-task
- Action probabilities
- Value estimates

**Example Selection:**
```
Subtask 0 → Worker 2 (llama-3-8b)  # Text extraction
Subtask 1 → Worker 0 (gpt-4)       # Summarization (needs quality)
Subtask 2 → Worker 1 (gpt-3.5)    # Generation (cost-effective)
```

---

### 5. Workflow → Execution

**Input:**
```python
model_selections = [2, 0, 1]  # Worker IDs for each subtask
adjacency = [[0, 1, 0],       # DAG structure
             [0, 0, 1],
             [0, 0, 0]]
```

**Processing:**
```python
# OrchestrationSystem
1. Topological sort → execution_order = [0, 1, 2]
2. For each task in order:
   - Check dependencies (all completed?)
   - If ready, execute in parallel (if possible)
   - Collect outputs
3. Combine outputs → final_result
```

**Execution Steps:**
```
Step 1: Execute task 0 (no dependencies)
  → Worker 2 (llama-3-8b)
  → Output: "Extracted text: ..."

Step 2: Execute task 1 (depends on 0)
  → Worker 0 (gpt-4)
  → Input: Output from task 0
  → Output: "Summary: ..."

Step 3: Execute task 2 (depends on 1)
  → Worker 1 (gpt-3.5)
  → Input: Output from task 1
  → Output: "Presentation: ..."
```

**Output:**
```python
ExecutionResult(
    success=True,
    outputs={-1: "Final presentation content"},
    total_cost=0.045,
    total_latency_ms=1200.0,
    task_metrics={
        0: {"cost": 0.001, "latency_ms": 300},
        1: {"cost": 0.030, "latency_ms": 500},
        2: {"cost": 0.014, "latency_ms": 400}
    }
)
```

---

## Training Data Flow

### 1. Execution Logging

**During Execution:**
```python
# OrchestrationSystem.execute()
execution_logger.log_execution(
    instruction="Summarize document...",
    planner_outputs={
        "instruction_embeddings": tensor([...]),
        "decomposition": {...},
        "workflow_graph": {...},
        "model_selections": [2, 0, 1],
        ...
    },
    execution_result={
        "success": True,
        "cost": 0.045,
        "latency_ms": 1200.0,
        ...
    }
)
```

**Log File Format (JSONL):**
```json
{
  "timestamp": "2024-12-24T10:30:00",
  "instruction": "Summarize document...",
  "success": true,
  "cost": 0.045,
  "latency_ms": 1200.0,
  "planner_outputs": {
    "model_selections": [2, 0, 1],
    "workflow_graph": {
      "adjacency": [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    },
    ...
  },
  "execution_result": {...}
}
```

---

### 2. Training Data Preparation

**Input:** JSONL log files

**Processing:**
```python
# scripts/prepare_training_data.py
for each log entry:
    1. Extract instruction
    2. Extract planner outputs (model_selections, adjacency, etc.)
    3. Extract execution results (success, cost, latency)
    4. Create training example:
       {
           "instruction": "...",
           "targets": {
               "model_selections": [2, 0, 1],
               "task_types": [3, 5, 7],
               "dependencies": [[], [0], [1]]
           },
           "rewards": {
               "success": 1.0,
               "cost": -0.045,
               "latency": -1.2
           }
       }
```

**Output:** `training_data.json`
```json
[
  {
    "instruction": "Summarize document...",
    "targets": {
      "model_selections": [2, 0, 1],
      "task_types": [3, 5, 7],
      "dependencies": [[], [0], [1]]
    },
    "rewards": {...}
  },
  ...
]
```

---

### 3. Supervised Training (Phase 1)

**Input:** Training dataset

**Processing:**
```python
# SupervisedTrainer
for batch in dataloader:
    # Forward pass
    outputs = planner(batch["instruction"])
    
    # Compute loss
    loss = planner.compute_supervised_loss(
        outputs,
        targets={
            "task_types": batch["task_types"],
            "model_selections": batch["model_selections"]
        }
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

**Loss Components:**
- **CE Loss**: Cross-entropy for task types and model selections
- **DAG Loss**: Penalty for invalid graph structures
- **Total Loss**: `ce_loss + lambda_dag * dag_loss`

---

### 4. RL Training (Phase 2)

**Input:** Execution environment (orchestrator)

**Processing:**
```python
# RLTrainer
for episode in range(num_episodes):
    # Collect trajectory
    trajectory = []
    for step in range(max_steps):
        # Execute workflow
        result = orchestrator.execute(instruction)
        
        # Compute reward
        reward = compute_reward(result)
        
        # Store transition
        trajectory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state
        })
    
    # Compute advantages (GAE)
    advantages = compute_gae(trajectory)
    
    # Update policy (PPO)
    loss = model_selector.compute_loss(
        states, actions, old_log_probs,
        advantages, returns
    )
    loss.backward()
    optimizer.step()
```

**Reward Computation:**
```python
reward = (
    10.0 * success +           # +10 if succeeds
    -0.1 * total_cost +       # Cost penalty
    -0.01 * latency_ms +      # Latency penalty
    0.1 * num_successful_tasks # Shaped reward
)
```

---

## Component Data Formats

### InstructionEncoder Output

```python
{
    "embeddings": torch.Tensor,  # [batch_size, hidden_size]
    # Example: [1, 768]
}
```

### TaskDecomposer Output

```python
{
    "subtask_embeddings": torch.Tensor,  # [batch, num_subtasks, hidden_size]
    "task_types": torch.Tensor,          # [batch, num_subtasks, num_task_types]
    "dependencies": torch.Tensor,        # [batch, num_subtasks, max_subtasks]
    "complexities": torch.Tensor,        # [batch, num_subtasks, 1]
    "stop_probs": torch.Tensor,          # [batch, num_subtasks, 1]
}
```

### WorkflowGraphGenerator Output

```python
{
    "node_embeddings": torch.Tensor,  # [batch, num_subtasks, output_dim]
    "edge_probs": torch.Tensor,       # [batch, num_subtasks, num_subtasks]
    "adjacency": torch.Tensor,         # [batch, num_subtasks, num_subtasks] (binary)
    "graphs": Optional[List[nx.DiGraph]],  # NetworkX graphs if return_graph=True
}
```

### RLModelSelector Output

```python
{
    "action_logits": torch.Tensor,   # [batch, action_dim]
    "action_probs": torch.Tensor,    # [batch, action_dim]
    "value": torch.Tensor,           # [batch, 1]
}
```

### WorkerOutput Format

```python
@dataclass
class WorkerOutput:
    content: Any                     # Task output (str, dict, etc.)
    metadata: Dict[str, Any]         # Additional info
    cost: float                      # Execution cost
    latency_ms: float                # Execution time in milliseconds
    success: bool                    # Whether task succeeded
    error: Optional[str]             # Error message if failed
```

### ExecutionResult Format

```python
@dataclass
class ExecutionResult:
    success: bool                    # Overall success
    outputs: Dict[int, Any]         # Sub-task outputs
    total_cost: float                # Total execution cost
    total_latency_ms: float          # Total execution time
    workflow_graph: Optional[Any]    # Workflow graph if requested
    error: Optional[str]             # Error message if failed
    retry_count: int                 # Number of retries
    task_metrics: Dict[int, Dict]    # Per-task metrics
```

---

## Tensor Shapes and Dimensions

### Common Dimensions

- **batch_size**: Usually 1 for inference, 32 for training
- **hidden_size**: 768 (BERT) or 512 (T5)
- **num_subtasks**: Variable, typically 2-10
- **max_subtasks**: 20 (maximum allowed)
- **num_task_types**: 10 (predefined task types)
- **action_dim**: 8 (number of worker models)
- **output_dim**: 128 (graph generator output)

### Shape Progression

```
Instruction: str
    ↓
InstructionEncoder: [batch, hidden_size]
    ↓
TaskDecomposer: [batch, num_subtasks, hidden_size]
    ↓
WorkflowGraphGenerator: [batch, num_subtasks, output_dim]
    ↓
RLModelSelector: [batch, action_dim] (per subtask)
    ↓
Execution: WorkerOutput (per subtask)
    ↓
Final Result: ExecutionResult
```

### Batch Processing

During training, batches are processed:
```python
# Batch of 32 instructions
instructions = ["task 1", "task 2", ..., "task 32"]  # List of 32 strings

# After encoding
embeddings = encoder(instructions)  # [32, 768]

# After decomposition
subtask_embeddings = decomposer(embeddings)  # [32, num_subtasks, 512]
# Note: num_subtasks may vary per instruction, so padding is used
```

---

## Data Serialization

### Tensor to JSON

PyTorch tensors are serialized to JSON-compatible formats:

```python
def _serialize_tensors(data):
    if isinstance(data, torch.Tensor):
        return data.tolist()  # Convert to Python list
    if isinstance(data, dict):
        return {k: _serialize_tensors(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize_tensors(elem) for elem in data]
    return data
```

**Example:**
```python
# Before serialization
tensor = torch.tensor([1, 2, 3])  # torch.Tensor

# After serialization
json_data = [1, 2, 3]  # Python list (JSON-compatible)
```

---

## Next Steps

- See `05_TRAINING_PIPELINE.md` for detailed training documentation
- See `06_CONFIGURATION_GUIDE.md` for configuration details
- See `07_API_INTEGRATION.md` for API integration documentation

