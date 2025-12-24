# Training Pipeline Documentation

This document explains how OrchestAI is trained, from data collection to model deployment.

## Table of Contents

1. [Training Overview](#training-overview)
2. [Phase 1: Supervised Pre-Training](#phase-1-supervised-pre-training)
3. [Phase 2: RL Fine-Tuning](#phase-2-rl-fine-tuning)
4. [Data Collection](#data-collection)
5. [Training Scripts](#training-scripts)
6. [Evaluation and Monitoring](#evaluation-and-monitoring)

---

## Training Overview

OrchestAI uses a two-phase training approach:

1. **Phase 1: Supervised Pre-Training**
   - Trains on expert-annotated workflows
   - Learns task decomposition and model selection
   - Provides good initialization for RL

2. **Phase 2: RL Fine-Tuning**
   - Fine-tunes using reinforcement learning
   - Optimizes for success rate, cost, and latency
   - Learns from execution feedback

**Why Two Phases?**
- Supervised learning is faster and more stable
- RL requires good initialization to converge
- Best of both worlds: efficiency + optimization

---

## Phase 1: Supervised Pre-Training

### Objective

Learn to decompose tasks and select models from expert demonstrations.

### Data Requirements

- **Instructions**: Natural language task descriptions
- **Expert Workflows**: Manually annotated task decompositions
- **Model Selections**: Expert-chosen worker models for each subtask
- **Dependencies**: Expert-annotated task dependencies

### Training Process

**1. Data Loading**
```python
dataset = WorkflowDataset(
    instructions=instructions,
    subtask_sequences=subtask_sequences,
    task_types=task_types,
    dependencies=dependencies,
    model_selections=model_selections,
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**2. Forward Pass**
```python
# Encode instructions
outputs = planner(instructions)

# Outputs include:
# - subtask_embeddings
# - task_types (predictions)
# - model_selections (predictions)
# - workflow_graph (adjacency matrix)
```

**3. Loss Computation**
```python
loss_dict = planner.compute_supervised_loss(
    outputs,
    targets={
        "task_types": target_task_types,
        "model_selections": target_model_selections,
    },
    lambda_dag=0.3,  # DAG validity weight
)

# Loss components:
# - ce_loss: Cross-entropy for task types and model selections
# - dag_loss: Penalty for invalid DAG structures
# - total_loss: ce_loss + lambda_dag * dag_loss
```

**4. Backward Pass**
```python
loss = loss_dict["total_loss"]
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(planner.parameters(), max_norm=1.0)
optimizer.step()
```

### Loss Components

**Cross-Entropy Loss:**
- Task type classification accuracy
- Model selection accuracy
- Measures how well model predicts expert choices

**DAG Loss:**
- Penalizes self-loops in adjacency matrix
- Penalizes cycles (violations of topological order)
- Ensures valid DAG structure

**Total Loss:**
```python
total_loss = ce_loss + lambda_dag * dag_loss
```

### Training Configuration

```yaml
training:
  phase1_supervised:
    batch_size: 32
    learning_rate: 1e-4
    num_epochs: 50
    lambda_dag: 0.3
    train_data_size: 300
    synthetic_augmentations: 1000
```

### Training Script

```bash
python scripts/train_phase1.py \
    --config config.yaml \
    --data training_data.json \
    --output_dir checkpoints/
```

---

## Phase 2: RL Fine-Tuning

### Objective

Optimize model selection for success rate, cost, and latency using execution feedback.

### Data Requirements

- **Execution Environment**: Orchestrator with real worker models
- **Instructions**: Diverse task instructions
- **Reward Signal**: Success, cost, latency from execution

### Training Process

**1. Episode Collection**
```python
for episode in range(num_episodes):
    trajectory = []
    
    # Execute workflow
    result = orchestrator.execute(instruction)
    
    # Compute reward
    reward = compute_reward(result)
    
    # Store transition
    trajectory.append({
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "log_prob": log_prob,
    })
```

**2. Reward Computation**
```python
def compute_reward(result: ExecutionResult) -> float:
    reward = (
        10.0 * result.success +                    # Success reward
        -0.1 * result.total_cost +                # Cost penalty
        -0.01 * result.total_latency_ms +          # Latency penalty
        0.1 * sum(1 for t in result.outputs if t)  # Shaped reward
    )
    return reward
```

**3. Advantage Estimation (GAE)**
```python
# Generalized Advantage Estimation
advantages = compute_gae(
    rewards=rewards,
    values=values,
    gamma=0.99,
    lambda_gae=0.95,
)
```

**4. PPO Update**
```python
loss = model_selector.compute_loss(
    states=states,
    actions=actions,
    old_log_probs=old_log_probs,
    advantages=advantages,
    returns=returns,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
)

optimizer.zero_grad()
loss["total_loss"].backward()
optimizer.step()
```

### PPO Algorithm

**Clipped Surrogate Objective:**
```python
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantages
surr2 = clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
policy_loss = -min(surr1, surr2).mean()
```

**Value Loss:**
```python
value_loss = mse_loss(values, returns)
```

**Entropy Bonus:**
```python
entropy = dist.entropy().mean()
```

**Total Loss:**
```python
total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

### Training Configuration

```yaml
training:
  phase2_rl:
    algorithm: "PPO"
    learning_rate: 3e-5
    num_episodes: 1000
    max_steps_per_episode: 50
    gamma: 0.99
    epsilon: 0.2
    alpha: 10.0      # Success reward weight
    beta: 0.1       # Cost penalty weight
    gamma_latency: 0.01  # Latency penalty weight
    shaped_reward: 0.1   # Reward per successful subtask
```

### Training Script

```bash
python scripts/train_phase2.py \
    --config config.yaml \
    --checkpoint checkpoints/phase1_best.pth \
    --output_dir checkpoints/
```

---

## Data Collection

### Execution Logging

**During Execution:**
```python
# OrchestrationSystem automatically logs executions
execution_logger.log_execution(
    instruction=instruction,
    planner_outputs=planner_outputs,
    execution_result=execution_result,
)
```

**Log Format (JSONL):**
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
    }
  },
  "execution_result": {
    "success": true,
    "outputs": {...},
    "total_cost": 0.045,
    "total_latency_ms": 1200.0
  }
}
```

### Data Collection Script

**Manual Collection:**
```bash
python test_real_api.py
# Executes single task and logs result
```

**Automated Collection:**
```bash
python scripts/collect_training_data.py \
    --num_executions 200 \
    --output_dir execution_logs/
```

**Script Features:**
- Diverse instruction generation
- Automatic execution
- Logging to JSONL files
- Progress tracking

### Training Data Preparation

**Convert Logs to Training Data:**
```bash
python scripts/prepare_training_data.py \
    --logs execution_logs/ \
    --output training_data.json
```

**Preparation Process:**
1. Load execution logs (JSONL)
2. Extract instructions and planner outputs
3. Extract execution results (success, cost, latency)
4. Create training examples with targets
5. Save to JSON format

**Output Format:**
```json
[
  {
    "instruction": "Summarize document...",
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
  },
  ...
]
```

---

## Training Scripts

### Phase 1 Training

**Script:** `scripts/train_phase1.py`

**Usage:**
```bash
python scripts/train_phase1.py \
    --config config.yaml \
    --data training_data.json \
    --output_dir checkpoints/ \
    --epochs 50 \
    --batch_size 32
```

**Features:**
- Loads training data
- Initializes PlannerModel
- Trains with supervised learning
- Validates on validation set
- Saves checkpoints
- Logs to Weights & Biases (optional)

### Phase 2 Training

**Script:** `scripts/train_phase2.py`

**Usage:**
```bash
python scripts/train_phase2.py \
    --config config.yaml \
    --checkpoint checkpoints/phase1_best.pth \
    --output_dir checkpoints/ \
    --episodes 1000
```

**Features:**
- Loads Phase 1 checkpoint
- Collects trajectories from execution
- Updates policy using PPO
- Logs training metrics
- Saves checkpoints

---

## Evaluation and Monitoring

### Metrics

**Success Rate:**
```python
success_rate = num_successful_executions / total_executions
```

**Average Cost:**
```python
avg_cost = sum(execution.cost for execution in executions) / len(executions)
```

**Average Latency:**
```python
avg_latency = sum(execution.latency_ms for execution in executions) / len(executions)
```

**Workflow Validity:**
```python
valid_workflows = sum(1 for wf in workflows if is_valid_dag(wf)) / len(workflows)
```

### Monitoring

**Weights & Biases Integration:**
```python
import wandb

wandb.init(project="orchestai")
wandb.log({
    "train_loss": loss,
    "val_loss": val_loss,
    "success_rate": success_rate,
    "avg_cost": avg_cost,
})
```

**Checkpoint Saving:**
```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": planner.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_loss": val_loss,
}
torch.save(checkpoint, "checkpoint_best.pth")
```

---

## Training Best Practices

### 1. Data Quality

- **Diverse Instructions**: Cover various task types
- **Expert Annotations**: High-quality workflow labels
- **Balanced Dataset**: Equal representation of task types

### 2. Hyperparameter Tuning

- **Learning Rate**: Start with 1e-4, adjust based on convergence
- **Batch Size**: Larger batches for stability (32-64)
- **Lambda DAG**: Balance between accuracy and DAG validity (0.3)

### 3. Training Stability

- **Gradient Clipping**: Prevent exploding gradients (max_norm=1.0)
- **Learning Rate Scheduling**: Reduce LR on plateau
- **Early Stopping**: Stop if validation loss doesn't improve

### 4. Evaluation

- **Validation Set**: Hold out 20% of data for validation
- **Test Set**: Separate test set for final evaluation
- **Metrics**: Track success rate, cost, latency, workflow validity

---

## Troubleshooting

### Common Issues

**1. Loss Not Decreasing**
- Check data quality
- Reduce learning rate
- Increase batch size
- Check for gradient clipping

**2. Invalid DAGs**
- Increase lambda_dag weight
- Check dependency predictions
- Verify graph generator output

**3. High Cost**
- Adjust reward weights (increase cost penalty)
- Check model selection distribution
- Verify cost tracking

**4. Slow Training**
- Reduce batch size
- Use GPU if available
- Optimize data loading

---

## Next Steps

- See `06_CONFIGURATION_GUIDE.md` for configuration details
- See `07_API_INTEGRATION.md` for API integration
- See `REAL_WORLD_TESTING_GUIDE.md` for real-world testing

