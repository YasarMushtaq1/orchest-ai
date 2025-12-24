# Option 3: Real-World Testing & Training Guide

Complete guide for connecting real APIs, collecting execution logs, and training the OrchestAI planner.

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Step 1: Connect Real APIs](#step-1-connect-real-apis)
4. [Step 2: Run Executions & Collect Data](#step-2-run-executions--collect-data)
5. [Step 3: Prepare Training Data](#step-3-prepare-training-data)
6. [Step 4: Train the Planner](#step-4-train-the-planner)
7. [Step 5: Evaluate & Iterate](#step-5-evaluate--iterate)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Option 3 combines:
- **Real API connections** for immediate testing
- **Automatic execution logging** for data collection
- **Training data preparation** from logs
- **Planner training** for optimization

This approach lets you:
1. Test with real APIs immediately
2. Collect training data automatically
3. Train the planner on your own workflows
4. Improve performance over time

---

## Setup

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
pip install openai  # For real API connections

# Set up environment
export OPENAI_API_KEY='sk-your-key-here'  # Optional but recommended
```

### Verify Setup

```bash
# Test basic functionality
python test_quick.py

# Test with real API
python test_real_api.py
```

---

## Step 1: Connect Real APIs

### OpenAI API Setup

The system automatically uses OpenAI API if `OPENAI_API_KEY` is set.

**Set API Key:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

**Or in Python:**
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

### How It Works

The `LLMWorker` automatically:
- Detects if OpenAI is available
- Uses real API if key is set
- Falls back to mock if no key
- Tracks actual API costs

**Model Mapping:**
- `gpt-4` → GPT-4
- `gpt-3.5-turbo` → GPT-3.5 Turbo
- `gpt-4o` → GPT-4o
- Others → GPT-3.5 Turbo (default)

### Verify API Connection

```python
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

config = load_config("config.yaml")
orchestrator = setup_system(config)

# This will use real API if key is set
result = orchestrator.execute(
    instruction="Summarize: Machine learning is AI.",
    input_data={"text": "Machine learning is AI."}
)

print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost:.4f}")  # Real API cost if connected
```

---

## Step 2: Run Executions & Collect Data

### Automatic Logging

Every execution is **automatically logged** to `execution_logs/` directory.

**Log Format:**
- File: `execution_logs/executions_YYYYMMDD.jsonl`
- Format: JSON Lines (one execution per line)
- Contains: Instruction, planner outputs, execution results, costs

### Run Multiple Executions

```python
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

config = load_config("config.yaml")
orchestrator = setup_system(config)

# Run various tasks to collect diverse data
tasks = [
    "Summarize this text: Machine learning is a subset of AI.",
    "Translate this to French: Hello, how are you?",
    "Generate a presentation about artificial intelligence",
    "Analyze this dataset and create visualizations",
    "Extract key points from this document",
]

for task in tasks:
    result = orchestrator.execute(
        instruction=task,
        input_data={"text": "..."}  # Provide appropriate input
    )
    print(f"Task: {task[:50]}... | Success: {result.success} | Cost: ${result.total_cost:.4f}")
```

### What Gets Logged

Each log entry contains:
```json
{
  "timestamp": "2025-01-XX...",
  "instruction": "User instruction",
  "success": true/false,
  "cost": 0.05,
  "latency_ms": 1234.5,
  "planner_outputs": {
    "model_selections": [...],
    "workflow_graph": {...},
    "decomposition": {...}
  },
  "execution_result": {
    "outputs": {...},
    "task_metrics": {...}
  }
}
```

### Collecting Good Data

**Tips for quality data:**
- Run **diverse tasks** (summarization, translation, generation, etc.)
- Use **real-world scenarios** (not just test cases)
- Include both **successful and failed** executions
- Collect **50-100+ executions** for meaningful training

**Minimum Data:**
- **50 executions**: Basic training
- **200-300 executions**: Good training (as per research proposal)
- **500+ executions**: Excellent training

---

## Step 3: Prepare Training Data

### Convert Logs to Training Format

```bash
python scripts/prepare_training_data.py \
    --log-file execution_logs/executions_20250115.jsonl \
    --output training_data.json
```

**What It Does:**
- Reads execution logs
- Extracts workflow structures
- Converts to training format
- Saves as JSON dataset

### Training Data Format

Output format:
```json
[
  {
    "instruction": "Summarize this text...",
    "subtasks": [
      {
        "id": 0,
        "task_type": 1,
        "dependencies": [],
        "model_selection": 0
      },
      {
        "id": 1,
        "task_type": 2,
        "dependencies": [0],
        "model_selection": 1
      }
    ]
  },
  ...
]
```

### Validate Training Data

```python
from orchestai.training.data_utils import validate_workflow, load_workflow_dataset

# Load dataset
workflows = load_workflow_dataset("training_data.json", split="train")

# Validate each workflow
valid_count = 0
for workflow in workflows:
    if validate_workflow(workflow):
        valid_count += 1

print(f"Valid workflows: {valid_count}/{len(workflows)}")
```

---

## Step 4: Train the Planner

### Phase 1: Supervised Pre-Training

**Purpose:** Learn valid task decompositions and reasonable model selections.

```bash
python scripts/train_phase1.py \
    --config config.yaml \
    --data training_data.json \
    --epochs 50 \
    --batch-size 32 \
    --use-wandb  # Optional: for visualization
```

**What Happens:**
1. Loads training data
2. Trains Planner Model components:
   - Instruction Encoder
   - Task Decomposer
   - Graph Generator
   - Model Selector
3. Optimizes for:
   - Valid DAG generation
   - Correct task decomposition
   - Reasonable model selection
4. Saves checkpoints: `checkpoint_best.pth`

**Expected Results:**
- Success rate: 60-70% on validation set
- Valid DAGs: 100% (no cycles)
- Training time: 2-4 hours (depending on data size)

### Phase 2: Reinforcement Learning Fine-Tuning

**Purpose:** Optimize for success, cost, and latency.

```bash
python scripts/train_phase2.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --episodes 1000 \
    --use-wandb  # Optional
```

**What Happens:**
1. Loads Phase 1 checkpoint
2. Runs RL training with PPO:
   - Generates plans
   - Executes workflows
   - Computes rewards (success - cost - latency)
   - Updates policy
3. Optimizes for:
   - Task success rate
   - Cost minimization
   - Latency reduction
4. Saves final model

**Expected Results:**
- Success rate: 75-85% (improved from Phase 1)
- Cost reduction: 15-20% vs. baseline
- Better model selection

### Monitor Training

**With Weights & Biases:**
```bash
# Install wandb
pip install wandb
wandb login

# Run training with --use-wandb flag
python scripts/train_phase1.py --use-wandb ...
```

**Metrics to Watch:**
- Loss (should decrease)
- Validation loss (should track training loss)
- Task success rate (should increase)
- DAG validity (should be 100%)

---

## Step 5: Evaluate & Iterate

### Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --tasks benchmark_tasks.json \
    --output evaluation_results.json
```

### Compare Before/After

```python
from orchestai.evaluation.benchmark import BenchmarkSuite

# Before training (untrained model)
baseline_results = benchmark.run_benchmark(tasks)

# After training (trained model)
trained_results = benchmark.run_benchmark(tasks)

# Compare
improvement = (
    trained_results["orchestai"]["metrics"]["task_success_rate"] -
    baseline_results["orchestai"]["metrics"]["task_success_rate"]
)
print(f"Success rate improvement: {improvement*100:.1f}%")
```

### Iterate

**If results are good:**
- Deploy trained model
- Continue collecting logs
- Periodically retrain with new data

**If results need improvement:**
- Collect more diverse data
- Adjust training hyperparameters
- Try different architectures
- Add more training epochs

---

## Complete Workflow Example

### Full Pipeline

```bash
# 1. Setup
export OPENAI_API_KEY='sk-...'
pip install -r requirements.txt

# 2. Run executions (collect data)
python test_real_api.py
# Run multiple times with different tasks
# Logs saved to execution_logs/

# 3. Prepare training data
python scripts/prepare_training_data.py \
    --log-file execution_logs/executions_20250115.jsonl \
    --output training_data.json

# 4. Train Phase 1
python scripts/train_phase1.py \
    --config config.yaml \
    --data training_data.json \
    --epochs 50

# 5. Train Phase 2 (optional)
python scripts/train_phase2.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --episodes 1000

# 6. Evaluate
python scripts/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoint_best.pth \
    --tasks benchmark_tasks.json
```

### Python Script Example

```python
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system
import os

# Setup
os.environ["OPENAI_API_KEY"] = "sk-your-key"
config = load_config("config.yaml")
orchestrator = setup_system(config)

# Collect data
tasks = [
    "Summarize: Machine learning is AI.",
    "Translate to French: Hello world",
    "Generate a presentation about AI",
]

for task in tasks:
    result = orchestrator.execute(
        instruction=task,
        input_data={"text": "..."}
    )
    print(f"Logged: {task[:30]}... | Success: {result.success}")

# After collecting, convert logs
# python scripts/prepare_training_data.py --log-file execution_logs/... --output training_data.json

# Then train
# python scripts/train_phase1.py --config config.yaml --data training_data.json
```

---

## Troubleshooting

### Issue: API Not Working

**Symptoms:** Still using mock responses

**Solutions:**
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Set it if missing
export OPENAI_API_KEY='sk-...'

# Verify in Python
import os
print(os.getenv("OPENAI_API_KEY"))
```

### Issue: No Logs Being Created

**Symptoms:** `execution_logs/` directory empty

**Solutions:**
- Check write permissions: `chmod -R 755 execution_logs/`
- Verify orchestrator is logging: Check `orchestrator.execution_logger` is not None
- Check disk space

### Issue: Training Data Conversion Fails

**Symptoms:** `prepare_training_data.py` errors

**Solutions:**
- Check log file exists and is readable
- Verify log format (should be JSONL)
- Check for successful executions in logs (only successful ones are converted)

### Issue: Training Fails

**Symptoms:** Training script crashes

**Solutions:**
- Verify training data format is correct
- Check GPU/memory availability
- Reduce batch size if OOM errors
- Check config.yaml settings

### Issue: Low Success Rate After Training

**Symptoms:** Trained model performs poorly

**Solutions:**
- Collect more training data (200+ examples)
- Ensure data quality (diverse, real-world tasks)
- Train for more epochs
- Check validation loss isn't increasing (overfitting)

---

## Best Practices

### Data Collection

1. **Diversity**: Collect various task types
2. **Quality**: Use real-world scenarios
3. **Volume**: Aim for 200-300+ executions
4. **Balance**: Mix simple and complex tasks

### Training

1. **Start Small**: Begin with 50 examples, scale up
2. **Monitor**: Watch training metrics closely
3. **Validate**: Always validate on held-out data
4. **Iterate**: Improve based on evaluation results

### Evaluation

1. **Benchmark**: Use consistent benchmark tasks
2. **Compare**: Compare before/after training
3. **Metrics**: Track success rate, cost, latency
4. **Real-World**: Test on actual use cases

---

## Expected Timeline

### Data Collection Phase
- **50 executions**: 1-2 hours
- **200 executions**: 4-8 hours
- **300 executions**: 6-12 hours

### Training Phase
- **Phase 1 (50 epochs)**: 2-4 hours
- **Phase 2 (1000 episodes)**: 4-8 hours
- **Total**: 6-12 hours

### Evaluation Phase
- **Benchmark evaluation**: 30 minutes
- **Analysis**: 1 hour

**Total Time:** ~1-2 days for complete pipeline

---

## Summary

**Option 3 Workflow:**
1. ✅ Connect real APIs (automatic if API key set)
2. ✅ Run executions (logs collected automatically)
3. ✅ Convert logs to training data (`prepare_training_data.py`)
4. ✅ Train Phase 1 (supervised pre-training)
5. ✅ Train Phase 2 (RL fine-tuning) - optional
6. ✅ Evaluate and iterate

**Key Files:**
- `test_real_api.py` - Test with real API
- `scripts/prepare_training_data.py` - Convert logs
- `scripts/train_phase1.py` - Phase 1 training
- `scripts/train_phase2.py` - Phase 2 training
- `execution_logs/` - Collected data

**Result:** A trained OrchestAI planner optimized for your specific use cases!

---

*Last Updated: 2025*

