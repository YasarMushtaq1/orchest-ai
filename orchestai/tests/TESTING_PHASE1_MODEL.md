# Testing Phase 1 Trained Model

This guide explains how to test the trained Phase 1 model.

## Quick Start

### Option 1: Run the Test Script (Recommended)

```bash
cd "/Users/yasar/Documents/work/orchestros ai"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
source venv/bin/activate
python orchestai/tests/test_phase1_model.py
```

This will run a comprehensive test suite with multiple test cases.

### Option 2: Test Specific Capabilities

```bash
python orchestai/tests/test_phase1_model.py --capabilities
```

This tests specific model capabilities (task decomposition, workflow generation, model selection).

### Option 3: Use the Load Script

```bash
python orchestai/tests/load_phase1_model.py
```

This loads the model and runs a few basic tests.

---

## Manual Testing

### Step 1: Load the Model

```python
import torch
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

# Load configuration
config = load_config("config.yaml")

# Setup system
orchestrator = setup_system(config)

# Load trained model
checkpoint = torch.load("checkpoints/phase1_best_model.pth", map_location="cpu")
orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])
orchestrator.planner.eval()

print("✅ Model loaded!")
```

### Step 2: Test with a Task

```python
# Execute a task
result = orchestrator.execute(
    instruction="Summarize this text: Machine learning is a subset of AI.",
    input_data={"text": "Machine learning is a subset of AI."}
)

# Check results
print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost:.4f}")
print(f"Latency: {result.total_latency_ms:.2f} ms")
print(f"Outputs: {result.outputs}")
```

---

## What to Test

### 1. Task Decomposition

**Test:** Can the model break down complex tasks?

```python
result = orchestrator.execute(
    instruction="Create a presentation about artificial intelligence",
    input_data={"topic": "artificial intelligence"}
)

# Check if subtasks were created
if result.task_metrics:
    print(f"Model created {len(result.task_metrics)} subtasks")
```

**Expected:** Model should decompose the task into multiple subtasks.

### 2. Workflow Graph Generation

**Test:** Does the model generate valid workflow graphs?

```python
result = orchestrator.execute(
    instruction="Summarize and translate this text",
    input_data={"text": "Sample text"}
)

# Check if workflow graph was generated
if result.workflow_graph:
    print("✅ Workflow graph generated")
```

**Expected:** Model should generate a valid DAG (no cycles).

### 3. Model Selection

**Test:** Does the model select appropriate workers?

```python
result = orchestrator.execute(
    instruction="Process this document",
    input_data={"text": "Document content"}
)

# Check worker selections
if result.task_metrics:
    for task_id, metrics in result.task_metrics.items():
        print(f"Task {task_id}: Worker {metrics.get('worker_id')}")
```

**Expected:** Model should select workers for each subtask.

### 4. Execution Success

**Test:** Do tasks execute successfully?

```python
result = orchestrator.execute(
    instruction="Summarize this text",
    input_data={"text": "Long text here..."}
)

print(f"Success: {result.success}")
print(f"Error: {result.error if result.error else 'None'}")
```

**Expected:** Tasks should execute successfully (success rate: 60-80% expected).

---

## Test Cases

### Simple Tasks

1. **Text Summarization:**
   ```python
   instruction = "Summarize this text: [text]"
   input_data = {"text": "Your text here"}
   ```

2. **Translation:**
   ```python
   instruction = "Translate this to French: [text]"
   input_data = {"text": "Hello, how are you?"}
   ```

3. **Text Analysis:**
   ```python
   instruction = "Analyze this text and extract key points"
   input_data = {"text": "Your text here"}
   ```

### Complex Tasks

1. **Multi-Step Processing:**
   ```python
   instruction = "Summarize this document and create a presentation outline"
   input_data = {"text": "Document content"}
   ```

2. **Sequential Tasks:**
   ```python
   instruction = "Extract text, summarize it, and translate to Spanish"
   input_data = {"text": "Document content"}
   ```

---

## Expected Results

### Performance Metrics

- **Success Rate:** 60-80% (baseline, before Phase 2 RL)
- **DAG Validity:** 100% (all graphs should be valid)
- **Task Decomposition:** Should break tasks into subtasks
- **Model Selection:** Should select workers (may not be optimal yet)

### What to Look For

✅ **Good Signs:**
- Tasks execute successfully
- Workflow graphs are generated
- No cycles in graphs (DAG validity)
- Model selects workers for each task
- Reasonable cost and latency

⚠️ **Areas for Improvement:**
- Higher success rate (will improve with Phase 2 RL)
- Better cost optimization (will improve with Phase 2 RL)
- Better latency optimization (will improve with Phase 2 RL)

---

## Troubleshooting

### Issue: "Checkpoint file not found"

**Solution:**
```bash
# Make sure you're in the project root
cd "/Users/yasar/Documents/work/orchestros ai"
ls checkpoints/phase1_best_model.pth
```

### Issue: "Module not found"

**Solution:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
source venv/bin/activate
```

### Issue: "CUDA out of memory"

**Solution:** The model is set to use CPU by default. If you have GPU issues:
```python
checkpoint = torch.load("checkpoints/phase1_best_model.pth", map_location="cpu")
```

### Issue: Tasks fail to execute

**Possible Reasons:**
1. API key not set (if using real APIs)
2. Model needs more training data
3. Task is too complex
4. Worker models not available

**Solutions:**
- Check API keys in `.env` file
- Test with simpler tasks first
- Verify worker models are configured

---

## Evaluation Metrics

### How to Measure Performance

1. **Success Rate:**
   ```python
   successful = sum(1 for result in results if result.success)
   success_rate = successful / len(results) * 100
   ```

2. **Average Cost:**
   ```python
   avg_cost = sum(r.total_cost for r in results) / len(results)
   ```

3. **Average Latency:**
   ```python
   avg_latency = sum(r.total_latency_ms for r in results) / len(results)
   ```

4. **DAG Validity:**
   ```python
   # Check if workflow graphs are valid DAGs
   # This should be 100% for Phase 1 model
   ```

---

## Next Steps After Testing

1. **Evaluate Results:**
   - Compare with expected performance
   - Identify areas for improvement
   - Document findings

2. **Collect More Data:**
   - Run more test cases
   - Collect execution logs
   - Use for Phase 2 training

3. **Proceed to Phase 2:**
   - Use test results to guide Phase 2 RL
   - Optimize for success rate, cost, latency
   - Fine-tune based on execution feedback

---

## Example Test Output

```
============================================================
Loading Phase 1 Trained Model
============================================================
Setting up system...
Loading checkpoint from: checkpoints/phase1_best_model.pth
✅ Model loaded successfully!
   Epoch: 12
   Validation Loss: 2.0954
   Training Loss: ~2.04

============================================================
Test: Simple Text Summarization
============================================================
Instruction: Summarize this text: Machine learning is...
Input: {'text': 'Machine learning is...'}

✅ Execution completed in 2.34s

📊 Results:
   Success: True
   Cost: $0.0045
   Latency: 1234.56 ms
   Retry Count: 0
   Outputs: 1 outputs
   Workflow Graph: Generated
   Task Metrics: 3 tasks executed
      Task 0: cost=$0.0010, latency=300.00ms
      Task 1: cost=$0.0020, latency=500.00ms
      Task 2: cost=$0.0015, latency=434.56ms
```

---

**Ready to test? Run:**
```bash
python orchestai/tests/test_phase1_model.py
```

