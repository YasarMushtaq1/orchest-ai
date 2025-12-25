# Model Checkpoints

This directory contains trained model checkpoints for OrchestAI.

## Available Checkpoints

### Phase 1: Supervised Pre-Training

**File:** `phase1_best_model.pth`  
**Size:** ~1.3 GB  
**Training Date:** December 2025  
**Epoch:** 12 (best validation loss)  
**Validation Loss:** 2.0954  
**Training Loss:** 2.04  

**Model Capabilities:**
- ✅ Task decomposition (breaks complex instructions into sub-tasks)
- ✅ Workflow graph generation (100% valid DAGs)
- ✅ Model selection (routes tasks to appropriate workers)
- ✅ Dependency handling (models task dependencies)

---

## How to Load and Use the Trained Model

### Method 1: Using the Setup System (Recommended)

```python
import torch
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

# Load configuration
config = load_config("config.yaml")

# Setup system (creates planner, worker layer, orchestrator)
orchestrator = setup_system(config)

# Load trained Phase 1 model
checkpoint_path = "checkpoints/phase1_best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Load model weights
orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])

# Set model to evaluation mode
orchestrator.planner.eval()

print(f"✅ Loaded Phase 1 model from epoch {checkpoint['epoch']}")
print(f"   Validation loss: {checkpoint['val_loss']:.4f}")

# Now use the trained model
result = orchestrator.execute(
    instruction="Summarize this document and create a presentation",
    input_data={"text": "Your document content here..."}
)

print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost:.4f}")
print(f"Latency: {result.total_latency_ms:.2f} ms")
print(f"Output: {result.outputs}")
```

### Method 2: Direct Model Loading

```python
import torch
from orchestai.utils.config_loader import load_config
from orchestai.planner.planner_model import PlannerModel

# Load configuration
config = load_config("config.yaml")
planner_config = config["planner"]

# Create planner model
planner = PlannerModel(
    instruction_encoder_config=planner_config["instruction_encoder"],
    task_decomposer_config=planner_config["task_decomposer"],
    graph_generator_config=planner_config["workflow_graph_generator"],
    model_selector_config=planner_config["model_selector"],
)

# Load checkpoint
checkpoint_path = "checkpoints/phase1_best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Load model weights
planner.load_state_dict(checkpoint["model_state_dict"])
planner.eval()

print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
```

### Method 3: Using a Script

Create a file `load_model.py`:

```python
#!/usr/bin/env python3
"""
Load and test Phase 1 trained model
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

def load_phase1_model(checkpoint_path="checkpoints/phase1_best_model.pth"):
    """Load Phase 1 trained model"""
    
    # Load configuration
    config = load_config("config.yaml")
    
    # Setup system
    orchestrator = setup_system(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model weights
    orchestrator.planner.load_state_dict(checkpoint["model_state_dict"])
    orchestrator.planner.eval()
    
    print(f"✅ Loaded Phase 1 model")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    
    return orchestrator

if __name__ == "__main__":
    # Load model
    orchestrator = load_phase1_model()
    
    # Test with a sample task
    result = orchestrator.execute(
        instruction="Summarize this text: Machine learning is a subset of artificial intelligence.",
        input_data={"text": "Machine learning is a subset of artificial intelligence."}
    )
    
    print(f"\n📊 Execution Result:")
    print(f"   Success: {result.success}")
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Latency: {result.total_latency_ms:.2f} ms")
```

Run it:
```bash
python load_model.py
```

---

## Checkpoint Contents

The checkpoint file contains:

```python
{
    "epoch": 12,                    # Training epoch
    "model_state_dict": {...},      # Model parameters
    "optimizer_state_dict": {...},  # Optimizer state (for resuming)
    "val_loss": 2.0954              # Validation loss
}
```

---

## Model Performance

**Training Metrics:**
- **Initial Loss:** 4.34
- **Final Loss:** 2.04 (train), 2.12 (val)
- **Loss Reduction:** 53%
- **DAG Validity:** 100% (all graphs are valid DAGs)
- **Overfitting:** None (train/val gap: 0.074)

**Model Capabilities:**
- Task decomposition accuracy: Moderate (loss ~2.0)
- DAG generation: Excellent (100% valid)
- Model selection: Moderate (pattern-based)
- Ready for Phase 2 RL fine-tuning

---

## Next Steps

1. **Evaluate Model:**
   - Test on held-out test set
   - Measure accuracy metrics
   - Compare with baseline

2. **Use for Real Tasks:**
   - Deploy for real-world testing
   - Collect execution logs
   - Monitor performance

3. **Phase 2 RL Fine-Tuning:**
   - Use this model as initialization
   - Optimize for success rate, cost, latency
   - Learn from execution feedback

---

## Troubleshooting

### Issue: "File not found"
**Solution:** Make sure you're running from the project root directory:
```bash
cd "/Users/yasar/Documents/work/orchestros ai"
```

### Issue: "CUDA out of memory"
**Solution:** Use CPU instead:
```python
checkpoint = torch.load(checkpoint_path, map_location="cpu")
```

### Issue: "Module not found"
**Solution:** Set PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Notes

- The model was trained on CPU (no GPU)
- Training took ~ 50 minutes for 50 epochs
- Best model was at epoch 12
- Model is ready for Phase 2 RL fine-tuning

