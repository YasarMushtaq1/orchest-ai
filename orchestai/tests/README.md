# OrchestAI Test Suite

This directory contains test scripts and utilities for testing OrchestAI components.

## Test Files

### `test_phase1_model.py`
Comprehensive test suite for the Phase 1 trained model.

**Usage:**
```bash
# Run full test suite
python orchestai/tests/test_phase1_model.py

# Test specific capabilities
python orchestai/tests/test_phase1_model.py --capabilities
```

**What it tests:**
- Task decomposition
- Workflow graph generation
- Model selection
- Execution success
- Cost and latency metrics

### `load_phase1_model.py`
Simple script to load and test the Phase 1 model with basic examples.

**Usage:**
```bash
python orchestai/tests/load_phase1_model.py
```

### `TESTING_PHASE1_MODEL.md`
Complete testing guide with instructions, examples, and troubleshooting.

## Running Tests

From the project root:

```bash
cd "/Users/yasar/Documents/work/orchestros ai"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
source venv/bin/activate
python orchestai/tests/test_phase1_model.py
```

## Requirements

- Trained Phase 1 model at `checkpoints/phase1_best_model.pth`
- Configuration file at `config.yaml`
- All dependencies installed in virtual environment

## Test Structure

```
orchestai/tests/
├── __init__.py              # Package init
├── test_phase1_model.py     # Main test suite
├── load_phase1_model.py     # Simple loader script
├── TESTING_PHASE1_MODEL.md  # Testing guide
└── README.md                # This file
```


