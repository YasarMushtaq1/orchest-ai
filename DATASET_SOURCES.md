# Dataset Sources for OrchestAI Training

This document lists sources where you can find or download existing datasets for training OrchestAI.

## Option 1: Use Existing Datasets

### HuggingFace Datasets

**Search for workflow/task datasets:**
- https://huggingface.co/datasets
- Search terms: "workflow", "task planning", "multi-step", "orchestration"

**Potential datasets:**
- `bigscience/P3` - Diverse prompts and tasks
- `allenai/prosocial-dialog` - Multi-turn conversations
- `allenai/ai2_arc` - Multi-step reasoning tasks

### GitHub Repositories

**AutoGen Examples:**
- https://github.com/microsoft/autogen/tree/main/python/samples
- Contains workflow examples and multi-agent scenarios

**LangChain Workflows:**
- https://github.com/langchain-ai/langchain
- Contains workflow examples and agent patterns

**HuggingGPT/TaskBench:**
- https://github.com/microsoft/JARVIS
- Contains task planning datasets

### Academic Datasets

**Research Papers with Datasets:**
- HuggingGPT paper: Task planning datasets
- AutoGen paper: Multi-agent workflow examples
- ReAct paper: Tool use datasets

## Option 2: Generate Synthetic Data

Use the synthetic data generator:

```bash
python scripts/download_datasets.py --method synthetic --num-examples 200 --output synthetic_data.json
```

This creates synthetic workflows based on templates.

## Option 3: Collect Your Own Data

Use the automated collection script:

```bash
python scripts/collect_training_data.py --num-executions 50
```

This runs diverse tasks and collects execution logs automatically.

## Recommended Approach

**Best: Combine all three**
1. Start with synthetic data (quick, 200 examples)
2. Collect your own data (real-world, 50-100 examples)
3. Supplement with existing datasets if available

**Minimum for training:**
- 50 examples: Basic training
- 200 examples: Good training
- 300+ examples: Excellent training

