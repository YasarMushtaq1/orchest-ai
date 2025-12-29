# OrchestAI Documentation

Welcome to the OrchestAI documentation! This directory contains comprehensive documentation explaining what OrchestAI does, how it works, why design decisions were made, and how to use it.

## Documentation Index

### 1. [Architecture Overview](01_ARCHITECTURE_OVERVIEW.md)
**What it covers:**
- High-level system architecture
- Core components and their roles
- Data flow through the system
- Key design decisions
- System capabilities and limitations

**Read this first** to understand the overall system design.

---

### 2. [Core Components](02_CORE_COMPONENTS.md)
**What it covers:**
- Detailed documentation of each component
- What each component does
- Why each component exists
- How each component works
- Component interactions

**Read this** to understand individual components in detail.

---

### 3. [Design Decisions](03_DESIGN_DECISIONS.md)
**What it covers:**
- Rationale behind architectural decisions
- Why certain technologies were chosen
- Trade-offs and alternatives considered
- Implementation decisions
- Lessons learned

**Read this** to understand the "why" behind design choices.

---

### 4. [Data Flow](04_DATA_FLOW.md)
**What it covers:**
- Execution flow from input to output
- Training data flow
- Component data formats
- Tensor shapes and dimensions
- Data serialization

**Read this** to understand how data moves through the system.

---

### 5. [Training Pipeline](05_TRAINING_PIPELINE.md)
**What it covers:**
- Phase 1: Supervised pre-training
- Phase 2: RL fine-tuning
- Data collection and preparation
- Training scripts and configuration
- Evaluation and monitoring

**Read this** to understand how to train OrchestAI.

---

### 6. [Configuration Guide](06_CONFIGURATION_GUIDE.md)
**What it covers:**
- Configuration file structure
- Planner configuration
- Worker configuration
- Training configuration
- System configuration
- Environment variables

**Read this** to configure OrchestAI for your use case.

---

### 7. [API Integration](07_API_INTEGRATION.md)
**What it covers:**
- OpenAI API integration
- API key management
- Cost tracking
- Error handling
- Mock mode
- Adding new APIs

**Read this** to understand API integration and usage.

---

## Quick Start Guide

### For Understanding the System

1. Start with [Architecture Overview](01_ARCHITECTURE_OVERVIEW.md) - Get the big picture
2. Read [Core Components](02_CORE_COMPONENTS.md) - Understand individual pieces
3. Review [Design Decisions](03_DESIGN_DECISIONS.md) - Understand the rationale

### For Using the System

1. Read [Configuration Guide](06_CONFIGURATION_GUIDE.md) - Configure the system
2. Review [API Integration](07_API_INTEGRATION.md) - Set up API keys
3. Check [Training Pipeline](05_TRAINING_PIPELINE.md) - Train the models

### For Development

1. Study [Data Flow](04_DATA_FLOW.md) - Understand data formats
2. Review [Design Decisions](03_DESIGN_DECISIONS.md) - Understand implementation choices
3. Check [Core Components](02_CORE_COMPONENTS.md) - Understand component interfaces

---

## Documentation Philosophy

This documentation follows a "why, how, what" structure:

- **Why**: Explains the rationale behind decisions (see [Design Decisions](03_DESIGN_DECISIONS.md))
- **How**: Explains how things work (see [Core Components](02_CORE_COMPONENTS.md) and [Data Flow](04_DATA_FLOW.md))
- **What**: Explains what things do (see [Architecture Overview](01_ARCHITECTURE_OVERVIEW.md))

---

## Key Concepts

### Learned Planning
OrchestAI uses neural networks to learn optimal task decomposition and model selection, rather than using rule-based systems.

### Graph Neural Networks
Workflows are represented as graphs (DAGs), and GNNs are used to generate and reason about these graphs.

### Reinforcement Learning
Model selection is optimized using RL (PPO) to balance success rate, cost, and latency.

### Hybrid Planning
Combines learned planning with LLM fallback for flexibility and robustness.

### Parallel Execution
Independent tasks are executed in parallel to reduce latency.

---

## Related Documentation

- **Main README**: `../README.md` - Project overview and quick start
- **Testing Guide**: `../TESTING_GUIDE.md` - How to test OrchestAI
- **Real-World Testing**: `../REAL_WORLD_TESTING_GUIDE.md` - Real-world testing and data collection
- **Analysis**: `../ANALYSIS.md` - Comparison with other systems (if available)
- **Improvements**: `../IMPROVEMENTS.md` - System improvements (if available)

---

## Contributing to Documentation

When adding new features or making changes:

1. **Update relevant documentation** - Keep docs in sync with code
2. **Explain the "why"** - Document design decisions
3. **Provide examples** - Include code examples and use cases
4. **Update this index** - Add new documents to the index

---

## Questions?

If you have questions about:
- **Architecture**: See [Architecture Overview](01_ARCHITECTURE_OVERVIEW.md)
- **Components**: See [Core Components](02_CORE_COMPONENTS.md)
- **Design**: See [Design Decisions](03_DESIGN_DECISIONS.md)
- **Usage**: See [Configuration Guide](06_CONFIGURATION_GUIDE.md) and [API Integration](07_API_INTEGRATION.md)
- **Training**: See [Training Pipeline](05_TRAINING_PIPELINE.md)

---

## Document Status

All documentation is up-to-date as of the latest codebase. Documentation is maintained alongside code changes to ensure accuracy.

**Last Updated**: December 2025

