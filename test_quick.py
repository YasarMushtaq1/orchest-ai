#!/usr/bin/env python3
"""
Quick test script for OrchestAI
"""

print("=" * 60)
print("ORCHESTAI QUICK TEST")
print("=" * 60)

# Test 1: Import
print("\n[Test 1] Testing imports...")
try:
    import orchestai
    print("✅ Import works")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Load config
print("\n[Test 2] Loading configuration...")
try:
    from orchestai.utils.config_loader import load_config
    config = load_config("config.yaml")
    print("✅ Config loaded successfully")
except Exception as e:
    print(f"❌ Config load failed: {e}")
    exit(1)

# Test 3: Setup system
print("\n[Test 3] Setting up system...")
try:
    from orchestai.utils.setup import setup_system
    orchestrator = setup_system(config)
    print("✅ System setup complete")
except Exception as e:
    print(f"❌ System setup failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Execute simple task
print("\n[Test 4] Executing simple task...")
try:
    result = orchestrator.execute(
        instruction="Summarize this text: Machine learning is a subset of artificial intelligence.",
        input_data={"text": "Machine learning is a subset of artificial intelligence."}
    )
    print(f"✅ Execution completed")
    print(f"   Success: {result.success}")
    print(f"   Cost: ${result.total_cost:.4f}")
    print(f"   Latency: {result.total_latency_ms:.2f} ms")
    print(f"   Outputs: {len(result.outputs)} outputs")
    if result.error:
        print(f"   Warning: {result.error}")
except Exception as e:
    print(f"❌ Execution failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test components individually
print("\n[Test 5] Testing individual components...")

# Test Instruction Encoder
try:
    from orchestai.planner.instruction_encoder import InstructionEncoder
    encoder = InstructionEncoder(model_name="t5-base", hidden_size=768)
    test_result = encoder.forward(["Test instruction"])
    print(f"✅ Instruction Encoder: Embedding shape {test_result['embeddings'].shape}")
except Exception as e:
    print(f"⚠️  Instruction Encoder: {e}")

# Test Task Decomposer
try:
    from orchestai.planner.task_decomposer import TaskDecomposer
    import torch
    decomposer = TaskDecomposer(input_dim=768, hidden_size=512, max_subtasks=20)
    test_emb = torch.randn(1, 768)
    decomp_output = decomposer(test_emb)
    print(f"✅ Task Decomposer: Generated {decomp_output['subtask_embeddings'].shape[1]} subtasks")
except Exception as e:
    print(f"⚠️  Task Decomposer: {e}")

# Test Graph Generator
try:
    from orchestai.planner.graph_generator import WorkflowGraphGenerator
    graph_gen = WorkflowGraphGenerator(input_dim=512, hidden_dim=256, output_dim=128)
    test_embs = torch.randn(1, 3, 512)
    test_deps = torch.randn(1, 3, 20)
    graph_output = graph_gen(test_embs, test_deps)
    print(f"✅ Graph Generator: Generated graph with {graph_output['adjacency'].shape[1]} nodes")
except Exception as e:
    print(f"⚠️  Graph Generator: {e}")

# Test Model Selector
try:
    from orchestai.planner.model_selector import RLModelSelector
    selector = RLModelSelector(state_dim=896, action_dim=8, hidden_dims=[256, 128])
    test_state = torch.randn(1, 896)
    action, _, _ = selector.select_action(test_state)
    print(f"✅ Model Selector: Selected model {action.item()}")
except Exception as e:
    print(f"⚠️  Model Selector: {e}")

# Test Worker Layer
try:
    from orchestai.worker.worker_layer import WorkerModelLayer
    worker_configs = [
        {"name": "test-llm", "model_type": "llm", "cost_per_token": 0.002, "latency_ms": 200}
    ]
    worker_layer = WorkerModelLayer(worker_configs)
    workers = worker_layer.list_workers()
    print(f"✅ Worker Layer: {len(workers)} workers available")
except Exception as e:
    print(f"⚠️  Worker Layer: {e}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)

