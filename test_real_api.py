#!/usr/bin/env python3
"""
Test OrchestAI with real API connections
"""

import os
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

print("=" * 60)
print("ORCHESTAI REAL API TEST")
print("=" * 60)

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("\n⚠️  WARNING: OPENAI_API_KEY not set")
    print("   Set it with: export OPENAI_API_KEY='sk-...'")
    print("   Will use mock responses\n")
else:
    print(f"\n✅ OpenAI API key found\n")

# Load config
config = load_config("config.yaml")

# Setup system
orchestrator = setup_system(config)

# Test with real instruction
print("[Test] Executing with real API...")
result = orchestrator.execute(
    instruction="Summarize this text: Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    input_data={"text": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."}
)

print(f"\n✅ Execution completed")
print(f"   Success: {result.success}")
print(f"   Cost: ${result.total_cost:.4f}")
print(f"   Latency: {result.total_latency_ms:.2f} ms")
print(f"   Outputs: {len(result.outputs)} outputs")

if result.outputs:
    print(f"\n📄 Output preview:")
    for key, value in list(result.outputs.items())[:2]:
        output_str = str(value)[:200]
        print(f"   {key}: {output_str}...")

if result.error:
    print(f"\n⚠️  Error: {result.error}")

print(f"\n📊 Execution logged to: execution_logs/")
print(f"   Use scripts/prepare_training_data.py to convert logs to training format")

