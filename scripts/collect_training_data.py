#!/usr/bin/env python3
"""
Automated data collection script for OrchestAI training
Runs multiple diverse tasks and collects execution logs
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from orchestai.utils.config_loader import load_config
from orchestai.utils.setup import setup_system

# Diverse task templates for data collection
TASK_TEMPLATES = [
    # Summarization tasks
    ("Summarize this text: {}", "text"),
    ("Create a brief summary of: {}", "text"),
    ("Provide a concise summary: {}", "text"),
    
    # Translation tasks
    ("Translate this to French: {}", "text"),
    ("Translate this to Spanish: {}", "text"),
    ("Translate this to German: {}", "text"),
    
    # Generation tasks
    ("Generate a presentation about {}", "topic"),
    ("Create a document about {}", "topic"),
    ("Write an article about {}", "topic"),
    
    # Analysis tasks
    ("Analyze this text: {}", "text"),
    ("Extract key points from: {}", "text"),
    ("Identify main ideas in: {}", "text"),
    
    # Complex multi-step tasks
    ("Generate a presentation about {} and create visualizations", "topic"),
    ("Summarize {} and translate to French", "text"),
    ("Extract information from {} and create a report", "text"),
]

# Sample data
SAMPLE_TEXTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Natural language processing enables computers to understand and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns.",
    "Reinforcement learning trains agents to make decisions through trial and error.",
]

SAMPLE_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "robotics",
    "data science",
    "neural networks",
]

def collect_data(num_executions=50):
    """
    Collect training data by running diverse tasks.
    
    Args:
        num_executions: Number of executions to run
    """
    print("=" * 60)
    print("ORCHESTAI AUTOMATED DATA COLLECTION")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY not set")
        print("   Will use mock responses (less useful for training)")
    else:
        print(f"\n✅ OpenAI API key found - using real API\n")
    
    # Setup system
    print("Setting up OrchestAI system...")
    config = load_config("config.yaml")
    orchestrator = setup_system(config)
    print("✅ System ready\n")
    
    # Collect data
    print(f"Running {num_executions} executions to collect training data...")
    print("-" * 60)
    
    successful = 0
    failed = 0
    total_cost = 0.0
    
    for i in range(num_executions):
        # Select random task template
        import random
        template, data_type = random.choice(TASK_TEMPLATES)
        
        # Fill template with sample data
        if data_type == "text":
            data = random.choice(SAMPLE_TEXTS)
            instruction = template.format(data)
            input_data = {"text": data}
        else:  # topic
            topic = random.choice(SAMPLE_TOPICS)
            instruction = template.format(topic)
            input_data = {"topic": topic}
        
        # Execute
        try:
            result = orchestrator.execute(
                instruction=instruction,
                input_data=input_data
            )
            
            if result.success:
                successful += 1
                status = "✅"
            else:
                failed += 1
                status = "❌"
            
            total_cost += result.total_cost
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{num_executions}] {status} Success: {successful}, Failed: {failed}, Cost: ${total_cost:.4f}")
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            failed += 1
            print(f"Error on execution {i+1}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Total executions: {num_executions}")
    print(f"Successful: {successful} ({successful/num_executions*100:.1f}%)")
    print(f"Failed: {failed} ({failed/num_executions*100:.1f}%)")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"\n📊 Logs saved to: execution_logs/")
    print(f"   Next step: python scripts/prepare_training_data.py --log-file execution_logs/executions_*.jsonl --output training_data.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect training data automatically")
    parser.add_argument("--num-executions", type=int, default=50, help="Number of executions to run")
    args = parser.parse_args()
    
    collect_data(args.num_executions)

