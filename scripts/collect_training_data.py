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

# Diverse task templates for data collection - 200 TEMPLATES for maximum diversity
TASK_TEMPLATES = [
    # Summarization tasks (simple) - 10 variants (reduced to reach 200)
    ("Summarize this text: {}", "text"),
    ("Create a brief summary of: {}", "text"),
    ("Provide a concise summary: {}", "text"),
    ("Write a summary: {}", "text"),
    ("Give me a summary of: {}", "text"),
    ("Condense this into a summary: {}", "text"),
    ("Summarize the following: {}", "text"),
    ("Create an executive summary of: {}", "text"),
    ("Summarize the key points: {}", "text"),
    ("Create a condensed version: {}", "text"),
    
    # Translation tasks - 15 languages (reduced to reach 200)
    ("Translate this to French: {}", "text"),
    ("Translate this to Spanish: {}", "text"),
    ("Translate this to German: {}", "text"),
    ("Translate this to Italian: {}", "text"),
    ("Translate this to Chinese: {}", "text"),
    ("Translate this to Japanese: {}", "text"),
    ("Translate this to Portuguese: {}", "text"),
    ("Translate this to Russian: {}", "text"),
    ("Translate this to Arabic: {}", "text"),
    ("Translate this to Hindi: {}", "text"),
    ("Translate this to Korean: {}", "text"),
    ("Translate this to Dutch: {}", "text"),
    ("Translate this to Polish: {}", "text"),
    ("Translate this to Turkish: {}", "text"),
    ("Translate this to Swedish: {}", "text"),
    
    # Generation tasks (single-step) - 15 variants (reduced to reach 200)
    ("Generate a presentation about {}", "topic"),
    ("Create a document about {}", "topic"),
    ("Write an article about {}", "topic"),
    ("Create a report on {}", "topic"),
    ("Write a blog post about {}", "topic"),
    ("Generate content about {}", "topic"),
    ("Create a paper about {}", "topic"),
    ("Write a guide on {}", "topic"),
    ("Generate a tutorial about {}", "topic"),
    ("Create a handbook for {}", "topic"),
    ("Write a whitepaper on {}", "topic"),
    ("Create a case study about {}", "topic"),
    ("Generate a proposal for {}", "topic"),
    ("Write a review of {}", "topic"),
    ("Create a newsletter about {}", "topic"),
    
    # Analysis tasks - 15 variants (reduced to reach 200)
    ("Analyze this text: {}", "text"),
    ("Extract key points from: {}", "text"),
    ("Identify main ideas in: {}", "text"),
    ("Find the key concepts in: {}", "text"),
    ("What are the main points in: {}", "text"),
    ("Break down this text: {}", "text"),
    ("Analyze the structure of: {}", "text"),
    ("Examine this content: {}", "text"),
    ("Review this text: {}", "text"),
    ("Evaluate this: {}", "text"),
    ("Assess this content: {}", "text"),
    ("Critique this text: {}", "text"),
    ("Interpret this: {}", "text"),
    ("Examine the meaning of: {}", "text"),
    ("Study this text: {}", "text"),
    
    # Question answering - 10 variants (reduced to reach 200)
    ("Answer this question: {}", "text"),
    ("Explain: {}", "text"),
    ("What does this mean: {}", "text"),
    ("Clarify: {}", "text"),
    ("Define: {}", "text"),
    ("Describe: {}", "text"),
    ("Elaborate on: {}", "text"),
    ("What is: {}", "text"),
    ("How does this work: {}", "text"),
    ("Why is this important: {}", "text"),
    
    # Complex multi-step tasks (2-3 steps) - 15 variants (reduced to reach 200)
    ("Summarize {} and translate to French", "text"),
    ("Extract information from {} and create a report", "text"),
    ("Analyze {} and generate a summary", "text"),
    ("Summarize {} and identify key points", "text"),
    ("Translate {} to Spanish and summarize", "text"),
    ("Analyze {} and extract main ideas", "text"),
    ("Summarize {} and create an outline", "text"),
    ("Translate {} and provide context", "text"),
    ("Analyze {} and suggest improvements", "text"),
    ("Extract data from {} and organize it", "text"),
    ("Analyze {} and create a summary document", "text"),
    ("Translate {} and explain key terms", "text"),
    ("Analyze {} and provide recommendations", "text"),
    ("Extract insights from {} and summarize", "text"),
    ("Process {} and create a summary", "text"),
    
    # Complex multi-step tasks (3-4 steps) - 20 variants
    ("Generate a presentation about {} and create visualizations", "topic"),
    ("Summarize {} and create a document with key points", "text"),
    ("Analyze {} and generate a report with recommendations", "text"),
    ("Extract data from {} and create a summary document", "text"),
    ("Translate {} to French, summarize it, and create a report", "text"),
    ("Research {} and create a comprehensive document", "topic"),
    ("Analyze {} and generate a detailed report with conclusions", "text"),
    ("Summarize {}, extract key points, and create a presentation", "text"),
    ("Translate {} to Spanish, analyze it, and summarize", "text"),
    ("Extract information from {}, organize it, and create a report", "text"),
    ("Analyze {}, identify patterns, and generate recommendations", "text"),
    ("Research {}, summarize findings, and create documentation", "topic"),
    ("Process {}, extract insights, and generate a summary", "text"),
    ("Translate {}, analyze content, and create a report", "text"),
    ("Summarize {}, extract data, and organize into a document", "text"),
    ("Analyze {}, generate insights, and create visualizations", "text"),
    ("Research {}, compile findings, and create a comprehensive report", "topic"),
    ("Extract data from {}, analyze patterns, and summarize", "text"),
    ("Translate {}, summarize, and create documentation", "text"),
    ("Process {}, analyze structure, and generate a detailed report", "text"),
    
    # Research and documentation tasks - 15 variants
    ("Research {} and create a comprehensive document", "topic"),
    ("Create a detailed report about {}", "topic"),
    ("Generate a research paper outline about {}", "topic"),
    ("Write a technical document about {}", "topic"),
    ("Create documentation for {}", "topic"),
    ("Develop a research proposal on {}", "topic"),
    ("Create a literature review about {}", "topic"),
    ("Generate a research methodology for {}", "topic"),
    ("Write a research summary on {}", "topic"),
    ("Create a research brief about {}", "topic"),
    ("Develop technical specifications for {}", "topic"),
    ("Create user documentation for {}", "topic"),
    ("Generate API documentation for {}", "topic"),
    ("Write a research abstract about {}", "topic"),
    ("Create a research framework for {}", "topic"),
    
    # Content creation tasks - 20 variants
    ("Create a marketing copy about {}", "topic"),
    ("Write a product description for {}", "topic"),
    ("Generate social media content about {}", "topic"),
    ("Create email content about {}", "topic"),
    ("Write a press release about {}", "topic"),
    ("Create a sales pitch for {}", "topic"),
    ("Generate ad copy about {}", "topic"),
    ("Write a product review for {}", "topic"),
    ("Create a landing page copy for {}", "topic"),
    ("Generate a tagline for {}", "topic"),
    ("Write a headline about {}", "topic"),
    ("Create a call-to-action for {}", "topic"),
    ("Generate a slogan for {}", "topic"),
    ("Write a testimonial about {}", "topic"),
    ("Create a FAQ section for {}", "topic"),
    ("Generate a product comparison for {}", "topic"),
    ("Write a how-to guide for {}", "topic"),
    ("Create a troubleshooting guide for {}", "topic"),
    ("Generate a feature list for {}", "topic"),
    ("Write a benefits description for {}", "topic"),
    
    # Data processing tasks - 15 variants
    ("Process and analyze: {}", "text"),
    ("Extract structured data from: {}", "text"),
    ("Parse and summarize: {}", "text"),
    ("Organize this information: {}", "text"),
    ("Categorize this content: {}", "text"),
    ("Classify this data: {}", "text"),
    ("Structure this information: {}", "text"),
    ("Format this data: {}", "text"),
    ("Clean and organize: {}", "text"),
    ("Extract and format: {}", "text"),
    ("Process and categorize: {}", "text"),
    ("Parse and structure: {}", "text"),
    ("Organize and summarize: {}", "text"),
    ("Extract and organize: {}", "text"),
    ("Process and format: {}", "text"),
    
    # Educational tasks - 15 variants
    ("Create a lesson plan about {}", "topic"),
    ("Explain {} in simple terms", "text"),
    ("Create study notes about {}", "topic"),
    ("Generate quiz questions about {}", "topic"),
    ("Create a study guide for {}", "topic"),
    ("Explain {} step by step", "text"),
    ("Create flashcards about {}", "topic"),
    ("Generate practice problems for {}", "topic"),
    ("Write a tutorial on {}", "topic"),
    ("Create a learning module about {}", "topic"),
    ("Explain {} to a beginner", "text"),
    ("Create an educational video script about {}", "topic"),
    ("Generate learning objectives for {}", "topic"),
    ("Create a curriculum for {}", "topic"),
    ("Write a textbook chapter about {}", "topic"),
    
    # Creative writing tasks - 15 variants
    ("Write a story about {}", "topic"),
    ("Create a poem about {}", "topic"),
    ("Generate a script about {}", "topic"),
    ("Write a dialogue about {}", "topic"),
    ("Create a narrative about {}", "topic"),
    ("Write a character description for {}", "topic"),
    ("Generate a plot outline about {}", "topic"),
    ("Create a scene about {}", "topic"),
    ("Write a monologue about {}", "topic"),
    ("Generate creative content about {}", "topic"),
    ("Create a short story about {}", "topic"),
    ("Write a screenplay about {}", "topic"),
    ("Generate dialogue for {}", "topic"),
    ("Create a storyboard for {}", "topic"),
    ("Write a creative piece about {}", "topic"),
    
    # Business and professional tasks - 15 variants
    ("Create a business plan for {}", "topic"),
    ("Write a proposal for {}", "topic"),
    ("Generate a strategy document about {}", "topic"),
    ("Create a project plan for {}", "topic"),
    ("Write a meeting agenda about {}", "topic"),
    ("Generate a SWOT analysis for {}", "topic"),
    ("Create a budget proposal for {}", "topic"),
    ("Write a performance review for {}", "topic"),
    ("Generate a market analysis for {}", "topic"),
    ("Create a business case for {}", "topic"),
    ("Write a strategic plan for {}", "topic"),
    ("Generate a feasibility study for {}", "topic"),
    ("Create a risk assessment for {}", "topic"),
    ("Write a compliance report for {}", "topic"),
    ("Generate a stakeholder analysis for {}", "topic"),
    
    # Technical and coding tasks - 5 variants (reduced to reach 200)
    ("Generate code documentation for {}", "topic"),
    ("Create a technical specification for {}", "topic"),
    ("Write a code review for {}", "text"),
    ("Generate test cases for {}", "topic"),
    ("Create a system design for {}", "topic"),
]

# Expanded sample data for diversity
SAMPLE_TEXTS = [
    # AI/ML topics
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Natural language processing enables computers to understand and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns.",
    "Reinforcement learning trains agents to make decisions through trial and error.",
    
    # Science topics
    "Quantum computing uses quantum mechanical phenomena to perform computations that would be impossible for classical computers.",
    "Climate change refers to long-term shifts in global temperatures and weather patterns.",
    "The theory of evolution explains how species change over time through natural selection.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    
    # Technology topics
    "Cloud computing allows users to access computing resources over the internet without managing physical infrastructure.",
    "Blockchain is a distributed ledger technology that maintains a continuously growing list of records.",
    "Cybersecurity involves protecting computer systems and networks from digital attacks.",
    "The Internet of Things connects everyday devices to the internet for data collection and automation.",
    
    # Business topics
    "Digital transformation involves using technology to fundamentally change how businesses operate.",
    "Agile methodology emphasizes iterative development and collaboration in software projects.",
    "Customer relationship management helps businesses manage interactions with current and potential customers.",
    
    # General knowledge
    "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th centuries.",
    "Democracy is a system of government where power is held by the people through voting.",
    "Globalization refers to the increasing interconnectedness of economies and cultures worldwide.",
]

SAMPLE_TOPICS = [
    # AI/ML
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "robotics",
    "data science",
    "neural networks",
    "reinforcement learning",
    "transfer learning",
    
    # Technology
    "cloud computing",
    "blockchain",
    "cybersecurity",
    "internet of things",
    "quantum computing",
    "edge computing",
    "5G networks",
    
    # Business
    "digital transformation",
    "agile methodology",
    "project management",
    "customer experience",
    "supply chain management",
    "business intelligence",
    
    # Science
    "climate change",
    "renewable energy",
    "space exploration",
    "genetics",
    "neuroscience",
    
    # General
    "education",
    "healthcare",
    "sustainability",
    "innovation",
    "leadership",
]

def collect_data(num_executions=50, batch_size=100, resume=False):
    """
    Collect training data by running diverse tasks.
    
    Args:
        num_executions: Number of executions to run
        batch_size: Number of executions per batch (for progress tracking)
        resume: If True, skip already collected logs
    """
    print("=" * 70)
    print("ORCHESTAI ENHANCED DATA COLLECTION - PHASE 1 STRENGTHENING")
    print("=" * 70)
    
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
    print(f"\n📊 Collection Plan:")
    print(f"   Target: {num_executions} executions")
    print(f"   Batch size: {batch_size}")
    print(f"   Estimated cost: ${num_executions * 0.15:.2f} - ${num_executions * 0.50:.2f}")
    print(f"\n🚀 Running {num_executions} executions to collect training data...")
    print("-" * 70)
    
    successful = 0
    failed = 0
    total_cost = 0.0
    start_time = time.time()
    
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
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (num_executions - (i + 1)) / rate if rate > 0 else 0
                print(f"[{i+1}/{num_executions}] {status} Success: {successful}, Failed: {failed}, Cost: ${total_cost:.4f}, ETA: {remaining/60:.1f}m")
            
            # Small delay to avoid rate limits
            time.sleep(0.3)
            
        except Exception as e:
            failed += 1
            print(f"Error on execution {i+1}: {e}")
    
    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("DATA COLLECTION COMPLETE")
    print("=" * 70)
    print(f"Total executions: {num_executions}")
    print(f"Successful: {successful} ({successful/num_executions*100:.1f}%)")
    print(f"Failed: {failed} ({failed/num_executions*100:.1f}%)")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Average cost per execution: ${total_cost/num_executions:.4f}")
    print(f"\n📊 Logs saved to: execution_logs/")
    print(f"\n📝 Next steps:")
    print(f"   1. Prepare training data:")
    print(f"      python scripts/prepare_training_data.py --log-file execution_logs/executions_*.jsonl --output training_data.json")
    print(f"   2. Check total examples:")
    print(f"      wc -l execution_logs/*.jsonl")
    print(f"   3. Train Phase 1 with more data:")
    print(f"      python scripts/train_phase1.py --config config.yaml --data training_data.json --epochs 50")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Collect training data automatically for Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 100 examples (minimum for improvement)
  python scripts/collect_training_data.py --num-executions 100
  
  # Collect 500 examples (strong Phase 1)
  python scripts/collect_training_data.py --num-executions 500
  
  # Collect 1000 examples (very strong Phase 1)
  python scripts/collect_training_data.py --num-executions 1000
  
Cost estimates:
  - 100 executions: ~$15-50
  - 500 executions: ~$75-250
  - 1000 executions: ~$150-500
        """
    )
    parser.add_argument("--num-executions", type=int, default=100, 
                       help="Number of executions to run (default: 100, recommended: 500-1000)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Batch size for progress tracking (default: 100)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing logs (skip already collected)")
    args = parser.parse_args()
    
    collect_data(args.num_executions, args.batch_size, args.resume)

