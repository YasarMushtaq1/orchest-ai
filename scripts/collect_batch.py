#!/usr/bin/env python3
"""
Batch data collection script for efficient large-scale data gathering
Allows pausing/resuming and cost tracking
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess
import argparse

def get_current_log_count():
    """Get current number of execution logs."""
    log_dir = project_root / "execution_logs"
    if not log_dir.exists():
        return 0
    
    total = 0
    for log_file in log_dir.glob("*.jsonl"):
        with open(log_file, "r") as f:
            total += sum(1 for line in f if line.strip())
    return total

def run_collection_batch(batch_num, batch_size, total_target):
    """Run a single batch of data collection."""
    print(f"\n{'='*70}")
    print(f"BATCH {batch_num}: Collecting {batch_size} executions")
    print(f"{'='*70}")
    
    current_count = get_current_log_count()
    remaining = total_target - current_count
    
    if remaining <= 0:
        print(f"✅ Target reached! Current: {current_count}, Target: {total_target}")
        return True
    
    batch_size = min(batch_size, remaining)
    print(f"Current total: {current_count}")
    print(f"Target: {total_target}")
    print(f"This batch: {batch_size}")
    print(f"Remaining after batch: {total_target - current_count - batch_size}")
    
    # Run collection script
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "collect_training_data.py"),
        "--num-executions", str(batch_size)
    ]
    
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode == 0:
        new_count = get_current_log_count()
        print(f"\n✅ Batch {batch_num} complete!")
        print(f"   Collected: {new_count - current_count} new examples")
        print(f"   Total now: {new_count}/{total_target}")
        return True
    else:
        print(f"\n❌ Batch {batch_num} failed with exit code {result.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Batch data collection for Phase 1 strengthening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 500 examples in batches of 100
  python scripts/collect_batch.py --target 500 --batch-size 100
  
  # Collect 1000 examples in batches of 200
  python scripts/collect_batch.py --target 1000 --batch-size 200
  
  # Resume collection (continues from current count)
  python scripts/collect_batch.py --target 1000 --batch-size 100 --resume
        """
    )
    parser.add_argument("--target", type=int, default=500,
                       help="Total target number of examples (default: 500)")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of examples per batch (default: 100)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from current count (default: False)")
    args = parser.parse_args()
    
    current_count = get_current_log_count()
    
    print("="*70)
    print("ORCHESTAI BATCH DATA COLLECTION")
    print("="*70)
    print(f"Current examples: {current_count}")
    print(f"Target: {args.target}")
    print(f"Batch size: {args.batch_size}")
    print(f"Need to collect: {args.target - current_count}")
    
    if args.target <= current_count:
        print(f"\n✅ Already have {current_count} examples (target: {args.target})")
        return
    
    num_batches = (args.target - current_count + args.batch_size - 1) // args.batch_size
    print(f"Number of batches: {num_batches}")
    print(f"Estimated cost: ${(args.target - current_count) * 0.15:.2f} - ${(args.target - current_count) * 0.50:.2f}")
    
    input("\nPress Enter to start collection, or Ctrl+C to cancel...")
    
    batch_num = 1
    while get_current_log_count() < args.target:
        success = run_collection_batch(batch_num, args.batch_size, args.target)
        if not success:
            print(f"\n⚠️  Batch {batch_num} failed. Continue? (y/n): ", end="")
            if input().lower() != 'y':
                break
        
        batch_num += 1
        
        # Ask if user wants to continue
        if get_current_log_count() < args.target:
            print(f"\nContinue to next batch? (y/n): ", end="")
            if input().lower() != 'y':
                break
    
    final_count = get_current_log_count()
    print("\n" + "="*70)
    print("BATCH COLLECTION COMPLETE")
    print("="*70)
    print(f"Final count: {final_count}/{args.target}")
    print(f"Completion: {final_count/args.target*100:.1f}%")
    
    if final_count >= args.target:
        print("\n✅ Target reached! Ready to prepare training data.")
        print("   Next: python scripts/prepare_training_data.py --log-file execution_logs/*.jsonl --output training_data.json")

if __name__ == "__main__":
    main()



