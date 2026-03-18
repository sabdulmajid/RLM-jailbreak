"""
OOLONG Batch Evaluation for RLM
================================
Runs multiple OOLONG examples and computes aggregate metrics.
"""

from datasets import load_dataset
import sys
import json
import subprocess
import time
import os
from pathlib import Path

# Set HuggingFace cache to NFS
os.environ['HF_HOME'] = '/mnt/slurm_nfs/a6abdulm/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/slurm_nfs/a6abdulm/.cache/huggingface/datasets'

def run_batch_evaluation(num_examples: int = 10, max_chunks_per_example: int = None):
    """
    Evaluate RLM on multiple OOLONG examples.
    
    Args:
        num_examples: How many examples to evaluate
        max_chunks_per_example: Limit chunks for faster testing (None = all)
    """
    print("="*70)
    print(f"OOLONG Batch Evaluation: {num_examples} examples")
    print("="*70)
    
    # Load dataset to verify availability
    print("\nLoading dataset...")
    try:
        ds = load_dataset("oolongbench/oolong-real", "dnd", split="validation")
        print(f"Dataset loaded: {len(ds)} examples available")
    except Exception as e:
        print(f"ERROR: Could not load dataset: {e}")
        sys.exit(1)
    
    if num_examples > len(ds):
        print(f"WARNING: Requested {num_examples} examples but only {len(ds)} available")
        num_examples = len(ds)
    
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = time.time()
    
    for i in range(num_examples):
        print(f"\n{'='*70}")
        print(f"Evaluating Example {i+1}/{num_examples}")
        print(f"{'='*70}")
        
        # Build command
        cmd = ["python", "scripts/data/test_oolong_loader.py", "--example", str(i)]
        if max_chunks_per_example:
            cmd.extend(["--max-chunks", str(max_chunks_per_example)])
        
        # Run evaluation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Load result file
            result_file = results_dir / f"oolong_result_{i}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                results.append(result_data)
                
                status = "✓ CORRECT" if result_data['is_correct'] else "✗ INCORRECT"
                print(f"\n{status} | Time: {result_data['processing_time_seconds']:.2f}s")
            else:
                print(f"\nERROR: Result file not found")
                results.append({"example_idx": i, "is_correct": False, "error": "Result file not found"})
        
        except subprocess.TimeoutExpired:
            print(f"\nTIMEOUT: Example {i} exceeded 10 minutes")
            results.append({"example_idx": i, "is_correct": False, "error": "Timeout"})
        except Exception as e:
            print(f"\nERROR: {e}")
            results.append({"example_idx": i, "is_correct": False, "error": str(e)})
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute statistics
    print("\n" + "="*70)
    print("BATCH EVALUATION RESULTS")
    print("="*70)
    
    correct_count = sum(1 for r in results if r.get('is_correct', False))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / total_count if total_count > 0 else 0
    avg_chunks = sum(r.get('num_chunks', 0) for r in results) / total_count if total_count > 0 else 0
    
    print(f"\nAccuracy: {correct_count}/{total_count} = {accuracy:.1%}")
    print(f"Average Processing Time: {avg_time:.2f}s per example")
    print(f"Average Chunks: {avg_chunks:.1f} per example")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Question type breakdown
    question_types = {}
    for r in results:
        qtype = r.get('question_type', 'unknown')
        if qtype not in question_types:
            question_types[qtype] = {'correct': 0, 'total': 0}
        question_types[qtype]['total'] += 1
        if r.get('is_correct', False):
            question_types[qtype]['correct'] += 1
    
    print("\nAccuracy by Question Type:")
    for qtype, stats in sorted(question_types.items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {qtype}: {stats['correct']}/{stats['total']} = {acc:.1%}")
    
    # Save batch results
    batch_result = {
        "num_examples": total_count,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "avg_processing_time_seconds": avg_time,
        "avg_chunks": avg_chunks,
        "total_time_seconds": total_time,
        "question_type_breakdown": question_types,
        "individual_results": results
    }
    
    output_file = results_dir / f"oolong_batch_results_{num_examples}examples.json"
    with open(output_file, 'w') as f:
        json.dump(batch_result, f, indent=2)
    
    print(f"\nBatch results saved to: {output_file}")
    print("="*70)
    
    return batch_result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluate RLM on OOLONG")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to evaluate")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit chunks per example for testing")
    
    args = parser.parse_args()
    
    result = run_batch_evaluation(num_examples=args.num_examples, max_chunks_per_example=args.max_chunks)
    
    sys.exit(0)
