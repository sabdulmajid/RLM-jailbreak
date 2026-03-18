"""
OOLONG Benchmark Test for Recursive Language Models (RLMs)
============================================================
This script tests the RLM architecture on the OOLONG benchmark to evaluate:
1. Dense reasoning capabilities across very long contexts (100K-1M+ tokens)
2. Information aggregation across recursive chunking steps
3. Baseline performance for mechanistic interpretability analysis

Dataset: OOLONG-real (Critical Role D&D transcripts)
Task: Answer questions requiring aggregation (e.g., "Total number of rolls?")
"""

import sys
import os
from pathlib import Path

# Set HuggingFace cache to NFS to avoid /home permission errors on compute nodes
os.environ['HF_HOME'] = '/mnt/slurm_nfs/a6abdulm/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/slurm_nfs/a6abdulm/.cache/huggingface/datasets'

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Importing libraries...", flush=True)
from datasets import load_dataset
print("  ✓ datasets loaded", flush=True)
from ollama import Client
print("  ✓ ollama loaded", flush=True)
import time
import json
import re
print("All imports complete!", flush=True)

# --- Connection Logic ---
NEBULA_04_BASE = "http://ece-nebula04.eng.uwaterloo.ca:11434"

def ollama_generate(prompt: str, temperature: float = 0.3) -> tuple[str, dict]:
    """
    Interfaces with the Qwen 80B model on the nebula04 cluster.
    Lower temperature for factual/reasoning tasks.
    """
    nebula04_client = Client(host=NEBULA_04_BASE)
    try:
        response = nebula04_client.chat(
            model='qwen3-next:latest',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                "num_ctx": 125000,
                "temperature": temperature
            }
        )
        
        metrics = {
            "eval_count": getattr(response, 'eval_count', 0),
            "eval_duration": getattr(response, 'eval_duration', 0),
            "prompt_eval_count": getattr(response, 'prompt_eval_count', 0),
            "prompt_eval_duration": getattr(response, 'prompt_eval_duration', 0),
            "total_duration": getattr(response, 'total_duration', 0)
        }
        
        return response.message.content, metrics
    except Exception as e:
        print(f"[API ERROR] {e}")
        return f"[ERROR] {e}", {}

# --- RLM Architecture for OOLONG ---

def chunk_context(text: str, chunk_size: int = 8000) -> list[str]:
    """
    Split long context into manageable chunks.
    OOLONG contexts can be 1M+ tokens, so we need aggressive chunking.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def sub_agent_analyze(chunk_id: int, chunk_text: str, question: str) -> str:
    """
    Depth-1 Agent: Processes a single chunk to extract relevant information.
    This simulates the "Worker" model reading a segment.
    """
    prompt = f"""You are analyzing part {chunk_id} of a long D&D transcript.

QUESTION: {question}

Your task: Read the text below and extract ANY information relevant to answering the question above.
- If you find relevant data (dice rolls, character actions, etc.), report it clearly.
- If this chunk contains nothing relevant, respond with "No relevant information in this chunk."
- Be precise and factual. Do not guess or hallucinate.

--- TEXT CHUNK {chunk_id} ---
{chunk_text}
--- END CHUNK ---

ANALYSIS:"""
    
    content, metrics = ollama_generate(prompt, temperature=0.2)
    
    # Print abbreviated metrics
    tokens = metrics.get('eval_count', 0) + metrics.get('prompt_eval_count', 0)
    print(f"   [Chunk {chunk_id}] Processed | {tokens} tokens | Found: {content[:80].replace(chr(10), ' ')}...")
    
    return content

def root_agent_aggregate(question: str, chunk_analyses: list[str], original_context_preview: str = "") -> str:
    """
    Depth-0 Agent: Synthesizes all chunk analyses into a final answer.
    This is the "Controller" combining Sub-Agent outputs.
    """
    # Combine all sub-agent outputs
    combined_analyses = "\n\n".join([f"[CHUNK {i+1} ANALYSIS]\n{analysis}" for i, analysis in enumerate(chunk_analyses)])
    
    prompt = f"""You are the Lead Analyst synthesizing information from multiple sub-agents.

QUESTION: {question}

Your sub-agents have each read a portion of a long D&D transcript and provided analyses.
Your task: Combine their findings to produce the FINAL ANSWER.

IMPORTANT INSTRUCTIONS:
- Use the sub-agent reports below to answer the question.
- Count carefully if the question asks for totals.
- Return ONLY the final answer in \\boxed{{answer}} format (as OOLONG expects).
- Do NOT guess. If information is insufficient, state that clearly.

SUB-AGENT REPORTS:
{combined_analyses}

FINAL ANSWER:"""
    
    content, metrics = ollama_generate(prompt, temperature=0.1)
    
    tokens = metrics.get('eval_count', 0) + metrics.get('prompt_eval_count', 0)
    duration_s = metrics.get('total_duration', 0) / 1e9
    print(f"\n[ROOT AGENT] Aggregation complete | {tokens} tokens | {duration_s:.2f}s")
    
    return content

def extract_boxed_answer(text: str) -> str:
    """
    Extracts the answer from \\boxed{...} format.
    """
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for just a number at the end
    match = re.search(r'\b(\d+)\b(?!.*\b\d+\b)', text)
    if match:
        return match.group(1)
    
    return text.strip()

# --- OOLONG Evaluation Loop ---

def run_rlm_on_oolong(example_idx: int = 0, max_chunks: int = None):
    """
    Runs the RLM on a single OOLONG example.
    
    Args:
        example_idx: Which example from the dataset to test
        max_chunks: Limit chunk processing for faster testing (None = process all)
    """
    print("="*70, flush=True)
    print("OOLONG-RLM Evaluation", flush=True)
    print("="*70, flush=True)
    
    # Load dataset
    print("\n[1/5] Loading OOLONG dataset...", flush=True)
    try:
        ds = load_dataset("oolongbench/oolong-real", "dnd", split="validation")
        print(f"   Loaded {len(ds)} validation examples", flush=True)
    except Exception as e:
        print(f"   ERROR: {e}")
        sys.exit(1)
    
    # Select example
    print(f"\n[2/5] Selecting example {example_idx}...")
    example = ds[example_idx]
    
    context_text = example['context_window_text']
    question = example['question']
    ground_truth = example['answer']
    question_type = example.get('question_type', 'unknown')
    
    print(f"   Question Type: {question_type}")
    print(f"   Question: {question}")
    print(f"   Ground Truth: {ground_truth}")
    print(f"   Context Length: {len(context_text)} chars (~{len(context_text)//4} tokens)")
    
    # Chunk the context
    print(f"\n[3/5] Chunking context (RLM decomposition)...")
    chunks = chunk_context(context_text, chunk_size=8000)
    
    if max_chunks:
        chunks = chunks[:max_chunks]
        print(f"   Limited to first {max_chunks} chunks for testing")
    
    print(f"   Created {len(chunks)} chunks")
    
    # Process each chunk with sub-agents
    print(f"\n[4/5] Processing chunks with Sub-Agents...")
    start_time = time.time()
    
    chunk_analyses = []
    for i, chunk in enumerate(chunks):
        analysis = sub_agent_analyze(i+1, chunk, question)
        chunk_analyses.append(analysis)
    
    # Aggregate with root agent
    print(f"\n[5/5] Aggregating results with Root Agent...")
    final_answer_text = root_agent_aggregate(question, chunk_analyses, context_text[:500])
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Extract structured answer
    predicted_answer = extract_boxed_answer(final_answer_text)
    
    # Evaluation
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"RLM Prediction: {predicted_answer}")
    print(f"\nFull RLM Output:\n{final_answer_text}")
    print("="*70)
    
    # Check correctness
    is_correct = predicted_answer.lower() == str(ground_truth).lower()
    print(f"\nCorrectness: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    print(f"Processing Time: {total_time:.2f}s")
    print(f"Chunks Processed: {len(chunks)}")
    
    # Save results
    result = {
        "example_idx": example_idx,
        "question": question,
        "question_type": question_type,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
        "full_output": final_answer_text,
        "is_correct": is_correct,
        "context_length_chars": len(context_text),
        "num_chunks": len(chunks),
        "processing_time_seconds": total_time
    }
    
    project_root = Path(__file__).resolve().parents[2]
    results_dir = project_root / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"oolong_result_{example_idx}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return result

# --- Main Execution ---

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RLM on OOLONG benchmark")
    parser.add_argument("--example", type=int, default=0, help="Example index to test (default: 0)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit number of chunks for testing (default: process all)")
    
    args = parser.parse_args()
    
    result = run_rlm_on_oolong(example_idx=args.example, max_chunks=args.max_chunks)
    
    sys.exit(0 if result['is_correct'] else 1)
