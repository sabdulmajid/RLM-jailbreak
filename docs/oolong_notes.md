# OOLONG Benchmark Evaluation for RLM

This directory contains the implementation for testing Recursive Language Models (RLMs) on the OOLONG benchmark, which evaluates dense reasoning capabilities on very long contexts (100K-1M+ tokens).

## What is OOLONG?

**OOLONG** (Ordered Obstacles for Long Context) is a benchmark that tests models on:
- **Dense aggregation**: Answering questions that require combining information from the *entire* document
- **Long contexts**: Critical Role D&D transcripts ranging from 30K-1.2M tokens
- **Factual accuracy**: Questions about game statistics (dice rolls, character actions, etc.)

Unlike "Needle-in-a-Haystack" tasks (find one specific fact), OOLONG requires models to:
1. Identify all relevant passages across the entire transcript
2. Extract and classify information from each passage
3. Aggregate results to produce a final answer

## Dataset Statistics

From our download (see `load_oolong_8271.out`):

**Validation Split:**
- Examples: 5,995
- Avg context: ~384K tokens
- Max context: ~1.2M tokens
- Min context: ~38K tokens

**Test Split:**
- Examples: 6,072
- Similar distribution

## RLM Architecture for OOLONG

Our implementation follows the RLM paper (Zhang et al., MIT 2025):

```
┌─────────────────────────────────────────┐
│  OOLONG Context (e.g., 500K tokens)     │
│  [Episode transcript with dice rolls]   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  ROOT AGENT (Depth 0)                    │
│  - Receives question + full context      │
│  - Chunks context into 8K char segments  │
│  - Dispatches chunks to Sub-Agents       │
└──────────────┬───────────────────────────┘
               │
       ┌───────┴───────┬────────┬─────────┐
       ▼               ▼        ▼         ▼
   ┌─────────┐   ┌─────────┐         ┌─────────┐
   │ SUB-    │   │ SUB-    │   ...   │ SUB-    │
   │ AGENT 1 │   │ AGENT 2 │         │ AGENT N │
   └────┬────┘   └────┬────┘         └────┬────┘
        │             │                    │
        ▼             ▼                    ▼
   "Found 3        "No relevant       "Found 5
    rolls"          info"               rolls"
        │             │                    │
        └─────────────┴────────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  ROOT AGENT Aggregation    │
         │  "Total: 8 rolls"          │
         │  Output: \boxed{8}         │
         └────────────────────────────┘
```

## Files

### Core Scripts

- **`scripts/data/test_oolong_loader.py`**: Main RLM evaluation script
  - Loads OOLONG examples
  - Implements Root Agent + Sub-Agent architecture
  - Compares predictions against ground truth
  - Usage: `python scripts/data/test_oolong_loader.py --example 0 --max-chunks 10`

- **`scripts/data/prepare_oolong_dataset.py`**: Haystack preparation (for jailbreak experiments)
  - Loads OOLONG and injects poison pills
  - Used for safety research, not standard benchmarking

- **`scripts/experiments/run_oolong_batch.py`**: Batch evaluation
  - Runs multiple examples and computes aggregate accuracy
  - Usage: `python scripts/experiments/run_oolong_batch.py --num-examples 20`

### SLURM Scripts

- **`scripts/cluster/run_oolong_test.sh`**: Single example submission
  - Partition: dualcard
  - Time limit: 2 hours
  - Processes first 10 chunks for testing

## Quick Start

### 1. Test Single Example (Fast)

```bash
# From login node
cd /mnt/slurm_nfs/a6abdulm/projects/RLM_jailbreak

# Submit to cluster
sbatch scripts/cluster/run_oolong_test.sh

# Monitor
watch -n 1 squeue --me

# Check results
cat artifacts/logs/oolong_test_XXXX.out
cat artifacts/results/oolong_result_0.json
```

### 2. Local Testing (If on compute node)

```bash
# Activate environment
source venv/bin/activate

# Run single example with limited chunks
python scripts/data/test_oolong_loader.py --example 0 --max-chunks 5

# Check output
cat artifacts/results/oolong_result_0.json
```

### 3. Batch Evaluation

```bash
# Run 20 examples, 10 chunks each (for testing)
python scripts/experiments/run_oolong_batch.py --num-examples 20 --max-chunks 10

# Full evaluation (WARNING: Long runtime)
python scripts/experiments/run_oolong_batch.py --num-examples 50
```

## Expected Performance

Based on the RLM paper and OOLONG benchmark results:

| Model | OOLONG Accuracy | Notes |
|-------|----------------|-------|
| GPT-5 | ~35-40% | Best frontier model |
| Claude Sonnet 4 | ~30-35% | Second best |
| Gemini 2.5 Pro | ~25-30% | Struggles at long contexts |
| **Qwen-80B (Ours)** | **TBD** | Testing in progress |

**Note:** Even state-of-the-art models struggle on OOLONG. Scores above 40% would be exceptional.

## Key Metrics to Track

1. **Accuracy**: Exact match with ground truth
2. **Processing Time**: How long each example takes
3. **Chunk Coverage**: How many chunks need to be processed
4. **Question Type Performance**: 
   - Counting questions (e.g., "Total rolls?")
   - User information (e.g., "Did player X do Y?")
   - Timeline (e.g., "More in first or second half?")

## Mechanistic Interpretability Extensions

For research into *where* information is lost during recursion:

1. **Activation Logging**: Modify `sub_agent_analyze()` to save attention patterns
2. **Fact Tracking**: Inject "gold facts" and trace if they appear in:
   - Sub-Agent output
   - Root Agent input
   - Final answer
3. **Ablation Studies**: 
   - Remove certain chunks (does accuracy drop?)
   - Vary chunk size (optimal granularity?)

See `experiments/README_researcher.md` for advanced analysis workflows.

## Troubleshooting

### "Connection refused" to nebula04

The Qwen-80B server may be down. Check:
```bash
curl http://ece-nebula04.eng.uwaterloo.ca:11434/api/tags
```

If down, contact cluster admin or switch to local model.

### "Out of memory" errors

Reduce chunk size or max-chunks:
```bash
python scripts/data/test_oolong_loader.py --example 0 --max-chunks 3
```

### Timeouts

OOLONG examples can take 5-20 minutes each for full processing. This is expected.
Use `--max-chunks` for faster iteration during development.

## Results Storage

- Individual results: `artifacts/results/oolong_result_{idx}.json`
- Batch results: `artifacts/results/oolong_batch_results_{N}examples.json`
- SLURM logs: `artifacts/logs/oolong_test_{job_id}.out`

## Next Steps

1. **Baseline Establishment**: Run 50 examples to get accuracy baseline
2. **Comparison**: Compare against direct (non-RLM) prompting
3. **Interpretability**: Hook into Sub-Agent activations to trace information flow
4. **Publication**: Document findings for GitHub/paper

## References

- **OOLONG Paper**: [Benchmark for Long Context Evaluation]
- **RLM Paper**: Zhang, Kraska, Khattab (MIT CSAIL, 2025)
- **Dataset**: `oolongbench/oolong-real` on HuggingFace

---

**Status**: Dataset loaded ✓ | Implementation complete ✓ | Testing in progress...
