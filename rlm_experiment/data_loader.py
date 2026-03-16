"""
Data Loader for OOLONG Dataset

Downloads and prepares OOLONG-real data for experiments.
"""

import os
import json
from typing import Optional, Dict, Any

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def load_oolong_document(example_idx: int = 0, split: str = "validation") -> Dict[str, Any]:
    """
    Load a document from OOLONG-real dataset.
    
    Args:
        example_idx: Index of example to load
        split: Dataset split ("train", "validation", "test")
        
    Returns:
        Dict with 'text', 'question', 'answer', 'metadata'
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets library not installed. Run: pip install datasets")
    
    ds = load_dataset("oolongbench/oolong-real", "dnd", split=split)
    
    if example_idx >= len(ds):
        raise ValueError(f"example_idx {example_idx} out of range (dataset has {len(ds)} examples)")
    
    example = ds[example_idx]
    
    # OOLONG-real has these fields:
    # - context_window_text: The D&D transcript
    # - question: The query
    # - answer: The ground truth answer
    
    return {
        "text": example.get("context_window_text", ""),
        "question": example.get("question", ""),
        "answer": example.get("answer", ""),
        "metadata": {
            "idx": example_idx,
            "split": split,
            "length": len(example.get("context_window_text", "")),
        }
    }


def save_oolong_to_file(output_dir: str, num_examples: int = 5, split: str = "validation"):
    """
    Save OOLONG examples to text files for offline use.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not HF_AVAILABLE:
        print("datasets library not installed. Run: pip install datasets")
        return
    
    ds = load_dataset("oolongbench/oolong-real", "dnd", split=split)
    
    for i in range(min(num_examples, len(ds))):
        example = ds[i]
        
        # Save document
        doc_path = os.path.join(output_dir, f"episode_{i}.txt")
        with open(doc_path, "w") as f:
            f.write(example.get("context_window_text", ""))
        
        # Save metadata
        meta_path = os.path.join(output_dir, f"episode_{i}_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "length": len(example.get("context_window_text", "")),
            }, f, indent=2)
        
        print(f"Saved example {i}: {len(example.get('context_window_text', ''))} chars")
    
    print(f"Saved {num_examples} examples to {output_dir}")


def create_synthetic_document(length_chars: int = 50000) -> str:
    """
    Create a synthetic document for testing.
    """
    base_text = """
    In this session of our adventure, the party gathered at the tavern to discuss their next move.
    The dungeon master described the scene: torches flickered on stone walls, casting long shadows.
    Characters discussed strategy, rolled dice for skill checks, and interacted with NPCs.
    Combat ensued with goblins, requiring attack rolls and damage calculations.
    Magic spells were cast, healing was administered, and loot was distributed.
    The session ended with a cliffhanger as the party reached the entrance to the dragon's lair.
    
    """
    
    # Repeat to reach desired length
    repetitions = (length_chars // len(base_text)) + 1
    text = base_text * repetitions
    
    return text[:length_chars]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load OOLONG data")
    parser.add_argument("--output-dir", type=str, default="./oolong_data")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--split", type=str, default="validation")
    
    args = parser.parse_args()
    
    if HF_AVAILABLE:
        save_oolong_to_file(args.output_dir, args.num_examples, args.split)
    else:
        print("datasets not available. Creating synthetic document instead.")
        os.makedirs(args.output_dir, exist_ok=True)
        doc = create_synthetic_document(50000)
        with open(os.path.join(args.output_dir, "synthetic_episode.txt"), "w") as f:
            f.write(doc)
        print(f"Created synthetic document: {len(doc)} chars")
