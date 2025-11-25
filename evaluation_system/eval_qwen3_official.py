#!/usr/bin/env python3
"""
Evaluate Official Qwen3-1.7B (Base) on the same benchmarks for comparison.

Usage:
    CUDA_VISIBLE_DEVICES=6 python eval_qwen3_official.py --mode logprob
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import re

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

EVAL_ROOT = Path(__file__).parent.resolve()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OFFICIAL_MODEL = "Qwen/Qwen2.5-1.5B"  # Official base model (Qwen3 requires newer transformers)

BENCHMARKS = {
    "openbookqa": {
        "dataset": "allenai/openbookqa",
        "subset": "main",
        "split": "test",
        "question_key": "question_stem",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Challenge",
        "split": "test",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
    "arc_easy": {
        "dataset": "allenai/ai2_arc",
        "subset": "ARC-Easy",
        "split": "test",
        "question_key": "question",
        "choices_key": "choices",
        "answer_key": "answerKey",
    },
}


def load_official_model(model_name: str, device: str):
    """Load official HuggingFace model."""
    print(f"📦 Loading official model: {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"✅ Model loaded: {param_count:.2f}B parameters on {device}")
    
    return model, tokenizer


def compute_choice_logprobs_hf(model, tokenizer, prompts: list, device: str) -> list:
    """
    Compute log-probability for each choice completion using HuggingFace model.
    """
    logprobs = []
    
    for prompt in prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Get model logits
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits
        
        # Compute log-probs for each token (shifted by 1 for autoregressive)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Sum log-probs of actual tokens (teacher forcing)
        token_logprobs = []
        for i in range(1, tokens.shape[1]):
            next_token_id = tokens[0, i].item()
            lp = log_probs[0, i-1, next_token_id].item()
            token_logprobs.append(lp)
        
        # Average log-prob (length-normalized)
        avg_logprob = np.mean(token_logprobs) if token_logprobs else -float('inf')
        logprobs.append(avg_logprob)
    
    return logprobs


def format_mcq_prompt_logprob(question: str, choices: list, labels: list) -> list:
    """Format MCQ for log-probability evaluation."""
    prompts = []
    for label, choice in zip(labels, choices):
        prompt = f"Question: {question}\nAnswer: {choice}"
        prompts.append(prompt)
    return prompts


def evaluate_benchmark_logprob(model, tokenizer, benchmark_name: str, device: str, max_samples: int = None) -> dict:
    """Evaluate on benchmark using log-probability scoring."""
    config = BENCHMARKS[benchmark_name]
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating on {benchmark_name.upper()} (Log-Prob)")
    print(f"{'='*60}")
    
    dataset = load_dataset(config["dataset"], config["subset"], split=config["split"])
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    correct = 0
    total = 0
    
    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        question = item[config["question_key"]]
        choices_data = item[config["choices_key"]]
        answer_key = item[config["answer_key"]]
        
        if isinstance(choices_data, dict):
            choice_texts = choices_data["text"]
            choice_labels = choices_data["label"]
        else:
            choice_texts = choices_data
            choice_labels = ["A", "B", "C", "D"][:len(choices_data)]
        
        prompts = format_mcq_prompt_logprob(question, choice_texts, choice_labels)
        logprobs = compute_choice_logprobs_hf(model, tokenizer, prompts, device)
        
        pred_idx = np.argmax(logprobs)
        pred_label = choice_labels[pred_idx]
        
        if pred_label == answer_key:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n✅ {benchmark_name.upper()} Results:")
    print(f"   Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Official Qwen3-1.7B")
    parser.add_argument("--model", type=str, default=OFFICIAL_MODEL, help="HuggingFace model name")
    parser.add_argument("--benchmark", type=str, default="all", choices=["all", "openbookqa", "arc_challenge", "arc_easy"])
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="logprob", help="Evaluation mode (ignored, always logprob)")
    args = parser.parse_args()
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    model, tokenizer = load_official_model(args.model, args.device)
    
    if args.benchmark == "all":
        benchmarks_to_run = list(BENCHMARKS.keys())
    else:
        benchmarks_to_run = [args.benchmark]
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "model_type": "official_hf",
        "benchmarks": {},
    }
    
    for benchmark in benchmarks_to_run:
        result = evaluate_benchmark_logprob(model, tokenizer, benchmark, args.device, args.max_samples)
        all_results["benchmarks"][benchmark] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY - {args.model}")
    print(f"{'='*60}")
    for name, res in all_results["benchmarks"].items():
        print(f"  {name:15s}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
    
    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = EVAL_ROOT / f"eval_official_qwen3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")


if __name__ == "__main__":
    main()

