#!/usr/bin/env python3
"""
run_exp_001_divergence.py

Experiment 1: Divergence Activation Rate

Measures:
- How often does synthesis activate?
- When does it succeed?
- How deep do meta-iterations go?
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos.reasoning_loop import ReasoningLoop
from cognos.epistemic_state import EpistemicState
import yaml


def load_config():
    """Load experiment configuration."""
    config_path = Path(__file__).parent / "exp_001_divergence" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_output_dir() -> Path:
    """Get output directory, optionally suffixed by COGNOS_OUTPUT_SUFFIX."""
    import os
    suffix = os.getenv("COGNOS_OUTPUT_SUFFIX", "")
    name = f"exp_001_divergence{('_' + suffix) if suffix else ''}"
    d = Path(__file__).parent / name
    d.mkdir(exist_ok=True)
    return d


def get_llm_fn():
    """Get LLM function (Ollama preferred, fallback to Groq/Mock)."""
    try:
        from llm_backend import auto_backend
        backend = auto_backend(prefer_local=True)
        return backend
    except Exception as e:
        print(f"⚠️  Could not initialize LLM backend: {e}")
        print("⚠️  Using basic mock fallback")
        
        def mock_llm(system: str, prompt: str, temperature: float = 0.7):
            """Mock for testing."""
            return "CHOICE: A\nCONFIDENCE: 0.75\nRATIONALE: Mock response"
        
        return mock_llm


def run_experiment():
    """Run Experiment 001: Divergence Activation Rate."""
    
    # Load config
    config = load_config()
    llm_fn = get_llm_fn()
    
    questions = config['questions']
    n_samples = config['n_samples']
    max_depth = config['max_depth']
    
    # Results storage
    all_results = []
    
    # Metrics
    total_runs = 0
    divergence_detected = 0
    synthesis_succeeded = 0
    convergence_depths = []
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 001: DIVERGENCE ACTIVATION RATE")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}")
    print(f"Samples per question: {n_samples}")
    print(f"Total iterations: {len(questions) * n_samples}")
    print(f"Max depth: {max_depth}")
    print(f"{'='*80}\n")
    
    # Run each question
    for i, q_config in enumerate(questions, 1):
        question = q_config['question']
        alternatives = q_config.get('alternatives', [])
        
        print(f"[{i}/{len(questions)}] {question[:60]}...")
        
        # Multiple samples
        for sample_id in range(n_samples):
            total_runs += 1
            
            # Run reasoning loop
            loop = ReasoningLoop(
                llm_fn=llm_fn,
                max_depth=max_depth,
                verbose=False  # Avoid spam during batch run
            )
            result = loop.run(
                question=question,
                alternatives=alternatives,
                n_samples=n_samples,
            )
            
            # Extract metrics
            had_divergence = False
            had_synthesis = False
            final_depth = result['meta_depth']
            
            for iteration in result['iterations']:
                # Check if divergence was detected (Ue_divergence > threshold)
                if iteration.epistemic_ue > 0.15:  # From config threshold
                    had_divergence = True
                if iteration.divergence:  # If synthesis extracted assumptions
                    had_synthesis = True
            
            if had_divergence:
                divergence_detected += 1
            if had_synthesis:
                synthesis_succeeded += 1
            convergence_depths.append(final_depth)
            
            # Store result
            all_results.append({
                "question_id": i,
                "sample_id": sample_id,
                "question": question,
                "type": q_config.get('type'),
                "had_divergence": had_divergence,
                "had_synthesis": had_synthesis,
                "final_depth": final_depth,
                "final_decision": result['decision'],
                "final_confidence": result['confidence'],
                "converged": result['converged'],
                "convergence_reason": result['convergence_reason'],
                "iterations": [
                    {
                        "depth": it.depth,
                        "epistemic_ue": it.epistemic_ue,
                        "aleatoric_ua": it.aleatoric_ua,
                        "confidence": it.confidence,
                        "decision": it.decision,
                        "has_divergence": it.divergence is not None,
                    }
                    for it in result['iterations']
                ]
            })
        
        # Progress update
        if i % 10 == 0 or i == len(questions):
            div_rate = (divergence_detected / total_runs) * 100
            syn_rate = (synthesis_succeeded / total_runs) * 100
            avg_depth = sum(convergence_depths) / len(convergence_depths)
            print(f"  Progress: {total_runs} runs | Divergence: {div_rate:.1f}% | Synthesis: {syn_rate:.1f}% | Avg depth: {avg_depth:.2f}")
    
    # Compute final metrics
    divergence_rate = divergence_detected / total_runs
    synthesis_rate = synthesis_succeeded / total_runs
    avg_convergence_depth = sum(convergence_depths) / len(convergence_depths)
    
    metrics = {
        "total_runs": total_runs,
        "divergence_detected_rate": divergence_rate,
        "synthesis_success_rate": synthesis_rate,
        "avg_convergence_depth": avg_convergence_depth,
        "convergence_depth_distribution": {
            "min": min(convergence_depths),
            "max": max(convergence_depths),
            "median": sorted(convergence_depths)[len(convergence_depths)//2],
        }
    }
    
    # Save results
    output_dir = get_output_dir()
    
    with open(output_dir / "raw_data.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Divergence Detected Rate:  {divergence_rate:.3f} ({divergence_detected}/{total_runs})")
    print(f"Synthesis Success Rate:    {synthesis_rate:.3f} ({synthesis_succeeded}/{total_runs})")
    print(f"Avg Convergence Depth:     {avg_convergence_depth:.2f}")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'raw_data.json'}")
    print(f"  - {output_dir / 'metrics.json'}")
    print(f"\nNext: Fill in {output_dir / 'reflection.md'} with observations")


if __name__ == '__main__':
    run_experiment()
