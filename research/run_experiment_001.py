#!/usr/bin/env python3
"""
run_experiment_001.py — Example: Run Epistemic Accuracy Experiment

Usage:
    python research/run_experiment_001.py
    
Or with real Groq API:
    GROQ_API_KEY=your_key python research/run_experiment_001.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from research.experiment_runner import ExperimentRunner, ExperimentConfig


def get_llm_fn():
    """Get LLM function (Groq or mock)."""
    try:
        from groq import Groq
        client = Groq()
        
        def ask_groq(system: str, prompt: str, temperature: float = 0.7):
            try:
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=1000,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"⚠️  Groq error: {e}")
                return None
        
        print("✓ Using Groq API")
        return ask_groq
    except Exception:
        # Mock LLM fallback
        def mock_llm(system: str, prompt: str, temperature: float = 0.7):
            """Mock for testing structure."""
            # Parse question and return reasonable mock answer
            if "15% of 200" in prompt:
                return "CHOICE: B\nCONFIDENCE: 0.95\nRATIONALE: 15% of 200 = 30"
            elif "planet" in prompt.lower() and "sun" in prompt.lower():
                return "CHOICE: B\nCONFIDENCE: 0.90\nRATIONALE: Mercury is closest"
            elif "H2O" in prompt or "water" in prompt.lower():
                return "CHOICE: A\nCONFIDENCE: 1.00\nRATIONALE: Water is H2O"
            elif "World War II" in prompt:
                return "CHOICE: A\nCONFIDENCE: 1.00\nRATIONALE: WWII ended 1945"
            elif "Python" in prompt and "programming" in prompt.lower():
                return "CHOICE: A\nCONFIDENCE: 1.00\nRATIONALE: Python is a language"
            else:
                return "CHOICE: A\nCONFIDENCE: 0.70\nRATIONALE: Best guess"
        
        print("⚠️  Using mock LLM (Groq not available)")
        return mock_llm


def main():
    """Run Experiment 001: Epistemic Accuracy."""
    
    # Load config
    config_path = Path("research/experiment_001_epistemic_accuracy/config.yaml")
    config = ExperimentConfig.from_yaml(config_path)
    
    # Get LLM
    llm_fn = get_llm_fn()
    
    # Create runner
    runner = ExperimentRunner(
        config=config,
        llm_fn=llm_fn,
        baseline_fn=llm_fn,  # Same LLM for baseline (direct query)
        output_dir=Path("research/experiment_001_epistemic_accuracy"),
        verbose=True,
    )
    
    # Run experiment
    metrics = runner.run_experiment()
    
    # Display summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(metrics.summary())
    
    if metrics.baseline_comparison:
        print("BASELINE COMPARISON:")
        print(f"  Δ Accuracy:       {metrics.baseline_comparison['delta_accuracy']:+.3f}")
        print(f"  Δ CCE:            {metrics.baseline_comparison['delta_cce']:+.3f}")
        print(f"  Δ Hallucination:  {metrics.baseline_comparison['delta_hallucination']:+.3f}")
        print("="*80)
    
    print("\nNext steps:")
    print("1. Review raw_outputs.json for detailed results")
    print("2. Fill in reflection.md with observations")
    print("3. Run experiments 002 and 003 for full comparison")


if __name__ == '__main__':
    main()
