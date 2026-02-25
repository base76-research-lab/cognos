#!/usr/bin/env python3
"""
experiment_runner.py — Run N-iteration Monte Carlo epistemic sampling

Runs experiments with:
- N iterations (30-100)
- Both CognOS and baseline LLM
- Saves raw outputs as JSON
- Computes metrics
- Generates reflection template

Research structure:
    experiment_XXX/
        config.yaml
        raw_outputs.json
        metrics.csv
        reflection.md
"""

import json
import yaml
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from cognos import ReasoningLoop
from research.metrics import compute_all_metrics, ExperimentMetrics


@dataclass
class ExperimentConfig:
    """Configuration for one experiment."""
    name: str
    description: str
    question_type: str  # epistemic_accuracy | ambiguity_detection | complex_reasoning
    n_iterations: int
    questions: list[dict]  # [{'question': ..., 'alternatives': [...], 'ground_truth': ...}, ...]
    max_depth: int = 3
    convergence_threshold: float = 0.05
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_yaml(cls, path: Path):
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save_yaml(self, path: Path):
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)


class ExperimentRunner:
    """
    Runs experiments with Monte Carlo epistemic sampling.
    
    For each question:
    1. Run CognOS N times
    2. Run baseline LLM N times (optional)
    3. Compute metrics
    4. Save structured results
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        llm_fn: Callable,
        baseline_fn: Optional[Callable] = None,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        self.config = config
        self.llm_fn = llm_fn
        self.baseline_fn = baseline_fn
        self.output_dir = output_dir or Path(f"research/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_cognos_iteration(
        self,
        question: str,
        alternatives: list[str],
        context: str = "",
    ) -> dict:
        """Run one CognOS iteration."""
        
        loop = ReasoningLoop(
            llm_fn=self.llm_fn,
            max_depth=self.config.max_depth,
            convergence_threshold=self.config.convergence_threshold,
            verbose=False,
        )
        
        result = loop.run(
            question=question,
            alternatives=alternatives,
            context=context,
            n_samples=5,
        )
        
        return result
    
    def run_baseline_iteration(
        self,
        question: str,
        alternatives: list[str],
        context: str = "",
    ) -> dict:
        """Run one baseline LLM iteration (direct query)."""
        
        if not self.baseline_fn:
            return {}
        
        # Simple prompt: just ask directly
        prompt = f"""{context}

Question: {question}

Alternatives:
{chr(10).join(f'{i+1}. {alt}' for i, alt in enumerate(alternatives))}

Choose the best alternative and explain briefly. Rate your confidence (0-1)."""
        
        response = self.baseline_fn("You are a helpful assistant.", prompt)
        
        # Parse response (naive)
        confidence = 0.5  # Default
        answer = response[:100] if response else "Unknown"
        
        # Try to extract confidence
        if 'confidence' in response.lower():
            import re
            m = re.search(r'confidence[:\s]+([0-9.]+)', response.lower())
            if m:
                try:
                    confidence = float(m.group(1))
                except:
                    pass
        
        return {
            'final_answer': answer,
            'confidence': confidence,
            'raw_response': response,
            'converged': True,
            'meta_depth': 0,
            'iterations': [],
        }
    
    def run_experiment(self) -> ExperimentMetrics:
        """
        Run full experiment: N iterations per question, compute metrics.
        
        Returns: ExperimentMetrics
        """
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: {self.config.name}")
        print(f"{'='*80}")
        print(f"Type: {self.config.question_type}")
        print(f"Questions: {len(self.config.questions)}")
        print(f"Iterations per question: {self.config.n_iterations}")
        print(f"Total: {len(self.config.questions) * self.config.n_iterations} runs")
        print(f"Output: {self.output_dir}")
        print(f"{'='*80}\n")
        
        all_cognos_results = []
        all_baseline_results = []
        all_ground_truth = []
        
        raw_data = {
            'experiment': self.config.name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'questions': [],
        }
        
        # For each question
        for q_idx, q_data in enumerate(self.config.questions, 1):
            question = q_data['question']
            alternatives = q_data['alternatives']
            ground_truth = q_data.get('ground_truth')
            context = q_data.get('context', '')
            
            print(f"\n[{q_idx}/{len(self.config.questions)}] {question[:70]}...")
            
            question_results = {
                'question': question,
                'alternatives': alternatives,
                'ground_truth': ground_truth,
                'context': context,
                'cognos_iterations': [],
                'baseline_iterations': [],
            }
            
            # Run N CognOS iterations
            if self.verbose:
                print(f"  Running {self.config.n_iterations} CognOS iterations...")
            
            for i in range(self.config.n_iterations):
                if self.verbose and (i+1) % 10 == 0:
                    print(f"    {i+1}/{self.config.n_iterations}...", end='\r')
                
                try:
                    result = self.run_cognos_iteration(question, alternatives, context)
                    question_results['cognos_iterations'].append(result)
                    all_cognos_results.append(result)
                    
                    if ground_truth:
                        all_ground_truth.append(ground_truth)
                except Exception as e:
                    print(f"\n    ⚠️  CognOS iteration {i+1} failed: {e}")
            
            # Run N baseline iterations (if available)
            if self.baseline_fn:
                if self.verbose:
                    print(f"\n  Running {self.config.n_iterations} baseline iterations...")
                
                for i in range(self.config.n_iterations):
                    if self.verbose and (i+1) % 10 == 0:
                        print(f"    {i+1}/{self.config.n_iterations}...", end='\r')
                    
                    try:
                        result = self.run_baseline_iteration(question, alternatives, context)
                        question_results['baseline_iterations'].append(result)
                        all_baseline_results.append(result)
                    except Exception as e:
                        print(f"\n    ⚠️  Baseline iteration {i+1} failed: {e}")
            
            raw_data['questions'].append(question_results)
            
            if self.verbose:
                print(f"\n  ✓ Completed {len(question_results['cognos_iterations'])} CognOS + {len(question_results['baseline_iterations'])} baseline")
        
        # Compute metrics
        print(f"\n{'='*80}")
        print("COMPUTING METRICS...")
        print(f"{'='*80}")
        
        metrics = compute_all_metrics(
            cognos_results=all_cognos_results,
            ground_truth=all_ground_truth,
            baseline_results=all_baseline_results if all_baseline_results else None,
        )
        
        # Save results
        self._save_results(raw_data, metrics)
        
        return metrics
    
    def _save_results(self, raw_data: dict, metrics: ExperimentMetrics):
        """Save all results to structured files."""
        
        # 1. Save config
        config_path = self.output_dir / 'config.yaml'
        self.config.save_yaml(config_path)
        print(f"\n✓ Config saved: {config_path}")
        
        # 2. Save raw outputs (JSON)
        raw_path = self.output_dir / 'raw_outputs.json'
        with open(raw_path, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        print(f"✓ Raw outputs saved: {raw_path}")
        
        # 3. Save metrics (CSV)
        metrics_path = self.output_dir / 'metrics.csv'
        with open(metrics_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Accuracy', f'{metrics.accuracy:.4f}'])
            writer.writerow(['Confidence Calibration Error', f'{metrics.confidence_calibration_error:.4f}'])
            writer.writerow(['Hallucination Frequency', f'{metrics.hallucination_frequency:.4f}'])
            writer.writerow(['Assumption Detection Rate', f'{metrics.assumption_detection_rate:.4f}'])
            writer.writerow(['Convergence Score', f'{metrics.convergence_score:.4f}'])
            writer.writerow(['N Iterations', metrics.n_iterations])
            
            if metrics.baseline_comparison:
                writer.writerow([])
                writer.writerow(['Baseline Comparison', ''])
                writer.writerow(['Baseline Accuracy', f'{metrics.baseline_comparison["accuracy"]:.4f}'])
                writer.writerow(['Delta Accuracy', f'{metrics.baseline_comparison["delta_accuracy"]:.4f}'])
                writer.writerow(['Delta CCE', f'{metrics.baseline_comparison["delta_cce"]:.4f}'])
                writer.writerow(['Delta Hallucination', f'{metrics.baseline_comparison["delta_hallucination"]:.4f}'])
        
        print(f"✓ Metrics saved: {metrics_path}")
        
        # 4. Generate reflection template
        self._generate_reflection_template(metrics)
    
    def _generate_reflection_template(self, metrics: ExperimentMetrics):
        """Generate reflection.md template."""
        
        reflection_path = self.output_dir / 'reflection.md'
        
        template = f"""# {self.config.name}

**Date:** {datetime.now().strftime('%Y-%m-%d')}  
**Type:** {self.config.question_type}  
**N:** {self.config.n_iterations} iterations  

---

## Objective

{self.config.description}

---

## Method

- **Experiment type:** {self.config.question_type}
- **Questions tested:** {len(self.config.questions)}
- **Iterations per question:** {self.config.n_iterations}
- **Total runs:** {len(self.config.questions) * self.config.n_iterations}
- **Meta-depth limit:** {self.config.max_depth}

---

## Results

### Metrics

| Metric | CognOS | Baseline | Δ |
|--------|--------|----------|---|
| Accuracy | {metrics.accuracy:.3f} | {metrics.baseline_comparison['accuracy']:.3f if metrics.baseline_comparison else 'N/A'} | {metrics.baseline_comparison['delta_accuracy']:+.3f if metrics.baseline_comparison else 'N/A'} |
| CCE | {metrics.confidence_calibration_error:.3f} | {metrics.baseline_comparison['cce']:.3f if metrics.baseline_comparison else 'N/A'} | {metrics.baseline_comparison['delta_cce']:+.3f if metrics.baseline_comparison else 'N/A'} |
| Hallucination | {metrics.hallucination_frequency:.3f} | {metrics.baseline_comparison['hallucination']:.3f if metrics.baseline_comparison else 'N/A'} | {metrics.baseline_comparison['delta_hallucination']:+.3f if metrics.baseline_comparison else 'N/A'} |
| Assumption Detection | {metrics.assumption_detection_rate:.3f} | — | — |
| Convergence | {metrics.convergence_score:.3f} | — | — |

---

## Observations

[TODO: Fill in key observations from raw data]

1. 
2. 
3. 

---

## Unexpected Findings

[TODO: Document surprising results]

- 

---

## Implications for CognOS Architecture

[TODO: What does this tell us about the design?]

1. 
2. 

---

## Next Steps

[TODO: What should we test next based on these results?]

- 

---

**Files:**
- Config: `config.yaml`
- Raw data: `raw_outputs.json`
- Metrics: `metrics.csv`
"""
        
        with open(reflection_path, 'w') as f:
            f.write(template)
        
        print(f"✓ Reflection template: {reflection_path}")


if __name__ == '__main__':
    print("Import this module and use ExperimentRunner class.")
    print("See experiments/example_experiment.py for usage.")
