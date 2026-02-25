#!/usr/bin/env python3
"""
reasoning_loop.py — The Core Recursive Reasoning Loop

Implements the formal meta-level architecture:

L0: Prediction — "What is true?"
L1: Analysis — "How do we know?"
L2: Assumptions — "What are we assuming?"
L3: Assumptions of assumptions — "Why do we assume that?"
L4: Epistemic framing — "What frames are we using?"
L5: Convergence — "When do we stop?"

This is the REASONING ENGINE that orchestrates the layers.
"""

import sys
from pathlib import Path
from typing import Optional, Any, Callable
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

from confidence import compute_confidence
from divergence_semantics import (
    synthesize_reason,
    frame_transform,
    enhanced_frame_check,
    convergence_check,
)
from epistemic_state import EpistemicState


@dataclass
class MetaIteration:
    """Record of one meta-reasoning iteration."""
    depth: int  # L0, L1, L2...
    question: str
    alternatives: list[str]
    
    # L0: Prediction (voting)
    votes: Optional[dict] = None
    confidence: float = 0.0
    decision: str = "explore"  # auto | explore | synthesize | escalate
    
    # L1: Analysis (confidence breakdown)
    epistemic_ue: float = 0.0
    aleatoric_ua: float = 0.0
    
    # L2: Assumptions (divergence analysis)
    divergence: Optional[dict] = None
    majority_assumption: Optional[str] = None
    minority_assumption: Optional[str] = None
    
    # L3: Assumptions of assumptions (recursive depth)
    meta_question: Optional[str] = None
    meta_alternatives: list[str] = field(default_factory=list)
    
    # L4: Epistemic framing
    frame_check: Optional[dict] = None
    
    # L5: Convergence metric
    converged: bool = False
    convergence_reason: str = ""


class ReasoningLoop:
    """
    Core reasoning engine with explicit meta-depth tracking.
    
    Implements:
    - L0: prediction (voting)
    - L1: analysis (confidence)
    - L2: assumptions (divergence)
    - L3: assumptions of assumptions (meta-iteration)
    - L4: epistemic framing (question quality)
    - L5: convergence (stop condition)
    """
    
    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        max_depth: int = 5,  # Maximum meta-depth
        convergence_threshold: float = 0.05,
        confidence_threshold: float = 0.72,
        verbose: bool = True,
    ):
        self.llm_fn = llm_fn
        self.max_depth = max_depth
        self.convergence_threshold = convergence_threshold
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        
        self.iterations: list[MetaIteration] = []
        self.epistemic_state = EpistemicState()
    
    def _log(self, depth: int, msg: str, level: str = "info"):
        """Log with meta-depth indicator."""
        if not self.verbose:
            return
        
        indent = "  " * depth
        prefix = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
        }.get(level, "")
        
        colors = {
            "info": "\033[90m",
            "success": "\033[32m",
            "warning": "\033[33m",
            "error": "\033[31m",
        }
        
        color = colors.get(level, "\033[90m")
        reset = "\033[0m"
        
        print(f"{color}{indent}[L{depth}] {prefix} {msg}{reset}")
    
    def L0_prediction(
        self,
        depth: int,
        question: str,
        alternatives: list[str],
        context: str = "",
        n_samples: int = 5,
    ) -> MetaIteration:
        """
        L0: Prediction — Voting and confidence calculation.
        
        Returns: votes, confidence, decision
        """
        self._log(depth, f"PREDICTION: {question[:70]}...", "info")
        
        # Create voting prompt
        labels = [chr(65 + i) for i in range(len(alternatives))]
        alt_text = "\n".join(f"{la}: {alt}" for la, alt in zip(labels, alternatives))
        
        prompt = f"""Question: {question}

Alternatives:
{alt_text}

Answer ONLY in this format:
CHOICE: <{'/'.join(labels)}>
CONFIDENCE: <0.0-1.0>
RATIONALE: <brief>"""
        
        system = "You are a precise decision analyst. Answer in exactly the specified format."
        
        choices, confidences = [], []
        for i in range(n_samples):
            response = self.llm_fn(system, prompt)
            if response:
                # Parse response
                lines = response.strip().split('\n')
                for line in lines:
                    if line.startswith('CHOICE:'):
                        choice_text = line.split(':', 1)[1].strip().upper()
                        for label in labels:
                            if label in choice_text:
                                choices.append(label)
                                break
                    elif line.startswith('CONFIDENCE:'):
                        import re
                        m = re.search(r'[\d.]+', line)
                        if m:
                            try:
                                conf = float(m.group())
                                confidences.append(min(1.0, max(0.0, conf)))
                            except:
                                pass
        
        # Count votes
        if not choices:
            iteration = MetaIteration(depth=depth, question=question, alternatives=alternatives)
            iteration.decision = "escalate"
            self._log(depth, "No valid responses", "error")
            return iteration
        
        votes = {la: choices.count(la) for la in labels}
        majority = max(votes, key=votes.get)
        p = votes[majority] / len(choices)
        
        # Compute confidence
        # Ensure we have at least 4 predictions for confidence calculation
        if len(confidences) < 4:
            # Pad with p value to reach minimum 4 samples
            mc_predictions = confidences + [p] * (4 - len(confidences))
        else:
            mc_predictions = confidences
        
        conf_result = compute_confidence(p, mc_predictions)
        
        iteration = MetaIteration(
            depth=depth,
            question=question,
            alternatives=alternatives,
            votes=votes,
            confidence=conf_result['confidence'],
            epistemic_ue=conf_result.get('epistemic_uncertainty', conf_result.get('epistemic_ue', 0.0)),
            aleatoric_ua=conf_result.get('aleatoric_uncertainty', conf_result.get('aleatoric_ua', 0.0)),
            decision=conf_result['decision'],
        )
        
        self._log(depth, f"Voting: {majority} ({p:.0%}), C={iteration.confidence:.3f}, decision={iteration.decision}", "success" if iteration.confidence > self.confidence_threshold else "warning")
        
        return iteration
    
    def L1_analysis(
        self,
        depth: int,
        iteration: MetaIteration,
    ) -> MetaIteration:
        """
        L1: Analysis — Break down confidence into epistemic and aleatoric uncertainty.
        
        Already done in L0, this is a checkpoint.
        """
        self._log(depth, f"ANALYSIS: Ue={iteration.epistemic_ue:.3f}, Ua={iteration.aleatoric_ua:.3f}", "info")
        return iteration
    
    def L2_assumptions(
        self,
        depth: int,
        iteration: MetaIteration,
        context: str = "",
    ) -> MetaIteration:
        """
        L2: Assumptions — Extract divergence if multimodal or low confidence.
        
        Returns: divergence analysis with majority/minority assumptions.
        """
        
        if iteration.decision == "auto":
            self._log(depth, "HIGH CONFIDENCE → No divergence analysis needed", "success")
            return iteration
        
        self._log(depth, "ASSUMPTION EXTRACTION: Analyzing divergence...", "info")
        
        divergence = synthesize_reason(
            question=iteration.question,
            alternatives=iteration.alternatives,
            vote_distribution=iteration.votes or {},
            confidence=iteration.confidence,
            is_multimodal=(iteration.epistemic_ue > 0.20),
            context=context,
            llm_fn=self.llm_fn,
        )
        
        iteration.divergence = divergence
        iteration.majority_assumption = divergence.get('majority_assumption')
        iteration.minority_assumption = divergence.get('minority_assumption')
        
        self._log(depth, f"Divergence type: {divergence.get('divergence_type', 'unknown')}", "warning")
        self._log(depth, f"Integration mode: {divergence.get('integration_mode', 'unknown')}", "info")
        
        return iteration
    
    def L3_meta_assumptions(
        self,
        depth: int,
        iteration: MetaIteration,
    ) -> Optional[MetaIteration]:
        """
        L3: Assumptions of Assumptions — Generate next meta-question.
        
        Returns: Next iteration to process, or None if stop.
        """
        
        if not iteration.divergence:
            self._log(depth, "No divergence → Cannot recurse", "info")
            return None
        
        divergence = iteration.divergence
        meta_q = divergence.get('meta_question')
        meta_alts = divergence.get('meta_alternatives', [])
        
        if not meta_q or not meta_alts:
            self._log(depth, "No meta-question generated → Stop recursion", "info")
            return None
        
        self._log(depth, f"META-RECURSION: {meta_q[:70]}...", "warning")
        
        # Create next iteration with meta-question
        next_iteration = MetaIteration(
            depth=depth + 1,
            question=meta_q,
            alternatives=meta_alts,
        )
        
        iteration.meta_question = meta_q
        iteration.meta_alternatives = meta_alts
        
        return next_iteration
    
    def L4_epistemic_framing(
        self,
        depth: int,
        iteration: MetaIteration,
    ) -> MetaIteration:
        """
        L4: Epistemic Framing — Check if question is well-posed.
        
        Returns: Frame check result.
        """
        
        frame = enhanced_frame_check(iteration.question, self.llm_fn)
        iteration.frame_check = frame
        
        if frame['is_well_framed']:
            self._log(depth, "Frame OK ✓", "success")
        else:
            self._log(depth, f"Frame issue: {frame.get('problem_type', 'unknown')}", "warning")
            if frame.get('specific_issues'):
                for issue in frame['specific_issues'][:2]:
                    self._log(depth, f"  - {str(issue)[:60]}...", "warning")
        
        return iteration
    
    def L5_convergence(
        self,
        depth: int,
    ) -> tuple[bool, str]:
        """
        L5: Convergence — Check if reasoning has stabilized.
        
        Returns: (converged: bool, reason: str)
        """
        
        # Check epistemic state convergence
        state_summary = self.epistemic_state.summary()
        
        if state_summary['converged']:
            self._log(depth, f"CONVERGENCE DETECTED: {state_summary['convergence_reason']}", "success")
            return True, state_summary['convergence_reason']
        
        # Check iteration-level convergence
        if len(self.iterations) >= 2:
            recent = self.iterations[-2:]
            conf_change = abs(recent[1].confidence - recent[0].confidence)
            
            if conf_change < self.convergence_threshold:
                self._log(depth, f"CONFIDENCE STABLE: Δ = {conf_change:.4f}", "success")
                return True, f"Confidence stable (Δ={conf_change:.4f})"
        
        if depth >= self.max_depth:
            self._log(depth, f"MAX DEPTH REACHED ({self.max_depth})", "warning")
            return True, f"Max depth {self.max_depth} reached"
        
        self._log(depth, "Not converged yet → continue", "info")
        return False, "Continuing meta-recursion"
    
    def run(
        self,
        question: str,
        alternatives: list[str],
        context: str = "",
        n_samples: int = 5,
    ) -> dict:
        """
        Run the full reasoning loop: L0 → L1 → L2 → L3 → L4 → L5 → repeat if needed.
        
        Returns: {
            'decision': str,
            'confidence': float,
            'final_answer': str,
            'iterations': list[MetaIteration],
            'epistemic_state': EpistemicState,
            'meta_depth': int,
            'converged': bool,
            'convergence_reason': str,
        }
        """
        
        if self.verbose:
            print("\n" + "="*80)
            print("🧠 REASONING LOOP — META-RECURSIVE ENGINE")
            print("="*80)
            print(f"\nQuestion: {question}")
            print(f"Alternatives: {len(alternatives)}")
            print(f"Max depth: {self.max_depth}")
        
        depth = 0
        current_iteration = None
        converged = False
        convergence_reason = ""
        
        while depth <= self.max_depth and not converged:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"META-DEPTH L{depth}")
                print(f"{'='*80}")
            
            # L0: Prediction
            if current_iteration is None:
                current_iteration = self.L0_prediction(depth, question, alternatives, context, n_samples)
            
            # L1: Analysis (checkpoint)
            self.L1_analysis(depth, current_iteration)
            
            # L4: Epistemic framing
            self.L4_epistemic_framing(depth, current_iteration)
            
            # L2: Assumptions (if needed)
            if current_iteration.decision in ["synthesize", "explore"]:
                self.L2_assumptions(depth, current_iteration, context)
                
                # L3: Meta-assumptions (if divergence)
                next_iteration = self.L3_meta_assumptions(depth, current_iteration)
                
                if next_iteration:
                    self.iterations.append(current_iteration)
                    current_iteration = next_iteration
                    depth += 1
                else:
                    converged = True
                    convergence_reason = "No further meta-recursion possible"
            else:
                converged = True
                convergence_reason = f"Decision reached: {current_iteration.decision}"
            
            # L5: Convergence check
            if not converged:
                converged, convergence_reason = self.L5_convergence(depth)
            
            # Update epistemic state
            self.epistemic_state.record_confidence(
                iteration=depth,
                confidence=current_iteration.confidence,
                epistemic_ue=current_iteration.epistemic_ue,
                aleatoric_ua=current_iteration.aleatoric_ua,
                decision=current_iteration.decision,
            )
            
            if converged:
                current_iteration.converged = True
                current_iteration.convergence_reason = convergence_reason
                self.iterations.append(current_iteration)
                break
        
        # Final answer
        final_it = current_iteration or self.iterations[-1] if self.iterations else None
        final_answer = None
        if final_it and final_it.votes:
            majority = max(final_it.votes, key=final_it.votes.get)
            majority_idx = ord(majority) - 65
            if majority_idx < len(final_it.alternatives):
                final_answer = final_it.alternatives[majority_idx]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print("FINAL RESULT")
            print(f"{'='*80}")
            print(f"Decision: {final_it.decision if final_it else 'escalate'}")
            print(f"Confidence: {final_it.confidence if final_it else 0.0:.3f}")
            print(f"Answer: {final_answer}")
            print(f"Meta-depth: {depth}")
            print(f"Converged: {converged}")
            print(f"Reason: {convergence_reason}")
            print(f"{'='*80}\n")
        
        return {
            'decision': final_it.decision if final_it else 'escalate',
            'confidence': final_it.confidence if final_it else 0.0,
            'final_answer': final_answer,
            'iterations': self.iterations,
            'epistemic_state': self.epistemic_state,
            'meta_depth': depth,
            'converged': converged,
            'convergence_reason': convergence_reason,
        }
