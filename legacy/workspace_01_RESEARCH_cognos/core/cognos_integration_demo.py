#!/usr/bin/env python3
"""
cognos_integration_demo.py — Visar hur de tre lagrena (confidence, divergence, meta) samverkar.

Denna demo visar en faktisk agentisk kedja:
  1. Strukturerad fråga → MC sampling
  2. CognOS confidence-beräkning
  3. Om SYNTHESIZE: divergence_semantics extraherar antaganden
  4. System rekurerar med ny fråga baserad på meta_question
  5. Konvergenskontroll stoppar när C och antaganden stabiliseras
"""

import sys
from pathlib import Path

sys.path.append('/media/bjorn/iic/cognos/core')
sys.path.append('/media/bjorn/iic/Jasper')

from confidence import compute_confidence
from divergence_semantics import synthesize_reason, convergence_check


def demo_scenario():
    """
    Scenario: Hypotes om CognOS.
    
    CognOS bedöms från tre vinklar:
    1. Ursprunglig fråga (falsifierbarhet)
    2. Om divergens ↦ synthesize_reason() extraherar antaganden
    3. Konvergens-check för att veta när vi är klara
    """

    print("=" * 100)
    print("COGNOS INTEGRATED STACK DEMO")
    print("=" * 100)

    # ─────────────────────────────────────────────────────────────────────────────
    # ITERATION 1: Originalfråga
    # ─────────────────────────────────────────────────────────────────────────────

    print("\n" + "🔵 " * 25)
    print("\n⏱️  ITERATION 1 — Originalfråga")
    print("─" * 100)

    q1 = ("Hur starkt falsifierbar är HYPOTHESIS_V02 i sin nuvarande form?")
    alt1 = [
        "A: Svag falsifierbarhet",
        "B: Delvis falsifierbar men kräver striktare mättrösklar",
        "C: Starkt falsifierbar med tydliga kriterier"
    ]
    votes1 = {"B": 3, "C": 2}  # Från vår tidigare körning
    mc_predictions1 = [0.7, 0.8, 0.9, 0.8, 0.7]  # Rapporterad konfidens från 5 samples
    p1 = 3 / 5  # 60% majoritet

    # Layer 1: CognOS confidence
    result1 = compute_confidence(p1, mc_predictions1)
    C1 = result1['confidence']
    decision1 = result1['decision']

    print(f"\n📝 Fråga: {q1}")
    print(f"🗳️  Röstfördelning: {votes1}")
    print(f"📊 p={p1:.2f}, Ue={result1['epistemic_uncertainty']:.4f}, "
          f"Ua={result1['aleatoric_uncertainty']:.3f}")
    print(f"🎯 Resultat: C={C1:.3f}, Beslut={decision1}")

    if decision1 == 'synthesize':
        print(f"\n✨ SYNTHESIZE DETEKTERAD — Analyser divergens...")

        # Layer 2: Divergence Semantics
        divergence1 = synthesize_reason(
            question=q1,
            alternatives=alt1,
            vote_distribution=votes1,
            confidence=C1,
            is_multimodal=result1['is_multimodal']
        )

        print(f"\n🔍 Divergentsanalys:")
        print(f"   Majoritet (B): {divergence1['majority_assumption']}")
        print(f"   Minoritet (C): {divergence1['minority_assumption']}")
        print(f"\n   🌉 Divergenskälla: {divergence1['divergence_source']}")
        print(f"   💡 Integration: {divergence1['integration_strategy']}")
        print(f"\n   ❓ Meta-fråga för nästa iteration:\n      {divergence1['meta_question']}")

        meta_q2 = divergence1['meta_question']
    else:
        print(f"\n✅ Redan konsensus eller tydlig beslut — ingen divergensanalys behövlig.")
        meta_q2 = None

    # ─────────────────────────────────────────────────────────────────────────────
    # ITERATION 2: Follow-up baserat på meta_question
    # ─────────────────────────────────────────────────────────────────────────────

    if meta_q2:
        print("\n\n" + "🔵 " * 25)
        print("\n⏱️  ITERATION 2 — Follow-up via Meta-question")
        print("─" * 100)

        print(f"\n🔄 Ursprunglig fråga genererade Meta-fråga:")
        print(f"   {meta_q2}")

        # Simulerad nya röster baserat på skärpt fråga
        q2 = meta_q2
        alt2 = [
            "A: Klara operationell definitioner krävs innan test",
            "B: Vi kan köra pilot och skärpa iterativt",
            "C: Definitionerna är redan tillräckliga"
        ]
        votes2 = {"B": 4, "A": 1}  # Någon bättre konsensus nu
        mc_predictions2 = [0.75, 0.75, 0.85, 0.80, 0.75]
        p2 = 4 / 5  # 80% majoritet

        result2 = compute_confidence(p2, mc_predictions2)
        C2 = result2['confidence']
        decision2 = result2['decision']

        print(f"\n📊 Nya resultat: C={C2:.3f}, Beslut={decision2}")
        print(f"   Konsensus förbättrad: p {p1:.2f} → {p2:.2f}")
        print(f"   Konfidens förbättrad: C {C1:.3f} → {C2:.3f}")

        # Layer 3: Convergence check
        confidence_history = [C1, C2]
        assumption_history = [
            divergence1.get('divergence_source', 'Initial'),
            "Iterativ skärpning av definitioner"
        ]

        convergence = convergence_check(
            iteration=2,
            confidence_history=confidence_history,
            assumption_history=assumption_history,
            threshold=0.05
        )

        print(f"\n🔄 Konvergens-check:")
        print(f"   Stabilitet: {convergence['stability_score']:.1%}")
        print(f"   {convergence['reason']}")
        print(f"   Fortsätt? {convergence['should_continue']}")

    # ─────────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────────

    print("\n\n" + "=" * 100)
    print("📋 SAMMANFATTNING DES ARKITEKTUREN")
    print("=" * 100)

    summary = """
    Layer 0 (Structured Input):
      - Struktuerad fråga (VAL/KONFIDENS) tvingar modell att vara diskret
      
    Layer 1 (Epistemisk Integritet):
      - compute_confidence() kombinerar p + Ue + Ua → C
      - Fyra beslut: auto | synthesize | explore | escalate
      
    Layer 2 (Divergence Semantics):
      - Vid SYNTHESIZE: synthesize_reason() extraherar UNDERLIGGANDE ANTAGANDEN
      - Inte "vi är oeniga" utan "vi antar olika saker om X"
      - Genererar meta_question för nästa iteration
      
    Layer 3 (Convergence Control):
      - convergence_check() detekterar när C och antaganden stabiliseras
      - Stoppar rekursion automatiskt
      - Möjliggör multi-turn agentiska kedjor

    🎯 RESULTAT:
      - Systemet vet när det vet (HIGH C: AUTO)
      - Systemet vet när det divergerar (SYNTHESIZE: extrahera divergensen)
      - Systemet vet när det bör stoppa (CONVERGENCE)
      
    🧠 FUNKTIONELLT EKVIVALENT MED:
      - ACC (Anterior Cingulate Cortex) — conflict monitoring
      - Prefrontal lager — meta-reasoning
      - Integration lager — semantisk sammanfattning
      
    📦 EXTERRALISERAD METAKOGNITION för AI
    """

    print(summary)

    print("\n" + "=" * 100)


if __name__ == '__main__':
    demo_scenario()
