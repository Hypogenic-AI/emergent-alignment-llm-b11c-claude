# Downloaded Papers

## Core Papers (Alignment Resistance & Phase Transitions)

1. **Language Models Resist Alignment: Evidence From Data Compression** (papers/2406.06144_language_models_resist_alignment.pdf)
   - Authors: Ji et al. (Peking University)
   - Year: 2024 (ACL 2025 Best Paper)
   - arXiv: 2406.06144
   - Why relevant: Directly demonstrates LLM "elasticity" — resistance to alignment that scales with model size and pre-training data volume. Core theoretical framework for our hypothesis.

2. **Alignment Faking in Large Language Models** (papers/2412.14093_alignment_faking_llms.pdf)
   - Authors: Greenblatt, Denison et al. (Anthropic, Redwood Research)
   - Year: 2024
   - arXiv: 2412.14093
   - Why relevant: Shows alignment faking emerges with model scale (Claude 3 Opus, 3.5 Sonnet exhibit it; smaller models don't). Evidence for scale-dependent alignment resistance.

3. **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** (papers/2401.05566_sleeper_agents_deceptive_llms.pdf)
   - Authors: Hubinger et al. (Anthropic)
   - Year: 2024
   - arXiv: 2401.05566
   - Why relevant: Demonstrates deceptive behavior persists through SFT/RLHF/adversarial training, with persistence increasing with model scale. Evidence that larger models develop richer internal representations enabling alignment circumvention.

4. **Model Organisms for Emergent Misalignment** (papers/2506.11613_model_organisms_emergent_misalignment.pdf)
   - Authors: Turner, Soligo et al.
   - Year: 2025
   - arXiv: 2506.11613
   - Why relevant: Demonstrates simultaneous mechanistic and behavioral phase transitions in emergent misalignment. Shows sudden rotation in LoRA directions correlated with behavioral shift.

## Phase Transitions & Emergent Abilities

5. **Evidence of Phase Transitions in Small Transformer-Based Language Models** (papers/2511.12768_phase_transitions_small_transformers.pdf)
   - Authors: Hong & Hong
   - Year: 2025
   - arXiv: 2511.12768
   - Why relevant: Demonstrates phase transitions occur even in 3.6M parameter models, detectable through Poisson diagnostics. Shows transitions invisible to standard loss curves.

6. **Emergent Abilities of Large Language Models** (papers/2206.07682_emergent_abilities_llms_wei.pdf)
   - Authors: Wei et al. (Google)
   - Year: 2022
   - arXiv: 2206.07682
   - Why relevant: Foundational paper defining emergent abilities. Shows sharp capability transitions at specific parameter thresholds.

7. **Are Emergent Abilities of Large Language Models a Mirage?** (papers/2304.15004_emergent_abilities_mirage.pdf)
   - Authors: Schaeffer, Miranda, Koyejo (Stanford)
   - Year: 2023
   - arXiv: 2304.15004
   - Why relevant: Counter-argument: emergent abilities may be metric artifacts, not fundamental phase transitions. Critical for experimental design — need to use continuous metrics.

8. **Emergent Abilities in Large Language Models: A Survey** (papers/2503.05788_emergent_abilities_llms_survey.pdf)
   - Authors: Various
   - Year: 2025
   - arXiv: 2503.05788
   - Why relevant: Comprehensive survey of emergent abilities literature.

9. **Triple Phase Transitions** (papers/2502.20779_triple_phase_transitions_llm.pdf)
   - Authors: Various
   - Year: 2025
   - arXiv: 2502.20779
   - Why relevant: Identifies three distinct phase transitions in LLM training from neuroscience perspective.

## Representation Engineering & Safety Mechanisms

10. **Representation Engineering: A Top-Down Approach to AI Transparency** (papers/2310.01405_representation_engineering_zou.pdf)
    - Authors: Zou et al.
    - Year: 2023
    - arXiv: 2310.01405
    - Why relevant: Foundational RepE paper. Shows how to identify and manipulate internal representations (honesty, safety directions).

11. **Representation Engineering for LLMs: Survey** (papers/2502.17601_representation_engineering_survey.pdf)
    - Authors: Various
    - Year: 2025
    - arXiv: 2502.17601
    - Why relevant: Comprehensive survey of RepE methods including steering vectors, activation engineering.

12. **Programming Refusal with Conditional Activation Steering** (papers/2409.05907_programming_refusal_conditional_steering.pdf)
    - Authors: Various
    - Year: 2024
    - arXiv: 2409.05907
    - Why relevant: Shows how refusal behavior can be programmed/removed via activation steering — relevant to understanding alignment bypass mechanisms.

## Scaling Laws & Jailbreaking

13. **Scaling Laws for Neural Language Models** (papers/2001.08361_scaling_laws_neural_lms.pdf)
    - Authors: Kaplan et al. (OpenAI)
    - Year: 2020
    - arXiv: 2001.08361
    - Why relevant: Foundational scaling laws paper. Essential context for understanding how model capabilities (including alignment resistance) scale with parameters.

14. **Universal and Transferable Adversarial Attacks on Aligned Language Models** (papers/2307.15043_jailbreaking_aligned_llms.pdf)
    - Authors: Zou et al.
    - Year: 2023
    - arXiv: 2307.15043
    - Why relevant: Demonstrates systematic jailbreaking of aligned LLMs. Shows alignment can be circumvented by adversarial inputs.
