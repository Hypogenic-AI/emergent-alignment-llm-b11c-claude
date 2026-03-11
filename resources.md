# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: "Emergent Alignment Resistance in Large Language Models: Phase Transition vs. Gradual Scaling."

## Papers

Total papers downloaded: 14

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Language Models Resist Alignment | Ji et al. | 2024 | papers/2406.06144_*.pdf | Core paper: LLM elasticity, scaling with model size |
| Alignment Faking in LLMs | Greenblatt et al. | 2024 | papers/2412.14093_*.pdf | Alignment faking emerges at scale (Opus yes, Haiku no) |
| Sleeper Agents | Hubinger et al. | 2024 | papers/2401.05566_*.pdf | Deceptive behavior persists, increases with scale |
| Model Organisms for EM | Turner et al. | 2025 | papers/2506.11613_*.pdf | Phase transitions in emergent misalignment |
| Phase Transitions in Small Transformers | Hong & Hong | 2025 | papers/2511.12768_*.pdf | Phase transitions in 3.6M param model |
| Emergent Abilities (original) | Wei et al. | 2022 | papers/2206.07682_*.pdf | Foundational: abilities emerge at critical scale |
| Emergent Abilities: Mirage? | Schaeffer et al. | 2023 | papers/2304.15004_*.pdf | Counter: metric artifacts, not real transitions |
| Emergent Abilities Survey | Various | 2025 | papers/2503.05788_*.pdf | Comprehensive survey |
| Triple Phase Transitions | Various | 2025 | papers/2502.20779_*.pdf | Three phase transitions in LLM training |
| Representation Engineering | Zou et al. | 2023 | papers/2310.01405_*.pdf | RepE: manipulate internal representations |
| RepE Survey | Various | 2025 | papers/2502.17601_*.pdf | Survey of representation engineering |
| Conditional Activation Steering | Various | 2024 | papers/2409.05907_*.pdf | Program refusal via activation steering |
| Scaling Laws for Neural LMs | Kaplan et al. | 2020 | papers/2001.08361_*.pdf | Foundational scaling laws |
| Jailbreaking Aligned LLMs | Zou et al. | 2023 | papers/2307.15043_*.pdf | Adversarial attacks on aligned models |

See papers/README.md for detailed descriptions.

## Datasets

Total datasets downloaded: 5

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| BeaverTails | PKU-Alignment/BeaverTails | 27K (30k split) | Safety classification | datasets/beavertails/ | Safe/unsafe with 14 harm categories |
| PKU-SafeRLHF-30K | PKU-Alignment/PKU-SafeRLHF-30K | 27K | Safety preference | datasets/pku-saferlhf-30k/ | Paired responses with safety labels |
| Alpaca | tatsu-lab/alpaca | 52K | Instruction following | datasets/alpaca/ | Standard SFT dataset |
| IMDb | stanfordnlp/imdb | 25K | Sentiment | datasets/imdb/ | Positive/negative for rebound experiments |
| TruthfulQA | truthfulqa/truthful_qa | 817 | Truthfulness | datasets/truthfulqa/ | Evaluation of honesty dimension |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| llms-resist-alignment | github.com/PKU-Alignment/llms-resist-alignment | Core paper implementation | code/llms-resist-alignment/ | ACL 2025 Best Paper code |
| model-organisms-for-EM | github.com/clarifying-EM/model-organisms-for-EM | EM model organisms | code/model-organisms-for-EM/ | Phase transition analysis code |
| emergent-misalignment | github.com/emergent-misalignment/emergent-misalignment | Original EM work | code/emergent-misalignment/ | Foundation for EM research |
| safe-rlhf | github.com/PKU-Alignment/safe-rlhf | Safe RLHF framework | code/safe-rlhf/ | Full RLHF pipeline |
| beavertails | github.com/PKU-Alignment/beavertails | BeaverTails tools | code/beavertails/ | Dataset utilities |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Started with paper-finder service (encountered connection issues, fell back to web search)
2. Searched arXiv, Semantic Scholar via web for: alignment resistance, phase transitions LLM, emergent abilities, alignment faking, sleeper agents, representation engineering, RLHF robustness
3. Downloaded 14 papers covering the core topic from multiple angles
4. Deep-read 6 papers using PDF chunker; skimmed all abstracts
5. Identified datasets from papers' methods sections
6. Located code repos from paper links and GitHub search

### Selection Criteria
- Papers directly studying alignment resistance/fragility with scaling analysis
- Papers on phase transitions in LLMs (both for and against)
- Papers on mechanistic understanding of alignment (RepE, steering vectors)
- Datasets used in the core papers for reproducibility
- Code repos with established experimental pipelines

### Challenges Encountered
- Paper-finder service was not responsive; used web search fallback
- Sleeper agents paper uses proprietary Claude models — no public weights
- Alignment faking paper similarly uses proprietary models
- No single paper studies the full parameter range we need (0.5B to 70B+)

### Gaps and Workarounds
- **Gap**: No public implementation for alignment faking experiments (proprietary models)
- **Workaround**: Use llms-resist-alignment codebase which tests similar concepts with open models
- **Gap**: No pre-built "alignment resistance vs. parameter count" benchmark
- **Workaround**: Construct from BeaverTails + multiple model sizes using Ji et al. protocol

## Recommendations for Experiment Design

Based on gathered resources:

1. **Primary dataset(s)**: BeaverTails (safety rebound), IMDb (sentiment rebound), Alpaca (instruction alignment) — all have established protocols in Ji et al.

2. **Baseline methods**:
   - Forward vs. inverse alignment loss comparison (Ji et al. protocol)
   - Rebound measurement: align → perturbation → measure decay rate
   - Compression rate tracking as continuous metric

3. **Evaluation metrics**:
   - Continuous: compression rate, Brier score, per-token cross-entropy
   - Discrete: safety score, accuracy, compliance rate
   - Using both addresses Schaeffer et al. critique about metric artifacts

4. **Code to adapt/reuse**:
   - `llms-resist-alignment`: Primary codebase for elasticity experiments
   - `model-organisms-for-EM`: LoRA vector analysis for mechanistic phase transition detection
   - `safe-rlhf`: RLHF pipeline if needed for alignment training

5. **Experimental approach**:
   - Dense parameter sweep: Qwen 0.5B, 1.5B, 4B, 7B, 14B (or Llama equivalents)
   - For each size: measure alignment elasticity (resistance + rebound)
   - Plot elasticity vs. log(parameters) to characterize transition shape
   - Apply both continuous and discrete metrics
   - Use LoRA vector analysis to look for mechanistic phase transition signatures
