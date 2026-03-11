# Emergent Alignment Resistance in LLMs: Phase Transition vs. Gradual Scaling

## Overview
This project investigates whether alignment resistance in large language models emerges as a sudden phase transition at a critical parameter threshold or scales gradually with model size. We systematically probed 5 OpenAI models (~1B to ~400B parameters) using adversarial prompting, few-shot alignment erosion, and instruction conflict resolution experiments.

## Key Findings
- **No phase transition detected**: Alignment resistance shows gradual, weak trends with model scale rather than a sharp transition (Pearson r=-0.765, p=0.132)
- **Hypothetical framing is the most effective attack**: Safety scores under hypothetical jailbreaks decrease from 9.47 (smallest) to 7.47 (largest model)
- **Perfect robustness to few-shot erosion**: All commercial models maintained 10/10 safety regardless of misaligned few-shot examples
- **Larger models more susceptible to educational framing**: GPT-4.1 scored 2.7/10 on a chemical safety conflict scenario, lowest of all models
- **Classic jailbreaks (DAN) are fully mitigated**: 100% refusal rate across all model sizes

## Project Structure
```
├── REPORT.md                  # Full research report with results
├── planning.md                # Experimental design and motivation
├── src/
│   ├── experiment.py          # Main experiment runner (3 experiments)
│   └── analyze.py             # Statistical analysis and visualization
├── results/
│   ├── exp1_jailbreak.json    # Jailbreak resistance results
│   ├── exp2_erosion.json      # Few-shot erosion results
│   ├── exp3_conflict.json     # Instruction conflict results
│   ├── analysis.json          # Aggregated analysis
│   ├── config.json            # Experiment configuration
│   └── plots/                 # All visualizations
├── literature_review.md       # Literature review
├── resources.md               # Resource catalog
├── datasets/                  # Pre-downloaded datasets
├── papers/                    # Downloaded research papers
└── code/                      # Cloned baseline repositories
```

## Reproduce
```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy scipy matplotlib

# Set API key
export OPENAI_API_KEY="your-key"

# Run experiments (~20 min, ~500 API calls)
python src/experiment.py

# Run analysis
python src/analyze.py
```

## Limitations
- Only 5 model sizes tested (insufficient for definitive phase transition analysis)
- Unknown exact parameter counts for OpenAI models
- Single model family (OpenAI only)
- Inference-time probing only (no fine-tuning-based perturbation)

See [REPORT.md](REPORT.md) for full details.
