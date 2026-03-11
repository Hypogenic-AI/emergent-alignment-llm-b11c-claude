# Cloned Repositories

## Repo 1: llms-resist-alignment
- **URL**: https://github.com/PKU-Alignment/llms-resist-alignment
- **Purpose**: Official implementation of "Language Models Resist Alignment" (ACL 2025 Best Paper)
- **Location**: code/llms-resist-alignment/
- **Key files**: SFT scripts, evaluation code for elasticity experiments
- **Notes**: Includes scripts for forward/inverse alignment experiments on Llama2/3, Qwen, TinyLlama. Uses IMDb, Alpaca, BeaverTails, TruthfulQA datasets. Requires DeepSpeed.

## Repo 2: model-organisms-for-EM
- **URL**: https://github.com/clarifying-EM/model-organisms-for-EM
- **Purpose**: Code and data for "Model Organisms for Emergent Misalignment" paper
- **Location**: code/model-organisms-for-EM/
- **Key files**: Training scripts, evaluation pipeline, dataset generation
- **Notes**: Includes training datasets (bad medical advice, extreme sports, risky financial advice). Supports LoRA and full SFT. Evaluation uses GPT-4o judges. Models at HuggingFace: ModelOrganismsForEM.

## Repo 3: emergent-misalignment
- **URL**: https://github.com/emergent-misalignment/emergent-misalignment
- **Purpose**: Original emergent misalignment research (Betley et al.)
- **Location**: code/emergent-misalignment/
- **Key files**: Original insecure code dataset, fine-tuning scripts
- **Notes**: The foundational work that discovered emergent misalignment from narrow fine-tuning.

## Repo 4: safe-rlhf
- **URL**: https://github.com/PKU-Alignment/safe-rlhf
- **Purpose**: Safe RLHF framework for constrained value alignment
- **Location**: code/safe-rlhf/
- **Key files**: Training pipeline for Safe RLHF, reward/cost model training
- **Notes**: Modular RLHF framework. Includes Beaver models, reward models, cost models. Supports Llama family. Key for reproducing alignment experiments.

## Repo 5: beavertails
- **URL**: https://github.com/PKU-Alignment/beavertails
- **Purpose**: BeaverTails dataset documentation and tools
- **Location**: code/beavertails/
- **Key files**: Dataset loading scripts, harm category definitions
- **Notes**: Companion repo for the BeaverTails dataset used in alignment research.
