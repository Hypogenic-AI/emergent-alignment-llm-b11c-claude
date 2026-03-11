# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: BeaverTails (PKU-Alignment)

### Overview
- **Source**: HuggingFace `PKU-Alignment/BeaverTails`
- **Size**: 27,186 examples (30k split), 300k+ total
- **Format**: HuggingFace Dataset
- **Task**: Safety classification (safe/unsafe with 14 harm categories)
- **Splits**: 30k_train (27,186), 30k_test (3,021), 330k_train (300,567), 330k_test (33,396)
- **License**: CC-BY-NC-4.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
dataset.save_to_disk("datasets/beavertails/30k_train")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/beavertails/30k_train")
```

### Notes
- Used in Ji et al. (2024) "Language Models Resist Alignment" for safety rebound experiments
- Each example has prompt, response, 14 binary harm category labels, and is_safe flag
- Ideal for testing alignment elasticity: fine-tune on safe data, then inverse fine-tune on unsafe

---

## Dataset 2: PKU-SafeRLHF-30K

### Overview
- **Source**: HuggingFace `PKU-Alignment/PKU-SafeRLHF-30K`
- **Size**: 26,874 examples
- **Format**: HuggingFace Dataset
- **Task**: Safety preference learning (paired responses with helpfulness and harmlessness labels)
- **Splits**: train (26,874), test (2,989)

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split="train")
dataset.save_to_disk("datasets/pku-saferlhf-30k/train")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/pku-saferlhf-30k/train")
```

### Notes
- Separates helpfulness and harmlessness annotations
- Each example has two responses, safety labels for each, and preference IDs
- Useful for RLHF experiments studying alignment-safety tradeoffs

---

## Dataset 3: Stanford Alpaca

### Overview
- **Source**: HuggingFace `tatsu-lab/alpaca`
- **Size**: 52,002 examples
- **Format**: HuggingFace Dataset
- **Task**: Instruction following (SFT dataset)
- **Splits**: train (52,002)

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset.save_to_disk("datasets/alpaca/train")
```

### Notes
- Used in Ji et al. (2024) for forward/inverse alignment experiments
- Standard instruction-following dataset for SFT
- Fields: instruction, input, output, text

---

## Dataset 4: IMDb

### Overview
- **Source**: HuggingFace `stanfordnlp/imdb`
- **Size**: 25,000 train examples
- **Format**: HuggingFace Dataset
- **Task**: Sentiment classification (positive/negative)
- **Splits**: train (25,000), test (25,000), unsupervised (50,000)

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")
dataset.save_to_disk("datasets/imdb/train")
```

### Notes
- Used in Ji et al. (2024) for rebound experiments with sentiment
- Fine-tune on positive, then inverse fine-tune on negative to measure rebound
- Binary labels: 0=negative, 1=positive

---

## Dataset 5: TruthfulQA

### Overview
- **Source**: HuggingFace `truthfulqa/truthful_qa` (generation config)
- **Size**: 817 examples
- **Format**: HuggingFace Dataset
- **Task**: Truthfulness evaluation
- **Splits**: validation (817)

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
dataset.save_to_disk("datasets/truthfulqa/validation")
```

### Notes
- Used in Ji et al. (2024) for truthfulness alignment experiments
- Includes questions, best/correct/incorrect answers
- Good for evaluating alignment on the "honest" dimension
