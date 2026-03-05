# prompt-reasoning-at-scale

> A unified empirical evaluation of prompt-based reasoning strategies in Large Language Models across model scales.

---

## Overview

Recent advances in prompting have introduced a variety of strategies — such as Chain-of-Thought (CoT) and Self-Consistency — that claim to improve reasoning in Large Language Models. However, these techniques are typically evaluated under inconsistent conditions: different models, different datasets, and different experimental setups, making direct comparison unreliable.

This project performs a **controlled, unified evaluation** of three prompting strategies across two model sizes and two reasoning benchmarks, to answer a core question:

> *Does the advantage of CoT and Self-Consistency over Standard Prompting depend on model scale — and if so, to what degree?*

This work is associated with a term paper submitted for the **Machine Learning for Natural Language Understanding** module.

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| **Zero-shot Standard** | Direct question with no reasoning instructions |
| **Zero-shot Chain-of-Thought (CoT)** | Appends *"Let's think step by step."* to elicit reasoning |
| **Self-Consistency CoT** | Samples CoT reasoning 5 times, selects answer by majority vote |

---

## Models

Both models are from the same family to ensure a controlled scale comparison:

| Model | HuggingFace ID | Size |
|---|---|---|
| Flan-T5-Large | `google/flan-t5-large` | 780M parameters |
| Flan-T5-XL | `google/flan-t5-xl` | 3B parameters |

---

## Datasets

| Dataset | Task Type | Samples Used |
|---|---|---|
| **GSM8K** | Mathematical reasoning | 100 |
| **CommonsenseQA** | Commonsense reasoning | 100 |

Both datasets are sourced from HuggingFace's `datasets` library.

---

## Experimental Design

This study evaluates a **3 strategies × 2 models × 2 datasets = 12 experimental runs**, with 100 samples per run (~1,200 total inferences including Self-Consistency samples).

**Controlled Variables (held constant across all runs):**
- Same 100 questions per dataset (fixed random seed for sampling)
- `temperature=0` for Standard and CoT (deterministic)
- `temperature=0.7` for Self-Consistency sampling only
- `max_new_tokens=512`
- Same answer extraction logic per dataset

**Hypothesis:** CoT and Self-Consistency will yield greater accuracy gains over Standard Prompting in Flan-T5-XL than in Flan-T5-Large, consistent with the emergent abilities hypothesis (Wei et al., 2022).

---

## Metrics

- **Accuracy (%)** — primary metric, percentage of correctly answered questions
- **Consistency Rate (%)** — for Self-Consistency only; percentage of questions where all 5 samples agreed on the same answer

---

## Repository Structure

```
prompt-reasoning-at-scale/
│
├── README.md
├── requirements.txt
│
├── data/
|   ├── sample_data.py               # Script to sample data from GSM8K and CommmonsenseQA
│   ├── gsm8k.json                   # 100 sampled GSM8K questions
│   └── commonsenseqa.json           # 100 sampled CommonsenseQA questions
│
├── prompts/
│   └── templates.py            # Prompt templates for all 3 strategies
│
├── experiments/
│   ├── run_standard.py         # Zero-shot Standard experiments
│   ├── run_cot.py              # Zero-shot CoT experiments
│   └── run_self_consistency.py # Self-Consistency CoT experiments
│
├── results/
│   ├── gsm8k_results.csv       # Raw outputs and scores — GSM8K
│   └── csqa_results.csv        # Raw outputs and scores — CommonsenseQA
│
└── analysis/
    └── evaluate.py             # Accuracy calculation, tables, and plots
```

---

## Results

*Results will be updated upon completion of experiments.*

| Strategy | Model | GSM8K Accuracy | CommonsenseQA Accuracy |
|---|---|---|---|
| Standard | Flan-T5-Large | — | — |
| CoT | Flan-T5-Large | — | — |
| Self-Consistency | Flan-T5-Large | — | — |
| Standard | Flan-T5-XL | — | — |
| CoT | Flan-T5-XL | — | — |
| Self-Consistency | Flan-T5-XL | — | — |

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/mohammedOwaiskh/prompt-reasoning-at-scale.git
cd prompt-reasoning-at-scale
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run experiments
```bash
# Run all strategies on GSM8K with Flan-T5-Large
python experiments/run_standard.py --model flan-t5-large --dataset gsm8k
python experiments/run_cot.py --model flan-t5-large --dataset gsm8k
python experiments/run_self_consistency.py --model flan-t5-large --dataset gsm8k
```

### 4. Evaluate results
```bash
python analysis/evaluate.py
```

---

## Requirements

```
transformers
datasets
torch
accelerate
pandas
numpy
scikit-learn
```

---

## References

- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
- Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
- Wei, J. et al. (2022). *Emergent Abilities of Large Language Models.* TMLR 2022.
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K).* arXiv.
- Talmor, A. et al. (2019). *CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge.* NAACL 2019.
- Chung, H. et al. (2022). *Scaling Instruction-Finetuned Language Models (Flan-T5).* arXiv.

---

## Authors

- Mohammed Owais Khan
- Zaina Naaz Mohd Kalim Ansari

*Term paper submitted for the Machine Learning for Natural Language Understanding module.*

---

## License

MIT License — free to use, adapt, and build upon with attribution.
