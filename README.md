# prompt-reasoning-at-scale

> A Unified Evaluation of Prompt-based reasoning strategies in Large Language Models

---

## Overview

Recent advances in prompting have introduced a variety of strategies — such as Chain-of-Thought (CoT) and Self-Consistency — that claim to improve reasoning in Large Language Models. However, these techniques are typically evaluated under inconsistent conditions: different models, different datasets, and different experimental setups, making direct comparison unreliable.

This project performs a **controlled, unified evaluation** of four prompting strategies on a single model across two reasoning benchmarks, to answer a core research question:

> *Which prompt-based reasoning strategies are most effective for mathematical reasoning vs. commonsense reasoning, and do the same strategies generalise across both task types?*

This work is associated with a term paper submitted for the **Trends in Machine Learning** module.

---

## Prompting Strategies

| # | Strategy | Description |
|---|---|---|
| 1 | **Zero-shot Standard** | Direct question with no examples and no reasoning instructions — baseline |
| 2 | **Few-shot Standard (3-shot)** | 3 solved examples provided before the question, no reasoning instructions |
| 3 | **Zero-shot CoT** | Appends *"Let's think step by step."* to elicit a reasoning chain |
| 4 | **Self-Consistency CoT** | Samples CoT reasoning 5 times at temperature=0.7, selects answer by majority vote |

---

## Model

| Model | HuggingFace ID | Type | Size |
|---|---|---|---|
| Gemma-2B-IT | `google/gemma-2b-it` | Decoder-only, instruction-tuned | 2B parameters |

> **Note:** Gemma is a gated model. You must accept the license at [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it) and use a HuggingFace access token to download it.

---

## Datasets

| Dataset | Task Type | Samples Used |
|---|---|---|
| **GSM8K** | Mathematical reasoning | 100 |
| **CommonsenseQA** | Commonsense reasoning | 100 |

Both datasets are sourced from HuggingFace's `datasets` library. 100 questions were sampled from each using a fixed random seed (seed=42) to ensure reproducibility.

---

## Experimental Design

This study evaluates **4 strategies × 2 datasets = 8 experimental runs**, with 100 samples per run (~600 total inferences including Self-Consistency samples).

**Controlled Variables (held constant across all runs):**
- Same 100 questions per dataset (fixed random seed=42)
- `temperature=0` for Standard, Few-shot, and CoT (deterministic)
- `temperature=0.7` for Self-Consistency sampling only
- `max_new_tokens=512`
- Same answer extraction logic per dataset
- Same 3 fixed few-shot examples across all Few-shot runs

**Research Questions:**
1. Does Chain-of-Thought prompting improve accuracy over Standard Prompting?
2. Do in-context examples (Few-shot) provide comparable gains to reasoning instructions (CoT)?
3. Does Self-Consistency further improve CoT accuracy through sampling diversity?
4. Do the same strategies generalise across both mathematical and commonsense reasoning tasks?

---

## Metrics

- **Accuracy (%)** — primary metric, percentage of correctly answered questions
- **Consistency Rate (%)** — Self-Consistency only; percentage of questions where all 5 samples agreed on the same final answer

---

## Repository Structure

```
prompt-reasoning-at-scale/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sample_data.py               # Downloads and samples 100 questions from each dataset
│   ├── gsm8k_100.json               # 100 sampled GSM8K questions (seed=42)
│   └── csqa_100.json                # 100 sampled CommonsenseQA questions (seed=42)
│
├── prompts/
│   └── templates.py                 # Prompt templates for all 4 strategies
│
├── experiments/
│   ├── utils.py                     # Shared model loading, inference, answer extraction
│   ├── run_standard.py              # Zero-shot Standard experiments
│   ├── run_fewshot.py               # Few-shot Standard experiments
│   ├── run_cot.py                   # Zero-shot CoT experiments
│   └── run_self_consistency.py      # Self-Consistency CoT experiments
│
├── results/
│   ├── *_gemma-2b_*.csv             # Raw outputs and scores per run
│   ├── summary_table.csv            # Aggregated accuracy table
│   └── plot_*.png                   # Generated plots
│
└── analysis/
    └── evaluate.py                  # Accuracy calculation, summary table, and plots
```

---

## Results

| Strategy | GSM8K Accuracy (%) | CommonsenseQA Accuracy (%) | Notes |
|---|---|---|---|
| Zero-shot Standard | 8% | 47% | Baseline |
| Few-shot Standard | 6% | 43% | — |
| Zero-shot CoT | 12% | 34% | — |
| Self-Consistency CoT | 19% | 29% | Consistency: 45.60% / 70.60% |

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

### 3. Set up HuggingFace authentication
This project uses Google Colab's built-in Secrets Manager to store tokens securely.

1. In Colab, click the 🔑 **key icon** in the left sidebar
2. Add a secret named `HF_TOKEN` with your HuggingFace access token as the value
3. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Make sure your HuggingFace account has accepted the Gemma license at [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)

Then load it in your notebook before running any experiments:
```python
from google.colab import userdata
from huggingface_hub import login

login(userdata.get('HF_TOKEN'))
```

> ⚠️ Never hardcode your token directly in a notebook cell that gets committed to GitHub.

### 4. Sample data (run once)
```bash
python data/sample_data.py
```

### 5. Run all experiments
```bash
python experiments/run_standard.py --model gemma-2b
python experiments/run_fewshot.py --model gemma-2b
python experiments/run_cot.py --model gemma-2b
python experiments/run_self_consistency.py --model gemma-2b
```

### 6. Evaluate and generate plots
```bash
python analysis/evaluate.py
```

---

## Requirements

```
transformers>=4.40.0
datasets>=2.18.0
torch>=2.0.0
accelerate>=0.27.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
sentencepiece>=0.1.99
protobuf>=3.20.0
python-dotenv>=1.0.0
```

---

## Hardware Notes

All experiments were conducted on a free-tier Google Colab instance with an NVIDIA T4 GPU (15GB VRAM). Gemma-2B-IT runs stably on this hardware in float16 precision. Larger models such as Gemma-7B were considered for a scale-based comparison but could not be stably loaded on the available hardware without quantization support. This is acknowledged as a limitation in the associated term paper.

---

## References

- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
- Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
- Wei, J. et al. (2022). *Emergent Abilities of Large Language Models.* TMLR 2022.
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners.* NeurIPS 2020.
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems (GSM8K).* arXiv.
- Talmor, A. et al. (2019). *CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge.* NAACL 2019.
- Team, G. et al. (2024). *Gemma: Open Models Based on Gemini Research and Technology.* arXiv.

---

## Authors

- Mohammed Owais Khan
- Zaina Naaz Mohd Kalim Ansari

*Term paper submitted for the Trends in Machine Learning module.*

---

## License

MIT License — free to use, adapt, and build upon with attribution.
