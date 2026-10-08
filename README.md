# PhishRadar: AI-Based Phishing Email Detection

**CSC 4240 – Artificial Intelligence, Term Project (Fall 2026)**

PhishRadar is a head-to-head comparison of three AI approaches for detecting phishing emails: a classical ML model, a fine-tuned transformer, and a prompted LLM. The focus is on **rigorous evaluation**, especially **cross-dataset generalization**, rather than on a new model architecture.

## Team

| Member 
|---|---|
| Connor Buttrey
| Oliver Conley
| Robert Morriss
| Ashley Smutnik
| Teagan Tobias

*(Detailed contributions are documented in the final report.)*

---

## Overview

Phishing emails are increasingly well written, and attackers now use AI to generate them, so hand-coded rules are not enough to catch them reliably. In this project the AI model's prediction **is** the decision-maker. There is no rule-based system or optimizer underneath.

**Primary task:** classify an email (subject + body text) as `phishing` or `legitimate`.

**Research questions**
1. How do classical ML, a fine-tuned BERT model, and a prompted LLM compare on precision, recall, F1, and false-positive rate?
2. How well does each approach **generalize across datasets** (train on one source, test on another)?
3. How sensitive is the LLM approach to **prompt wording** (zero-shot vs. few-shot)?
4. *(Stretch)* Do the models hold up against **LLM-generated phishing emails**?

## Approaches Compared

| # | Approach | Training? | Details |
|---|---|---|---|
| A | **Classical ML** | Trained from scratch | TF-IDF features + Logistic Regression / SVM (scikit-learn) |
| B | **Fine-tuned BERT** | Fine-tuned from pretrained | `bert-base-uncased` + classification head (Hugging Face Transformers, PyTorch) |
| C | **LLM prompting** | None (zero/few-shot) | One fixed-version LLM, 3–4 prompt variants (plain zero-shot, detailed zero-shot, few-shot) |

> **Pinned LLM version:** `TBD` (set in `src/llm_classifier.py`)

## Datasets

| Dataset | Role | Link |
|---|---|---|
| Phishing Email Dataset (naserabdullahalam), about 82,500 emails merged from Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin | Primary training/testing; sub-sources used for cross-dataset evaluation | https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset |
| Phishing Email Detection (subhajournal), about 18.6k emails | Independent external test set | https://www.kaggle.com/datasets/subhajournal/phishingemails |

See [`data/README.md`](data/README.md) for download instructions and expected file locations. **Raw data is not committed to this repo.**

### Evaluation Protocol
- **In-dataset:** stratified train/validation/test split.
- **Cross-source:** train on one sub-source, test on the others, reporting a full train-source × test-source matrix.
- **External:** evaluate all models on the independent second dataset.
- **Adversarial (stretch):** evaluate on LLM-generated phishing emails stored in `data/adversarial/`.
- **Metrics:** precision, recall, F1, false-positive rate, confusion matrices, plus error analysis of misclassified emails.

---

## Repository Structure

```
phishradar/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── README.md          # dataset download instructions
│   ├── raw/               # downloaded files (not committed)
│   ├── processed/         # cleaned/split data (not committed)
│   └── adversarial/       # LLM-generated phishing emails
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classical_ml.ipynb
│   ├── 03_bert_finetuning.ipynb
│   ├── 04_llm_prompting.ipynb
│   ├── 05_cross_dataset_eval.ipynb
│   └── 06_adversarial_and_error_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── classical.py
│   ├── bert_model.py
│   ├── llm_classifier.py
│   └── evaluation.py
├── prompts/               # each LLM prompt variant as a text file
├── results/
│   ├── figures/
│   ├── tables/
│   └── llm_outputs/       # saved raw LLM responses
└── docs/
```

Core logic lives in `src/`; the notebooks import from it and walk through each experiment in order.

---

## Setup

### Requirements
- Python 3.10+
- A GPU is strongly recommended for BERT fine-tuning (free Google Colab GPU is sufficient)
- An API key for the chosen LLM provider (only needed for Approach C and adversarial generation)

### Installation

```bash
git clone https://github.com/<your-org>/phishradar.git
cd phishradar
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API Key

```bash
cp .env.example .env
# then edit .env and add your key, e.g.
# ANTHROPIC_API_KEY=your_key_here
```

Never commit `.env`. Each team member uses their own key.

### Data
Download the datasets as described in [`data/README.md`](data/README.md) and place them in `data/raw/`.

---

## How to Run

Run the notebooks in order from the `notebooks/` folder:

1. **`01_data_exploration`**: load, clean, and split the data; inspect class balance and per-source statistics.
2. **`02_classical_ml`**: train and tune TF-IDF + Logistic Regression/SVM. *Runs in minutes on CPU.*
3. **`03_bert_finetuning`**: fine-tune BERT. **Run on a GPU** (Colab recommended; about 20 min to 2 hrs depending on data size and settings).
4. **`04_llm_prompting`**: run the prompt variants against the fixed LLM version. *Makes API calls, so outputs are cached in `results/llm_outputs/`.*
5. **`05_cross_dataset_eval`**: build the cross-source and external-dataset result matrices.
6. **`06_adversarial_and_error_analysis`**: test on LLM-generated emails and analyze misclassifications.

**Running on Colab:** open any notebook in Colab, run the first cell to clone the repo and install dependencies, and set your API key through Colab secrets.

**Reproducibility:** random seeds are fixed in `src/` (default `42`). LLM responses are saved to disk so analysis can be re-run without repeating API calls.

---

## Results

*To be completed.*

| Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| TF-IDF + LR/SVM | – | – | – | – |
| Fine-tuned BERT | – | – | – | – |
| LLM (best prompt) | – | – | – | – |

Cross-dataset generalization matrices, prompt-sensitivity results, and error analysis will be added to `results/` and summarized here.

## Limitations and Future Work

- Public phishing datasets can differ from real-world email traffic, and some sources are old (for example Enron-era corpora).
- Dataset sources may differ in formatting and style, which can inflate in-dataset scores. This is why cross-dataset evaluation is a central part of the study.
- LLM results depend on the specific model version and prompts used.
- *Future work:* URL/website-based detection as a secondary task, larger and more recent datasets, and testing on real-world mailboxes.

## Contributions

*(To be filled in as the project progresses. Each member's specific contributions are also recoverable from the Git commit history.)*

## License and Disclaimer

This project is for educational purposes as part of CSC 4240/5240. Phishing examples in this repository are used for research only and must not be used in live environments.
