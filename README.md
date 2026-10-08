# PhishRadar: AI-Based Phishing Email Detection

**CSC 4240 – Artificial Intelligence, Term Project (Fall 2026)**

PhishRadar is a head-to-head comparison of three AI approaches for detecting phishing emails: a classical ML model, a fine-tuned transformer, and a prompted LLM. The focus is on **rigorous evaluation**, especially **cross-dataset generalization**, rather than on a new model architecture.

## Team
| Member | Github |
|---|---|
| Connor Buttrey | [@cbuttrey0216](https://github.com/cbuttrey0216) |
| Oliver Conley | [@opconley97](https://github.com/opconley97) |
| Robert Morriss | [@rjmorriss](https://github.com/rjmorriss) |
| Ashley Smutnik | [@asmutnik](https://github.com/asmutnik) |
| Teagan Tobias | [@ttobias28](https://github.com/ttobias28) |

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

> **Pinned LLM version:** `TBD`

## Datasets

| Dataset | Role | Link |
|---|---|---|
| Phishing Email Dataset (naserabdullahalam), about 82,500 emails merged from Enron, Ling, CEAS, Nazario, Nigerian Fraud, and SpamAssassin | Primary training/testing; sub-sources used for cross-dataset evaluation | https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset |

### Evaluation Protocol
- **In-dataset:** stratified train/validation/test split.
- **Cross-source:** train on one sub-source, test on the others, reporting a full train-source × test-source matrix.
- **Metrics:** precision, recall, F1, false-positive rate, confusion matrices, plus error analysis of misclassified emails.

---

## Setup

### Requirements
- Python 3.10+
- A GPU is strongly recommended for BERT fine-tuning (free Google Colab GPU is sufficient)
- An API key for the chosen LLM provider (only needed for Approach C and adversarial generation)

---

## Limitations and Future Work

- Public phishing datasets can differ from real-world email traffic, and some sources are old (for example Enron-era corpora).
- Dataset sources may differ in formatting and style, which can inflate in-dataset scores. This is why cross-dataset evaluation is a central part of the study.
- LLM results depend on the specific model version and prompts used.
- *Future work:* URL/website-based detection as a secondary task, larger and more recent datasets, and testing on real-world mailboxes.

## Contributions

*(To be filled in as the project progresses. Each member's specific contributions are also recoverable from the Git commit history.)*

## License and Disclaimer

This project is for educational purposes as part of CSC 4240/5240. Phishing examples in this repository are used for research only and must not be used in live environments.
