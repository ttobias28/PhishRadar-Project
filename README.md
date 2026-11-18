# PhishRadar: AI-Based Phishing Email Detection

**CSC 4240 – Artificial Intelligence, Term Project (Fall 2026)**

PhishRadar compares three AI approaches for detecting phishing and scam emails: a classical ML model, a fine-tuned transformer, and a prompted LLM. The focus is on **rigorous evaluation**, especially **cross-source generalization**, rather than on a new model architecture.

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
 
Deceptive emails are increasingly well written, and attackers now use AI to generate them, so hand-coded rules are not enough to catch them reliably. In this project the AI model's prediction **is** the decision-maker. There is no rule-based system or optimizer underneath.
 
**Task:** classify an email (subject + body text) as `phishing/scam` (1) or `legitimate` (0).
 
**Definition used in this project:** we use a broad definition of phishing: deceptive emails designed to manipulate the recipient into acting against their own interest (clicking, paying, or giving up information). Our training data is a mix of spam, advance-fee fraud, and credential phishing (see Dataset below), so we describe the task as *phishing and scam detection*.
 
**Research questions**
1. How do classical ML, a fine-tuned DistilBERT model, and a prompted LLM compare on precision, recall, F1, and false-positive rate?
2. How well does each approach **generalize across sources** (train on some email corpora, test on others), and why do detectors fail on new sources?
3. Does a model trained on mostly spam and scam email also catch **true credential phishing**?
4. How sensitive is the LLM approach to **prompt wording** (plain zero-shot, detailed zero-shot, few-shot, intent-focused)?
5. **Experiment A:** can **source-invariant training** (making DistilBERT unable to tell which corpus an email came from) improve cross-source generalization? 
6. **Experiment B ("AI polish"):** do detectors catch phishing because of *what it says* or because it *looks sloppy*? An LLM copy-edits real phishing emails, and we compare each model's detections before and after.

## Approaches Compared
 
| # | Approach | Training? | Details |
|---|---|---|---|
| A | **Classical ML** | Trained from scratch | TF-IDF features + Logistic Regression (scikit-learn) |
| B | **Fine-tuned DistilBERT** | Fine-tuned from pretrained | DistilBERT + classification head (Hugging Face Transformers, PyTorch) |
| C | **LLM prompting** | None (zero/few-shot) | One fixed-version LLM, 4 prompt variants (plain zero-shot, detailed zero-shot, few-shot, intent-focused) |
 
> **Pinned LLM versions:** classifier `claude-haiku-5-5` (temperature 0; the four prompts in `prompts/` were frozen before the test runs, with fingerprints in `results/llm_prompt_hashes.json`). Experiment B rewriter: `claude-sonnet-5-5` (prompt `prompts/rewrite_polish.txt`).
 
## Dataset

**Primary dataset:** [Phishing Email Dataset (Kaggle, naserabdullahalam)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset): 82,486 emails from six public corpora (Enron, Ling, SpamAssassin, CEAS_08, Nazario, Nigerian Fraud). The source files, splits, schema and known limitations are described in [`data/README.md`](data/README.md). The emails themselves are not stored in this repo.

**Evaluation protocol**
- **In-source:** Enron, Ling, and SpamAssassin are split 80/10/10 into train/validation/test, with near-duplicate mass-mailed emails kept together so templates cannot leak between splits.
- **Cross-source (main experiment):** models train on Enron + Ling + SpamAssassin and are tested on CEAS_08, Nazario, and Nigerian_Fraud, which they never saw. Nazario and Nigerian contain no legitimate emails, so only recall is reported on them. Nazario paired 50/50 with held-out legitimate CEAS emails measures precision and false-positive rate.
- **Metrics:** precision, recall, F1, false-positive rate, confusion matrices, and error analysis of misclassified emails.

---

## Results

All numbers are in `results/` (summary tables) and `results/predictions/` (one file per model and test set: `row_id, y_true, y_pred, score`). Precision, F1 and FPR are not defined on Nazario and Nigerian because they contain only phishing, so only recall is reported there.

### Cross-source results (positive class = phishing/scam)

| Approach | In-source recall | In-source F1 | In-source FPR | CEAS recall | CEAS FPR | Nazario recall | Nigerian recall |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9912 | 0.9899 | 0.0082 | 0.8500 | 0.0896 | 0.7902 | 0.9935 |
| DistilBERT (full training set) | 0.9918 | 0.9906 | 0.0078 | 0.8537 | 0.0645 | 0.8128 | 0.9935 |
| LLM, plain zero-shot † | 0.9179 | 0.9524 | 0.0068 | 0.7588 | 0.0329 | 0.9780 | 0.9980 |
| LLM, detailed zero-shot † | 0.8937 | 0.9415 | 0.0034 | 0.7588 | 0.0288 | 0.9780 | 0.9980 |
| LLM, few-shot † | 0.8986 | 0.9418 | 0.0068 | 0.7588 | 0.0288 | 0.9840 | 0.9980 |
| LLM, intent-focused † | 0.8792 | 0.9333 | 0.0034 | 0.7588 | 0.0165 | 0.9760 | 0.9960 |

† The LLM was scored on a fixed random sample of 500 emails per test set (row IDs in `results/llm_sample_ids.csv`), not on the full sets. Emails the model refused to answer (up to 3.4% on the in-source set) are scored as predicted legitimate.

**What stands out**
* All three approaches score about 99% in-source and drop on unseen sources. DistilBERT misses about 19% of Nazario phishing and flags 6.5% of legitimate CEAS emails; TF-IDF is slightly worse on both.
* The LLM catches far more Nazario phishing (about 0.98 recall) but misses more of CEAS (0.76). The four prompts differ little, and most gaps are within sampling noise at 500 emails per set.
* TF-IDF diagnostics: holding out Enron drops Nazario recall to 0.29 (vs. 0.79), and removing style cues lowers CEAS recall from 0.85 to 0.78 (`results/tfidf_leave_one_source_out.csv`, `results/tfidf_normalized_vs_original.csv`).

### Experiment A: source-invariant training

The training sources differ in style and in their share of phishing, so "which collection is this from?" is a shortcut that partly predicts the label. Experiment A tests whether removing it helps, using three DistilBERT versions with identical settings: `baseline`, `balanced` (each source-and-label combination weighted equally) and `source_invariant` (balanced, plus an adversary that tries to guess the source while DistilBERT learns to make it fail). Notebook: `PhishRadar_DistilBERT_SourceInvariant.ipynb`.

| Version | In-source F1 | CEAS F1 | CEAS FPR | Nazario recall | Nazario-paired F1 | Nigerian recall | Source probe † |
|---|---|---|---|---|---|---|---|
| Baseline | 0.9893 | 0.8990 | 0.0621 | 0.8238 | 0.8712 | 0.9938 | 0.791 |
| Balanced | 0.9899 | 0.8919 | 0.0637 | 0.8199 | 0.8701 | 0.9935 | 0.766 |
| Source-invariant | 0.9861 | 0.8895 | 0.0662 | 0.7947 | 0.8527 | 0.9932 | 0.792 |

† How well a simple classifier can still guess the source from the model's internal representation (chance is about 0.33). Lower means less source information.

**Result: removing the shortcut did not help.** Balancing gave no gain on Nazario (McNemar p = 0.57) and a small drop in CEAS F1 (p < 0.001). The adversarial version was significantly worse on Nazario recall (0.7947 vs. 0.8238, p = 0.0002), and its probe score did not drop, so it did not remove the source information. Full tables: `results/si_*.csv`.

**Caveats:** one seed per version; this baseline is a separate retrain, so it differs slightly from the main DistilBERT row; and the adversary's learning rate and size were adjusted once after seeing quick-run test scores (all runs are logged in the notebook).

### Experiment B: AI polish

An LLM (`claude-sonnet-5-5`) copy-edited 150 Nazario and 150 Nigerian phishing emails (spelling, grammar, formatting), with 300 legitimate CEAS emails as a control. Each model scored the original and the rewrite, and a McNemar test checks whether detections change.

| Model | Phishing recall (original → polished) | Caught→missed / missed→caught | McNemar p | FPR on legitimate (original → polished) |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.9773 → 0.9830 | 0 / 1 | 1.00 | 0.0842 → 0.0875 |
| DistilBERT | 0.9830 → 0.9830 | 0 / 0 | 1.00 | 0.0640 → 0.0640 |
| LLM, plain zero-shot | 0.9943 → 0.9943 | 0 / 0 | 1.00 | 0.0101 → 0.0135 |
| LLM, detailed zero-shot | 0.9943 → 0.9943 | 0 / 0 | 1.00 | 0.0034 → 0.0034 |
| LLM, few-shot | 0.9943 → 0.9886 | 1 / 0 | 1.00 | 0.0135 → 0.0135 |
| LLM, intent-focused | 0.9943 → 0.9943 | 0 / 0 | 1.00 | 0.0101 → 0.0034 |

176 phishing pairs (149 Nigerian, 27 Nazario) and 297 legitimate pairs.

**Result: no measurable effect.** Every McNemar p = 1.00, with at most one email changing.

**Caveats:** the rewriter refused 127 of 600 emails (123 of the 150 Nazario), so Nazario rests on only 27 pairs; Nigerian recall was already about 99%, leaving little room for a drop; and the edit was copy-editing only. A manual check that the rewrites kept the original intent has been completed.

---

## Setup

**Requirements:** Python 3.10+ and `pip install -r requirements.txt` for local runs. The notebooks are written for Google Colab; only DistilBERT needs a GPU (Runtime → Change runtime type → T4).

**1. Build the data** (the emails are not stored in this repo). Download the two Kaggle datasets into `data/raw/` as described in [`data/README.md`](data/README.md), then run:

```
python src/clean_data.py data/raw/phishing_email_dataset data/raw/phishing_email_detection data/cleaned
python src/make_splits.py data/cleaned data/splits
```

For Colab, put the resulting CSVs in `MyDrive/phishradar/data/splits/` (the notebooks read and write under `MyDrive/phishradar/`).

**2. Run the notebooks in this order** (each one: open in Colab, then Runtime → Run all):

| Step | Notebook | Needs | Produces |
|---|---|---|---|
| 1 | `PhishRadar_tfidf_baseline.ipynb` | CPU | `results/predictions/tfidf_*.csv`, TF-IDF tables |
| 2 | `PhishRadar_DistilBERT.ipynb` | T4 GPU, splits in Drive | `results/predictions/distilbert_*.csv` |
| 3 | `PhishRadar_DistilBERT_SourceInvariant.ipynb` (Experiment A) | T4 GPU (about 2 hours), splits in Drive | `results/predictions/distilbert_{baseline,balanced,source_invariant}_*.csv`, `results/si_*` |
| 4 | `PhishRadar_LLM.ipynb` | Anthropic API key | `results/predictions/llm_*.csv`, Experiment B rewrites (`b_data/`) |
| 5 | Experiment B sections at the end of the TF-IDF and DistilBERT notebooks | output of step 4 | `results/b_predictions/` |
| 6 | `PhishRadar_comparison.ipynb` | CPU, all prediction files above (steps 1–5) | comparison tables, McNemar tests, charts, error analysis |

The LLM notebook reads `ANTHROPIC_API_KEY` from Colab Secrets or a local `.env` file (see `.env.example`; never commit your key). The full run makes roughly 15,000 paid API calls, cached in your Drive so re-runs are free. **To check the results without re-running anything**, the prediction files are already in `results/`, so you can run only step 6.

**Colab folder layout.** All five notebooks read and write under `MyDrive/phishradar/`:
* `data/splits/`: the split CSVs from step 1
* `results/predictions/` and `results/b_predictions/`: prediction files written by the model notebooks and read by the comparison notebook
* `b_data/`: Experiment B rewrites (from the LLM notebook)
* `models/`: saved DistilBERT models (not committed; about 250 MB each)

Allow the Google Drive connection when a notebook asks.

**Reproducing without retraining.** Copy this repo's `results/predictions/` and `results/b_predictions/` folders into `MyDrive/phishradar/results/`, put the splits in `MyDrive/phishradar/data/splits/`, then run `PhishRadar_comparison.ipynb`.

**Sanity check.** A correct run gives about: TF-IDF in-source F1 0.99 and CEAS recall 0.85; DistilBERT Nazario recall 0.81; LLM Nazario recall 0.98. GPU training can differ from our numbers by about a point.

**Time and cost (approximate).** TF-IDF: minutes on CPU. DistilBERT: about 10 minutes for a quick test (SAMPLE_SIZE = 5000), and about 20 minutes for the full run on all ~30,000 training emails on a T4. Experiment A: about 2 hours on a T4. LLM: about 15,000 paid API calls (about $5.00), cached afterwards. Comparison: a few minutes.

Tested with Python 3.13, torch 2.14, transformers 5.22. `requirements.txt` is not version-pinned.

---

## Limitations

- The positive class is mostly spam and scam rather than credential phishing, and several sources are old (Enron-era corpora).
- Source corpora differ in style and formatting, which can inflate in-source scores. This is why cross-source evaluation is central to the study.
- Two sources contain only positive emails, which limits what can be measured on them.
- LLM results depend on the specific model version and prompts used. The LLM was scored on 500-email samples, and a few percent of emails were refused by the API's safety filter (counted as predicted legitimate).

## License and Disclaimer

This project is for educational purposes as part of CSC 4240. Phishing examples in this repository are used for research only and must not be used in live environments.
