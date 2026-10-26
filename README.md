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
2. How well does each approach **generalize across sources** (train on some email corpora, test on others)?
3. Does a model trained on mostly spam and scam email also catch **true credential phishing**?
4. How sensitive is the LLM approach to **prompt wording** (zero-shot vs. few-shot)?
5. *(Stretch)* Do the models hold up against **LLM-generated phishing emails**?

## Approaches Compared
 
| # | Approach | Training? | Details |
|---|---|---|---|
| A | **Classical ML** | Trained from scratch | TF-IDF features + Logistic Regression / SVM (scikit-learn) |
| B | **Fine-tuned DistilBERT** | Fine-tuned from pretrained | DistilBERT + classification head (Hugging Face Transformers, PyTorch) |
| C | **LLM prompting** | None (zero/few-shot) | One fixed-version LLM, 3–4 prompt variants (plain zero-shot, detailed zero-shot, few-shot) |
 
> **Pinned LLM version:** `TBD`
 
## Dataset
 
**Primary dataset:** [Phishing Email Dataset (Kaggle, naserabdullahalam)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset): 82,486 emails (42,891 label 1 / 39,595 label 0) from six public corpora, distributed as one combined file plus one file per source.
 
| Source file | Rows | Notes |
|---|---|---|
| `CEAS_08.csv` | 39,154 | Spam + legitimate; has sender, receiver, date, urls |
| `Enron.csv` | 29,767 | Spam + legitimate; subject, body only |
| `Ling.csv` | 2,859 | Spam + legitimate; subject, body only |
| `SpamAssasin.csv` | 5,809 | Spam + legitimate; has metadata columns |
| `Nazario.csv` | 1,565 | **Phishing only**; real credential-phishing emails |
| `Nigerian_Fraud.csv` | 3,332 | **Phishing only**; advance-fee fraud |
| `phishing_email.csv` | 82,486 | Combined file: `text_combined`, `label` (no source column) |

### Processed schema
`text` (subject + body), `label` (0/1), `source` (which corpus the email came from). The combined file has no source column, so we rebuild it from the per-source files.
 
### Evaluation Protocol
- **In-source:** Enron, Ling, and SpamAssassin are split into train/validation/test (80/10/10), with near-duplicate mass-mailed emails kept together so templates cannot leak between splits.
- **Cross-source (main experiment):** models train on Enron + Ling + SpamAssassin and are tested on CEAS_08, Nazario, and Nigerian_Fraud, which they never saw. Nazario and Nigerian contain no legitimate emails, so we report recall on them. We also test Nazario paired 50/50 with held-out legitimate CEAS emails to measure precision and false-positive rate.
- **Optional extra true-phishing test:** real phishing samples from phishing_pot, if time allows.
- **Adversarial (stretch):** LLM-generated phishing emails.
- **Metrics:** precision, recall, F1, false-positive rate, confusion matrices, and error analysis of misclassified emails.

**Data notes:** the source files differ in formatting (Enron and Ling are already lowercased and tokenized; the others keep their original case and punctuation). We keep this as is and report results per source, since style differences can act as a shortcut for models. An additional 18,650-email Kaggle phishing dataset was explored but excluded: about 89% of its emails also appear in the Enron, Ling, and SpamAssassin files here, so it would leak data between train and test.

---

## Setup

1. **Get the data.** The CSVs are not stored in this repo. Follow the instructions in `data/README.md` to download the source datasets and rebuild the splits, or get the ready-made splits from the team.
2. **Run a notebook in Google Colab** (recommended): upload the notebook (File → Upload notebook) and run the cells top to bottom. The TF-IDF notebook reads the CSVs from Google Drive at `MyDrive/phishradar/data/splits/` and needs no GPU. The DistilBERT notebook needs a GPU (Runtime → Change runtime type → T4 GPU) and asks you to upload the splits zip when you run it.
3. **Or run locally** with Python 3.10+: `pip install -r requirements.txt`, put the CSVs in `data/splits/`, and open the notebook in Jupyter or VS Code.
4. **LLM approach:** needs an API key in a local `.env` file (see `.env.example`). Never commit it.

### Requirements
- Python 3.10+
- A GPU is strongly recommended for DistilBERT fine-tuning (free Google Colab GPU is sufficient)
- An API key for the chosen LLM provider (only needed for Approach C and adversarial generation)

---

## Limitations and Future Work

- The positive class is mostly spam and scam rather than credential phishing, and several sources are old (Enron-era corpora).
- Source corpora differ in style and formatting, which can inflate in-source scores. This is why cross-source evaluation is central to the study.
- Two sources contain only positive emails, which limits what can be measured on them.
- LLM results depend on the specific model version and prompts used.
- *Future work:* URL/website-based detection as a secondary task, more recent real-world phishing data, and testing on live mailboxes.

## Contributions

*(To be filled in as the project progresses. Each member's specific contributions are also recoverable from the Git commit history.)*

## License and Disclaimer

This project is for educational purposes as part of CSC 4240. Phishing examples in this repository are used for research only and must not be used in live environments.
