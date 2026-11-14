# PhishRadar data cleaning report

**Output:** `train.csv`, `val.csv`, `test.csv`, each with the columns `text`, `label` (1 = phishing, 0 = legit) and `source`. These are an **intermediate** random split. The splits used in all experiments are built from them by `src/make_splits.py` (see "Splits" below and `data/README.md`).
**Script:** `src/clean_data.py` reproduces every step below (random seed 42).

## Input data

| Dataset | File(s) used | Emails |
|---|---|---:|
| Phishing Email Dataset (naserabdullahalam) | 6 source files: CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud, SpamAssassin | 82,486 |
| Phishing Email Detection (subhajournal) | Phishing_Email.csv | 18,650 |
| **Total** | | **101,136** |

**Why the 6 source files and not `phishing_email.csv`:** the combined file contains the same 82,486 emails, but they've already been lowercased and stripped of punctuation and common words like "for", "the" and "your". BERT is pre-trained on normal text, so this preprocessing throws away signals it uses (for example "URGENT!!!" or "Dear Customer,"). The source files keep the original text, and they also tell us which collection each email came from.

## What was removed and why

| Step | Removed | Why |
|---|---:|---|
| Missing text or `empty` placeholder | 552 | Nothing to learn from. The Detection dataset uses the literal word "empty" for 533 blank emails. |
| Mailbox system message | 2 | "FOLDER INTERNAL DATA" messages are mail-software artifacts, not emails, but were labeled phishing. |
| Too short (under 20 characters) | 23 | Fragments like "hi here" or "anderson jeromy" carry no usable signal. |
| Copy of an email already in the other dataset | 17,027 | 91% of the Detection dataset is a lowercased copy of the Enron, Ling and SpamAssassin emails already in the first dataset. All 17,027 copies had matching labels, so the original-case versions were kept. |
| Same email labeled both phishing and legit | 0 | Checked; no contradictions were found. |
| Exact duplicate | 673 | Identical emails. |
| Near duplicate | 575 | Emails identical except for capitalization, spacing or punctuation. |
| **Total removed** | **18,852** | |
| **Final dataset** | **82,284** | 42,669 phishing (51.9%), 39,615 legit |

Duplicates matter because a copy of an email that lands in both train and test lets the model "remember" the answer, which inflates scores.

## What was changed (but kept)

- **Subject + body combined** into one `text` field: subject first, then a blank line, then the body.
- **Encoding repaired** in 3,426 emails (garbled characters like `â€™` turned back into `'`) using the `ftfy` library.
- **Whitespace collapsed** to single spaces. BERT ignores line breaks and spacing anyway.
- **Length capped at 20,000 characters** for 415 very long newsletters and digests (the longest was 17 million characters). BERT reads only roughly the first 1,500 characters at `MAX_LENGTH = 256`, so nothing it uses was lost. The median email is 761 characters.
- **Not changed on purpose:** capitalization, punctuation, URLs, numbers. These are real phishing signals, and the BERT model lowercases text by itself. The TF-IDF approach can apply its own extra preprocessing on its copy.
- **Sender, receiver and date columns were dropped.** These columns only exist for some sources, and fields like CEAS's `@ceas-challenge.cc` addresses would let a model identify the source instead of learning about phishing.

## Splits

### Intermediate split (output of `clean_data.py`)

A random 80 / 10 / 10 split, stratified by **source and label**:

| Split | Emails | Phishing share |
|---|---:|---:|
| train | 65,827 | 51.9% |
| val | 8,228 | 51.8% |
| test | 8,229 | 51.9% |

**This split is not used for the experiments.** Train and test share the same sources, so scores on it are too optimistic (see caveat 3).

### Final splits (output of `src/make_splits.py`)

`make_splits.py` reads the three files above and builds the cross-source design used by all three approaches:

- **Dropped:** the 966 `PhishingEmailDetection` rows left over from the second dataset (it is a repackaged copy of the other corpora).
- **Train / val / in-source test:** Enron, Ling and SpamAssassin, split 80/10/10 with near-duplicate mass mailings kept together so templates cannot leak between splits.
- **Cross-source tests (never trained or tuned on):** CEAS_08, Nazario and Nigerian_Fraud, after removing 1 CEAS and 9 Nigerian emails that also appear in the training sources. A paired Nazario test adds an equal number of legitimate CEAS emails so precision and false-positive rate can be computed.
- **Checks:** the script asserts that no test email overlaps training data, exactly or by opening text.

Row counts for each final file are listed in `data/README.md`. **All three approaches (TF-IDF, DistilBERT, LLM) should be scored on these final files** so the comparison is fair.

## Emails by source (after cleaning, before `make_splits.py`)

| Source | Emails | Phishing | Legit | % phishing |
|---|---:|---:|---:|---:|
| CEAS_08 | 38,749 | 21,470 | 17,279 | 55.4% |
| Enron | 29,364 | 13,933 | 15,431 | 47.4% |
| SpamAssassin | 5,553 | 1,463 | 4,090 | 26.3% |
| Nigerian_Fraud | 3,252 | 3,252 | 0 | 100% |
| Ling | 2,856 | 458 | 2,398 | 16.0% |
| Nazario | 1,544 | 1,544 | 0 | 100% |
| PhishingEmailDetection | 966 | 549 | 417 | 56.8% |

`PhishingEmailDetection` is dropped later by `make_splits.py`, so it does not appear in the final experiments.

## Caveats worth mentioning in the write-up

1. **"Phishing" here mostly means spam.** In the Enron, CEAS, SpamAssassin and Ling collections, label 1 means spam, which includes plenty of ordinary ads (pharmacy, mortgage, dating) that are not phishing. Only Nazario (phishing) and Nigerian_Fraud (advance-fee scams) are strictly scams. The models are really learning "malicious or unwanted vs. legitimate."
2. **Source leakage risk.** Nazario and Nigerian_Fraud contain *only* phishing, and each collection has a recognizable style: Enron and Ling are lowercased with spaces around punctuation, while CEAS and SpamAssassin are in original case. In a quick check, a simple TF-IDF model could tell which collection an email came from with 95% accuracy. A model could therefore score well partly by recognizing the collection. Per-source scores, cross-source testing and a source-invariant training experiment (Experiment A) are the ways we test this.
3. **In-source scores are easy; cross-source scores are not.** A quick TF-IDF + logistic regression check on the intermediate random split scored about 98.7% accuracy (F1 0.987). On the final in-source test the TF-IDF baseline reaches F1 of about 0.99, but on unseen CEAS emails it drops to about 0.88. The interesting differences between approaches are therefore in generalization to new sources, the false-positive rate, the error analysis, and performance on AI-written phishing.
