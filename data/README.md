# Data

**The data files are not stored in this repository.** This page explains how to get them and how to rebuild the exact train/test splits used in the project.


### 1. Download

|Dataset|Link|Files to use|
|-|-|-|
|Phishing Email Dataset (naserabdullahalam)|https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset|`CEAS_08.csv`, `Enron.csv`, `Ling.csv`, `Nazario.csv`, `Nigerian_Fraud.csv`, `SpamAssasin.csv` into `data/raw/phishing_email_dataset/`|
|Phishing Email Detection (subhajournal)|https://www.kaggle.com/datasets/subhajournal/phishingemails|`Phishing_Email.csv` into `data/raw/phishing_email_detection/`|

**Source files in the primary dataset**

|Source file|Rows|Notes|
|-|-:|-|
|`CEAS_08.csv`|39,154|Spam + legitimate; has sender, receiver, date, urls|
|`Enron.csv`|29,767|Spam + legitimate; subject, body only|
|`Ling.csv`|2,859|Spam + legitimate; subject, body only|
|`SpamAssasin.csv`|5,809|Spam + legitimate; has metadata columns|
|`Nazario.csv`|1,565|**Phishing only**; real credential-phishing emails|
|`Nigerian_Fraud.csv`|3,332|**Phishing only**; advance-fee fraud|
|`phishing_email.csv`|82,486|Combined file (`text_combined`, `label`, no source column); not used|

**Use the six per-source files, not the combined `phishing_email.csv`.** The combined file is already lowercased and stripped of punctuation, and it has no source column.

Note: the second dataset is a repackaged copy of material from the first (about 91% of its emails are duplicates). The cleaning script removes those copies, and the final splits contain no rows from it. It is only read as an input to the cleaning step.

### 2. Clean

```
python src/clean_data.py data/raw/phishing_email_dataset data/raw/phishing_email_detection data/cleaned
```

This removes missing and placeholder emails, repairs garbled text, drops duplicates, and writes `train.csv`, `val.csv`, and `test.csv` (columns `text`, `label`, `source`) plus a stats file. See `docs/cleaning_report.md` for every step.

### 3. Build the cross-source splits

```
python src/make_splits.py data/cleaned data/splits
```

Random seed is fixed (42), so everyone gets identical files.

## Final splits (`data/splits/`)

All files have the columns `text`, `label` (1 = phishing/scam/spam, 0 = legitimate) and `source`.

|File|Rows|Source(s)|Purpose|
|-|-:|-|-|
|`train.csv`|30,217|Enron, Ling, SpamAssassin|Training|
|`val.csv`|3,778|Enron, Ling, SpamAssassin|Tuning and prompt development|
|`test_insource.csv`|3,778|Enron, Ling, SpamAssassin|Easy baseline (same sources as training)|
|`test_ceas.csv`|38,748|CEAS_08|Cross-source test, both classes|
|`test_nazario.csv`|1,544|Nazario|True credential phishing (all label 1, so recall only)|
|`test_nigerian.csv`|3,243|Nigerian_Fraud|Advance-fee scams (all label 1, so recall only)|
|`test_nazario_paired.csv`|3,088|Nazario + CEAS legit|Nazario paired with an equal number of legit CEAS emails, so precision and false-positive rate exist|

**Rules for using the splits**

* Train only on `train.csv`. Tune only on `val.csv`.
* Never train or tune on any `test_*` file.
* Training sources and cross-source test sets share no emails, exact or near-duplicate (checked by the script).

## Known limitations

* **The "phishing" label is mostly spam.** In Enron, CEAS, SpamAssassin, and Ling, label 1 means spam, including ordinary ads. Only Nazario (credential phishing) and Nigerian_Fraud (advance-fee scams) are strictly scams. We therefore describe the task as phishing and scam detection.
* **Source style differs.** Enron and Ling are lowercased and tokenized, while the other sources keep original capitalization and punctuation. Models can pick up on style, so we report results per source.
* **Nazario and Nigerian_Fraud contain no legitimate emails.** We report recall only on them.
* **The data is old.** Enron-era and mid-2000s spam corpora do not reflect current phishing campaigns.

## Citations and licensing

* Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (2024). *Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection.* arXiv:2405.11619. (The Kaggle page for the Phishing Email Dataset asks users to cite this.)
* The Phishing Email Dataset combines the Enron, Ling-Spam, CEAS-08, Nazario, Nigerian Fraud, and SpamAssassin corpora. Please see the Kaggle pages above for each dataset's license and original credits.

## Responsible use

These emails are for research and education only. Do not use them, or any phishing examples in `adversarial/`, in live environments.
