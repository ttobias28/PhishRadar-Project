"""PhishRadar data cleaning: merges the two Kaggle datasets into clean train/val/test splits.
Usage: python clean_data.py <folder with the 6 source CSVs> <folder with Phishing_Email.csv> <output folder>
"""
import json, re, sys, unicodedata
from collections import OrderedDict
import pandas as pd
import ftfy
from sklearn.model_selection import train_test_split

Z1, Z2, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = 42
MIN_CHARS = 20
log = OrderedDict()   # step -> rows removed
def note(step, n): log[step] = log.get(step, 0) + int(n)

# ---------- 1. Load ----------
frames = []
raw_counts = {}
for src in ["CEAS_08", "Enron", "Ling", "Nazario", "Nigerian_Fraud", "SpamAssasin"]:
    d = pd.read_csv(f"{Z1}/{src}.csv")
    raw_counts[src] = len(d)
    subj = d["subject"].fillna("").astype(str).str.strip()
    body = d["body"].fillna("").astype(str)
    frames.append(pd.DataFrame({
        "text": (subj + "\n\n" + body).str.strip(),
        "body_only": body,
        "label": d["label"].astype(int),
        "source": src if src != "SpamAssasin" else "SpamAssassin",
        "dataset": "Phishing Email Dataset",
    }))
d2 = pd.read_csv(f"{Z2}/Phishing_Email.csv")
raw_counts["Phishing Email Detection"] = len(d2)
frames.append(pd.DataFrame({
    "text": d2["Email Text"],
    "body_only": d2["Email Text"],
    "label": d2["Email Type"].map({"Phishing Email": 1, "Safe Email": 0}),
    "source": "PhishingEmailDetection",
    "dataset": "Phishing Email Detection",
}))
df = pd.concat(frames, ignore_index=True)
n_raw = len(df)
assert df["label"].notna().all()

# ---------- 2. Missing / placeholder text ----------
mask = df["text"].isna() | df["text"].astype(str).str.strip().str.lower().isin(["", "empty", "nan"])
note("Missing text or 'empty' placeholder", mask.sum())
df = df[~mask].copy()

# ---------- 3. Repair text ----------
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏﻿]")
def clean_text(t):
    t = ftfy.fix_text(str(t))                 # fixes mojibake like "â€™" -> "'"
    t = unicodedata.normalize("NFKC", t)
    t = CTRL.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()        # BERT ignores whitespace layout anyway
    return t
before_fix = df["text"].astype(str)
df["text"] = before_fix.map(clean_text)
df["body_only"] = df["body_only"].astype(str).map(clean_text)
n_fixed = int((before_fix.map(lambda t: re.sub(r"\s+", " ", t).strip()) != df["text"]).sum())

# ---------- 4. Junk rows ----------
mask = df["text"].str.contains("FOLDER INTERNAL DATA", case=False, regex=False)
note("Mailbox system message (not a real email)", mask.sum())
df = df[~mask]
mask = df["text"].str.len() < MIN_CHARS
note(f"Too short (< {MIN_CHARS} characters)", mask.sum())
short_examples = df.loc[mask, "text"].head(8).tolist()
df = df[~mask]

# ---------- 5. Duplicates ----------
def key(t):  # lowercase, letters+digits only: catches copies that differ in case/spacing/punctuation
    return re.sub(r"[^a-z0-9]+", "", t.lower())
df["key"] = df["text"].map(key)
df["body_key"] = df["body_only"].map(key)

# 5a. Cross-dataset: Detection rows that already exist in the Phishing Email Dataset
is_d2 = df["dataset"] == "Phishing Email Detection"
d1_keys = set(df.loc[~is_d2, "key"]) | set(df.loc[~is_d2, "body_key"])
cross = is_d2 & df["key"].isin(d1_keys)
# label agreement among cross-dataset copies
k2l = {}
for k, l in zip(df.loc[~is_d2, "key"], df.loc[~is_d2, "label"]): k2l.setdefault(k, set()).add(l)
for k, l in zip(df.loc[~is_d2, "body_key"], df.loc[~is_d2, "label"]): k2l.setdefault(k, set()).add(l)
agree = sum(l in k2l.get(k, set()) for k, l in zip(df.loc[cross, "key"], df.loc[cross, "label"]))
cross_label_check = (int(agree), int(cross.sum()))
note("Copy of an email already in the other dataset", cross.sum())
df = df[~cross]

# 5b. Conflicting labels: same email labeled both phishing and legit
lab_per_key = df.groupby("key")["label"].nunique()
conflict_keys = set(lab_per_key[lab_per_key > 1].index)
mask = df["key"].isin(conflict_keys)
note("Same email labeled both phishing and legit (all copies dropped)", mask.sum())
df = df[~mask]

# 5c. Exact and near duplicates (same label)
exact = df.duplicated(subset="text")
note("Exact duplicate", exact.sum())
df = df[~exact]
near = df.duplicated(subset="key")
note("Near duplicate (differs only in case/spacing/punctuation)", near.sum())
df = df[~near]

df = df[["text", "label", "source"]].reset_index(drop=True)

# 5d. Cap extreme lengths (huge newsletters/digests). BERT reads only the first ~1,500 chars anyway.
MAX_CHARS = 20000
n_truncated = int((df["text"].str.len() > MAX_CHARS).sum())
df["text"] = df["text"].str.slice(0, MAX_CHARS)
dup_after_cut = df["text"].map(key).duplicated()   # two long emails can match once cut
note("Near duplicate (differs only in case/spacing/punctuation)", dup_after_cut.sum())
df = df[~dup_after_cut].reset_index(drop=True)

# ---------- 6. Split 80/10/10, stratified by source+label ----------
strat = df["source"] + "_" + df["label"].astype(str)
train, temp = train_test_split(df, test_size=0.2, stratify=strat, random_state=SEED)
val, test = train_test_split(temp, test_size=0.5, stratify=strat.loc[temp.index], random_state=SEED)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part.sample(frac=1, random_state=SEED).to_csv(f"{OUT}/{name}.csv", index=False)

# ---------- 7. Stats for report ----------
lens = df["text"].str.len()
stats = {
    "raw_counts": raw_counts, "n_raw": n_raw, "removed": dict(log), "n_final": len(df),
    "n_fixed_encoding": n_fixed, "n_truncated": n_truncated,"cross_label_agree": cross_label_check,
    "short_examples": short_examples,
    "by_source": df.groupby("source")["label"].agg(total="count", phishing="sum").assign(
        legit=lambda x: x.total - x.phishing, pct_phishing=lambda x: (100 * x.phishing / x.total).round(1)
    ).reset_index().to_dict("records"),
    "label_counts": df["label"].value_counts().to_dict(),
    "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
    "split_phish_pct": {n: round(100 * p["label"].mean(), 1) for n, p in [("train", train), ("val", val), ("test", test)]},
    "length": {"median": int(lens.median()), "p90": int(lens.quantile(0.9)), "max": int(lens.max())},
}
json.dump(stats, open(f"{OUT}/_stats.json", "w"), indent=1, default=int)
print(json.dumps(stats, indent=1, default=int))
