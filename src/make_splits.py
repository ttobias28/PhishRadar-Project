"""PhishRadar: build cross-source splits from the cleaned train/val/test CSVs.

Usage: python make_splits.py <folder with cleaned train.csv/val.csv/test.csv> <output folder>

Design (matches the project plan):
  TRAIN SOURCES : Enron, Ling, SpamAssassin  -> train / val / in-source test (80/10/10, group-aware)
  CROSS-SOURCE TESTS (never trained or tuned on): CEAS_08, Nazario, Nigerian_Fraud
Fixes vs. the cleaned random split:
  * drops the leftover 'PhishingEmailDetection' rows (it is a re-packaged copy of the other corpora)
  * splits by opening-text group so mass-mailed templates cannot sit in both train and test
  * removes cross-source test emails whose text (or opening text) also appears in training data
"""
import re, sys, json
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

IN, OUT = sys.argv[1], sys.argv[2]
SEED = 42
TRAIN_SOURCES = ["Enron", "Ling", "SpamAssassin"]
TEST_SOURCES = ["CEAS_08", "Nazario", "Nigerian_Fraud"]

df = pd.concat([pd.read_csv(f"{IN}/{n}.csv") for n in ["train", "val", "test"]], ignore_index=True)
n0 = len(df)
df = df[df["source"] != "PhishingEmailDetection"].reset_index(drop=True)
print(f"dropped PhishingEmailDetection rows: {n0 - len(df)}")

key = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())
df["k"] = df["text"].map(key)
df["grp"] = df["k"].str[:80]          # opening-text group (catches mass-mailed templates)

pool = df[df["source"].isin(TRAIN_SOURCES)].reset_index(drop=True)
ext = df[df["source"].isin(TEST_SOURCES)].reset_index(drop=True)

# ---- train / val / in-source test: 10 group-aware folds -> 1 test, 1 val, 8 train
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
fold = pd.Series(-1, index=pool.index)
for i, (_, idx) in enumerate(sgkf.split(pool, pool["label"], groups=pool["grp"])):
    fold.iloc[idx] = i
train = pool[fold >= 2]
val = pool[fold == 1]
test_in = pool[fold == 0]

# ---- cross-source tests: drop rows that also appear (exactly or by opening text) in train/val
seen_k, seen_g = set(pool["k"]), set(pool["grp"])
leak = ext["k"].isin(seen_k) | ext["grp"].isin(seen_g)
print("cross-source test rows removed for overlap with train sources:")
print(ext[leak].groupby("source").size().to_string() if leak.any() else "  none")
ext = ext[~leak]

cols = ["text", "label", "source"]
outs = {
    "train": train, "val": val, "test_insource": test_in,
    "test_ceas": ext[ext.source == "CEAS_08"],
    "test_nazario": ext[ext.source == "Nazario"],
    "test_nigerian": ext[ext.source == "Nigerian_Fraud"],
}
# Nazario has no legit emails: pair with an equal number of CEAS legit emails so precision / FPR exist.
ceas_legit = outs["test_ceas"][outs["test_ceas"].label == 0]
outs["test_nazario_paired"] = pd.concat(
    [outs["test_nazario"], ceas_legit.sample(len(outs["test_nazario"]), random_state=SEED)])

for name, part in outs.items():
    part[cols].sample(frac=1, random_state=SEED).to_csv(f"{OUT}/{name}.csv", index=False)

# ---- sanity checks
tr_k = set(train["k"]) | set(val["k"])
assert not (set(test_in["k"]) & tr_k), "in-source test overlaps train/val"
assert not (set(test_in["grp"]) & (set(train["grp"]) | set(val["grp"]))), "template leakage in-source"
for n in ["test_ceas", "test_nazario", "test_nigerian"]:
    assert not (set(outs[n]["k"]) & set(pool["k"])), f"{n} overlaps training sources"

summary = {n: {"rows": len(p), "label1": int(p.label.sum()), "label0": int((p.label == 0).sum()),
               "sources": p.source.value_counts().to_dict()} for n, p in outs.items()}
print(json.dumps(summary, indent=1))
json.dump(summary, open(f"{OUT}/_split_summary.json", "w"), indent=1)
