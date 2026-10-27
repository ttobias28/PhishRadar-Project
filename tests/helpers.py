import shutil

import numpy as np
import pandas as pd

from llm_classifier import LLMClassifier, MockBackend, ResponseCache, Settings

PHISHY = ["Verify your account password now: click here", "You are a winner! Claim your free money",
          "Dear friend, urgent bank transfer offer", "Limited time credit offer, click to claim"]
LEGIT = ["Meeting moved to 3pm, see attached agenda", "Thanks for dinner last weekend",
         "Minutes from the project review are attached", "Can you send me the contract draft?"]


def toy_emails(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (np.arange(n) % 2).astype(int)
    texts = [f"{rng.choice(PHISHY if y else LEGIT)} (message {i})" for i, y in enumerate(labels)]
    return pd.DataFrame({"text": texts, "label": labels, "source": "toy"})


def make_classifier(tmp_path, repo_root, **kw) -> LLMClassifier:
    prompts = tmp_path / "prompts"
    if not prompts.exists():
        shutil.copytree(repo_root / "prompts", prompts, ignore=shutil.ignore_patterns("*.json"))
    fewshot = toy_emails(6).assign(row_id=range(6))
    return LLMClassifier(Settings(), prompts, MockBackend(), ResponseCache(tmp_path / "cache.jsonl"),
                         fewshot=fewshot, **kw)
