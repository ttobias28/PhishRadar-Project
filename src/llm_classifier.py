"""PhishRadar Approach C: classify emails by prompting an LLM.

Shared by PhishRadar_LLM_prompting.ipynb, the adversarial step and any evaluation script.
Nothing here is trained: only the prompt changes. Main pieces:

  * prompts: load prompt files, render one request per email, few-shot examples
  * backends: AnthropicBackend (real API calls) and MockBackend (free, fake answers)
  * ResponseCache: stores every completed answer, keyed on a hash of the full request
  * LLMClassifier: runs a prompt variant over a DataFrame of emails
  * samples, prompt freeze, metrics with bootstrap CIs, cost estimates
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(iterable, **kwargs):
        return iterable

SEED = 42
LABEL_TO_INT = {"phishing": 1, "legitimate": 0}
TOOL_NAME = "classify_email"

# Claude Haiku 5.5 prices in USD per million tokens, prompts up to 100K tokens.
# Checked 2026-10-27 at https://platform.claude.com/docs/en/about-claude/pricing
PRICES = {"input": 0.10, "output": 0.50, "cache_write_5m": 0.125, "cache_read": 0.01}
# Tokens the API adds for the tool-use system prompt when tool_choice forces a tool (Haiku 5.5, same page).
FORCED_TOOL_OVERHEAD = 406


@dataclass(frozen=True)
class Settings:
    """Everything that changes the request. State these values in the report."""
    model: str = "claude-haiku-5-5"
    max_tokens: int = 512           # room for the tool call plus a short reason
    max_chars: int = 2000           # every email is cut to this many characters, for every prompt
    fewshot_max_chars: int = 600    # few-shot examples are cut shorter to keep the prompt small
    normalize_spacing: bool = True  # undo "doesn ' t" spacing from Enron/Ling, as the DistilBERT notebook does


# Each variant changes one thing relative to the one it builds on (research question 4).
VARIANTS = {
    "zs_plain":      {"prompt": "zs_plain.txt",      "reason": False, "fewshot": False},
    "zs_definition": {"prompt": "zs_definition.txt", "reason": False, "fewshot": False},
    "zs_cues":       {"prompt": "zs_cues.txt",       "reason": False, "fewshot": False},
    "zs_reasoned":   {"prompt": "zs_reasoned.txt",   "reason": True,  "fewshot": False},
    "fs_balanced":   {"prompt": "fs_balanced.txt",   "reason": False, "fewshot": True, "order_seed": 0},
}


# ---------------------------------------------------------------- data

def load_split(path) -> pd.DataFrame:
    """Read a split CSV. row_id is the row position in the file, the same as in the TF-IDF and
    DistilBERT prediction files (verified by joining on row_id and comparing labels)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the split CSVs in data/splits/ (see data/data_README.md).")
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df)))
    if df["row_id"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate row_ids")
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    if n >= len(df):
        return df.copy()
    parts = []
    for _, group in df.groupby("label"):
        k = min(len(group), max(1, round(len(group) * n / len(df))))
        parts.append(group.sample(n=k, random_state=seed))
    return pd.concat(parts)


def fixed_sample(df: pd.DataFrame, name: str, n: int, samples_dir, seed: int = SEED) -> pd.DataFrame:
    """Draw a label-stratified sample once and save its row_ids. Later calls load the saved ids."""
    path = Path(samples_dir) / f"{name}_rowids.csv"
    if path.exists():
        ids = pd.read_csv(path)["row_id"]
        sample = df[df["row_id"].isin(ids)]
        if len(sample) != len(ids):
            raise ValueError(f"{path.name}: {len(ids) - len(sample)} saved row_ids are missing from the split. "
                             "Did the splits change?")
        status = "loaded"
    else:
        sample = stratified_sample(df, n, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        sample[["row_id"]].sort_values("row_id").to_csv(path, index=False)
        status = "created"
    sample = sample.sort_values("row_id").reset_index(drop=True)
    print(f"{status:8s} {path.name:32s} {len(sample):5d} rows, phishing share {sample['label'].mean():.3f}")
    return sample


# ---------------------------------------------------------------- text

def normalize_spacing(t):
    # Copied verbatim from PhishRadar_DistilBERT.ipynb so both models see the same text.
    t = re.sub(r"(\w) ' (s|t|re|ve|ll|d|m)\b", r"\1'\2", t)   # doesn ' t -> doesn't
    t = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", t)                # word , -> word,
    t = re.sub(r"([(\[{$])\s+", r"\1", t)                      # ( word -> (word
    t = re.sub(r"(\d) ([.,:/]) (\d)", r"\1\2\3", t)             # 6 / 10 -> 6/10
    t = re.sub(r"(\d)([.,:/]) (\d)", r"\1\2\3", t)
    return t


_TAG_RE = re.compile(r"<\s*(/?)\s*(email|example|label)\b", re.IGNORECASE)


def sanitize(text: str) -> str:
    """'<email>' -> '<_email>', '</email>' -> '</_email>': an email can't open or close our tags."""
    return _TAG_RE.sub(lambda m: f"<{m.group(1)}_{m.group(2)}", text)


def prepare_email(text: str, max_chars: int, normalize: bool = True) -> str:
    if normalize:
        text = normalize_spacing(text)
    return sanitize(text[:max_chars].strip())


# ---------------------------------------------------------------- prompts

def load_prompt(path) -> tuple[str, str]:
    """A prompt file has a '=== SYSTEM ===' and a '=== USER ===' section. Lines above SYSTEM are comments."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if "=== SYSTEM ===" not in raw or "=== USER ===" not in raw:
        raise ValueError(f"{path.name} needs '=== SYSTEM ===' and '=== USER ===' sections")
    system, user = raw.split("=== SYSTEM ===", 1)[1].split("=== USER ===", 1)
    system, user = system.strip(), user.strip()
    if "{email}" not in user:
        raise ValueError(f"{path.name}: the USER section needs an {{email}} placeholder")
    return system, user


def make_tool(with_reason: bool) -> dict:
    props = {}
    if with_reason:   # listed first so the model writes the reason before it picks a label
        props["reason"] = {"type": "string",
                           "description": "One short sentence explaining the decision. Write this before the label."}
    props["label"] = {"type": "string", "enum": ["phishing", "legitimate"]}
    props["phishing_probability"] = {"type": "integer",
                                     "description": "How likely the email is phishing or a scam, from 0 to 100."}
    return {
        "name": TOOL_NAME,
        "description": "Record the classification of the email.",
        "input_schema": {"type": "object", "properties": props, "required": list(props)},
    }


# Candidate pools for hand-picking few-shot examples (one category of the project's definition each).
FEWSHOT_CATEGORIES = {
    "credential": (1, r"verify your|your account (?:has been|will be)|update your (?:account|billing|information)|"
                      r"confirm your (?:account|identity|details)|password|log ?in to your|suspend"),
    "scam":       (1, r"inheritance|next of kin|beneficiary|million (?:us )?dollars|lottery|you have won|winner|"
                      r"claim your prize|fund transfer|business proposal|work from home|earn \$"),
    "bulk_ad":    (1, r"unsubscribe|remove(?:d)? from (?:our|this) list|special offer|order now|low price|"
                      r"save \d+%|free shipping"),
    "work":       (0, r"meeting|attached|schedule|conference call|deal|contract|project|let me know"),
    "personal":   (0, r"\b(?:mom|dad|dinner|weekend|birthday|wedding|party|kids|vacation|love,)\b"),
    "list":       (0, r"mailing list|newsletter|digest|unsubscribe|list-?id|sourceforge|lists\.|subscribe"),
}


def _readable(t: str) -> bool:
    letters = sum(c.isalpha() for c in t)
    vowels = sum(c in "aeiouAEIOU" for c in t)
    return letters / max(len(t), 1) > 0.7 and vowels / max(letters, 1) > 0.3 and not re.search(r"(.)\1{5,}", t)


def fewshot_candidates(train_df: pd.DataFrame, n_per_source: int = 6, seed: int = SEED,
                       sources=("Enron", "SpamAssassin")) -> pd.DataFrame:
    """Readable train.csv emails (250-1200 chars, mostly letters, has vowels) per category and source."""
    pool = train_df[train_df["text"].str.len().between(250, 1200)]
    pool = pool[pool["text"].map(_readable)]
    out = []
    for cat, (label, pattern) in FEWSHOT_CATEGORIES.items():
        for src in sources:
            g = pool[(pool["label"] == label) & (pool["source"] == src)
                     & pool["text"].str.contains(pattern, case=False, regex=True)]
            out.append(g.sample(n=min(n_per_source, len(g)), random_state=seed).assign(category=cat))
    return pd.concat(out).reset_index(drop=True)


def load_fewshot(fewshot_file, train_df: pd.DataFrame) -> pd.DataFrame:
    """Few-shot examples chosen by hand, stored as row_ids of train.csv in fewshot_examples.json."""
    path = Path(fewshot_file)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Pick examples from fewshot_candidates(train_df) and "
                                "record their row_ids there.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("source") != "train.csv":
        raise ValueError(f"{path.name}: few-shot examples must come from train.csv")
    ids = [e["row_id"] for e in meta["examples"]]
    ex = train_df.set_index("row_id").loc[ids].reset_index()
    wrong = [e["row_id"] for e, lab in zip(meta["examples"], ex["label"]) if e["label"] != lab]
    if wrong:
        raise ValueError(f"{path.name}: labels don't match train.csv for row_ids {wrong}")
    return ex


def format_examples(examples: pd.DataFrame, settings: Settings, order_seed=None) -> str:
    if order_seed is not None:
        examples = examples.sample(frac=1, random_state=order_seed)
    blocks = []
    for text, label in zip(examples["text"], examples["label"]):
        email = prepare_email(text, settings.fewshot_max_chars, settings.normalize_spacing)
        blocks.append(f"<example>\n<email>\n{email}\n</email>\n"
                      f"<label>{'phishing' if label == 1 else 'legitimate'}</label>\n</example>")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- responses

def estimate_tokens(text: str) -> int:
    # Rough: about 4 characters per token, plus 30% for the newer tokenizer.
    return int(len(text) / 4 * 1.3) + 1


def validate_tool_input(inp, stop_reason, usage: dict) -> dict:
    inp = inp if isinstance(inp, dict) else {}
    label, prob, reason = inp.get("label"), inp.get("phishing_probability"), inp.get("reason")
    if isinstance(prob, str):
        try:
            prob = float(prob)
        except ValueError:
            prob = None
    ok_prob = isinstance(prob, (int, float)) and not isinstance(prob, bool) and 0 <= prob <= 100
    if label in LABEL_TO_INT and ok_prob:
        return {"status": "ok", "label": label, "prob": int(round(prob)), "reason": reason,
                "stop_reason": stop_reason, **usage}
    return {"status": "unparseable", "raw": json.dumps(inp)[:300], "stop_reason": stop_reason, **usage}


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    return {k: int(getattr(u, k, 0) or 0) for k in
            ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}


def parse_response(resp, tool_name: str = TOOL_NAME) -> dict:
    usage = _usage(resp)
    if resp.stop_reason == "refusal":
        details = getattr(resp, "stop_details", None)
        return {"status": "refusal", "stop_reason": "refusal",
                "refusal_category": getattr(details, "category", None), **usage}
    for block in resp.content:   # select blocks by type, not position (thinking blocks may come first)
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            return validate_tool_input(block.input, resp.stop_reason, usage)
    return {"status": "unparseable", "stop_reason": resp.stop_reason, **usage}


# ---------------------------------------------------------------- backends

class AnthropicBackend:
    """Real API calls with a forced tool call. Haiku 5.5 takes no temperature/top_p/top_k and no prefill."""
    name = "anthropic"

    def __init__(self, settings: Settings, max_retries: int = 5, client=None):
        import anthropic
        if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Put it in .env (never commit .env).")
        # The SDK retries 408/409/429/5xx and connection errors with backoff and honors retry-after.
        self.client = client or anthropic.Anthropic(max_retries=max_retries)
        self.settings = settings
        # Errors that would fail every row the same way: stop the run instead of recording them per row.
        self.fatal_errors = (anthropic.BadRequestError, anthropic.AuthenticationError,
                             anthropic.PermissionDeniedError, anthropic.NotFoundError)

    def _request(self, system: str, user: str, tool: dict) -> dict:
        # cache_control on the system block caches tools + system across emails (prompt caching).
        # It doesn't change the answer, only the price; prefixes under 512 tokens silently don't cache.
        return dict(model=self.settings.model,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    tools=[tool], tool_choice={"type": "tool", "name": tool["name"]},
                    messages=[{"role": "user", "content": user}])

    def classify(self, system: str, user: str, tool: dict) -> dict:
        resp = self.client.messages.create(max_tokens=self.settings.max_tokens,
                                           **self._request(system, user, tool))
        return parse_response(resp, tool["name"])

    def count_tokens(self, system: str, user: str, tool: dict) -> int:
        # Free endpoint: exact input token count without running the model.
        return self.client.messages.count_tokens(**self._request(system, user, tool)).input_tokens


class MockBackend:
    """Fake model for building and testing the pipeline at $0. Its predictions mean nothing.
    About 1% of answers are fake refusals and 2% are malformed, so failure handling gets exercised."""
    name = "mock"
    fatal_errors: tuple = ()
    WORDS = ["verify", "password", "account", "bank", "click", "urgent", "winner", "free",
             "offer", "money", "$", "credit", "limited time", "dear friend"]

    def count_tokens(self, system: str, user: str, tool: dict) -> int:
        return estimate_tokens(system + user + json.dumps(tool)) + FORCED_TOOL_OVERHEAD

    def classify(self, system: str, user: str, tool: dict) -> dict:
        roll = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) % 1000
        usage = {"input_tokens": self.count_tokens(system, user, tool), "output_tokens": 40,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if roll < 10:
            return {"status": "refusal", "stop_reason": "refusal", "refusal_category": None, **usage}
        if roll < 30:
            return validate_tool_input({"label": "maybe"}, "tool_use", usage)
        m = re.search(r"<email>\n(.*)\n</email>", user, re.S)
        email = (m.group(1) if m else user).lower()
        hits = sum(w in email for w in self.WORDS)
        prob = min(95, 5 + 20 * hits)
        inp = {"label": "phishing" if prob >= 50 else "legitimate", "phishing_probability": prob}
        if "reason" in tool["input_schema"]["properties"]:
            inp = {"reason": f"Mock: {hits} keyword hits.", **inp}
        return validate_tool_input(inp, "tool_use", usage)


# ---------------------------------------------------------------- cache

class ResponseCache:
    """Append-only JSONL of completed answers (refusals included), safe to use from several threads.
    API errors are never stored, so a rerun retries them. The file may contain email text: keep it git-ignored."""

    def __init__(self, path):
        self.path = Path(path)
        self._data: dict = {}
        self._lock = threading.Lock()
        self.skipped_lines = 0
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        self._data[rec["key"]] = rec["result"]
                    except (json.JSONDecodeError, KeyError):   # e.g. a line cut off by a crash
                        self.skipped_lines += 1

    def __len__(self):
        return len(self._data)

    def get(self, key):
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, result: dict) -> None:
        with self._lock:
            self._data[key] = result
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")


def cache_key(backend_name: str, settings: Settings, system: str, user: str, tool: dict) -> str:
    """Hash of the full request: model, settings, exact prompt text, email text and tool schema."""
    payload = json.dumps({"backend": backend_name, "model": settings.model, "max_tokens": settings.max_tokens,
                          "system": system, "user": user, "tool": tool}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- runner

class LLMClassifier:
    def __init__(self, settings: Settings, prompts_dir, backend, cache: ResponseCache,
                 variants: dict = VARIANTS, fewshot: pd.DataFrame | None = None, max_workers: int = 4):
        self.settings = settings
        self.prompts_dir = Path(prompts_dir)
        self.backend = backend
        self.cache = cache
        self.variants = variants
        self.fewshot = fewshot
        self.max_workers = max_workers

    def build_variant(self, name: str) -> tuple[str, str, dict]:
        v = self.variants[name]
        system, user_template = load_prompt(self.prompts_dir / v["prompt"])
        if v.get("fewshot"):
            if "{examples}" not in system:
                raise ValueError(f"{v['prompt']}: a few-shot variant needs an {{examples}} placeholder in SYSTEM")
            if self.fewshot is None:
                raise ValueError(f"{name} needs few-shot examples, but none were loaded")
            system = system.replace("{examples}", format_examples(self.fewshot, self.settings, v.get("order_seed")))
        return system, user_template, make_tool(v.get("reason", False))

    def render_user(self, user_template: str, email_text: str) -> str:
        email = prepare_email(email_text, self.settings.max_chars, self.settings.normalize_spacing)
        return user_template.replace("{email}", email)

    def fingerprint(self, name: str) -> str:
        system, user_template, tool = self.build_variant(name)
        s = self.settings
        payload = json.dumps({"model": s.model, "max_tokens": s.max_tokens, "max_chars": s.max_chars,
                              "normalize_spacing": s.normalize_spacing, "system": system,
                              "user": user_template, "tool": tool}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _call(self, key: str, system: str, user: str, tool: dict) -> dict:
        try:
            result = self.backend.classify(system, user, tool)
        except self.backend.fatal_errors:
            raise
        except Exception as e:   # rate limits after retries, server and network errors: not cached
            return {"status": "api_error", "error": repr(e)[:300]}
        result["cached_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.cache.put(key, result)
        return result

    def run_variant(self, name: str, df: pd.DataFrame, desc: str = "", progress: bool = True) -> pd.DataFrame:
        system, user_template, tool = self.build_variant(name)
        users = [self.render_user(user_template, t) for t in df["text"]]
        keys = [cache_key(self.backend.name, self.settings, system, u, tool) for u in users]

        results, hits = {}, set()
        todo = {}
        for k, u in zip(keys, users):   # identical requests are sent once
            cached = self.cache.get(k)
            if cached is not None:
                results[k] = cached
                hits.add(k)
            else:
                todo.setdefault(k, u)

        if todo:
            pool = ThreadPoolExecutor(max_workers=self.max_workers)
            try:
                futures = {pool.submit(self._call, k, system, u, tool): k for k, u in todo.items()}
                done = as_completed(futures)
                if progress:
                    done = tqdm(done, total=len(futures), desc=desc or name)
                for fut in done:
                    results[futures[fut]] = fut.result()
            finally:   # on a fatal error, don't start the rest
                pool.shutdown(wait=True, cancel_futures=True)

        rows = []
        for row_id, y_true, k in zip(df["row_id"], df["label"], keys):
            r = results[k]
            rows.append({"row_id": row_id, "y_true": y_true, "status": r["status"],
                         "label": r.get("label"), "prob": r.get("prob"), "reason": r.get("reason"),
                         "stop_reason": r.get("stop_reason"), "refusal_category": r.get("refusal_category"),
                         "input_tokens": r.get("input_tokens", 0), "output_tokens": r.get("output_tokens", 0),
                         "cache_creation_input_tokens": r.get("cache_creation_input_tokens", 0),
                         "cache_read_input_tokens": r.get("cache_read_input_tokens", 0),
                         "cache_hit": k in hits})
        out = pd.DataFrame(rows)
        out["y_pred"] = out["label"].map(LABEL_TO_INT).fillna(-1).astype(int)   # -1 = failed call
        out["score"] = pd.to_numeric(out["prob"], errors="coerce") / 100.0

        new = out[~out["cache_hit"]]
        counts = out["status"].value_counts().to_dict()
        print(f"{desc or name}: {len(out)} rows | {int(out['cache_hit'].sum())} from cache | {counts} | "
              f"new tokens in/out {int(new['input_tokens'].sum()):,}/{int(new['output_tokens'].sum()):,} | "
              f"new cost ${cost_usd(new):.4f}")
        if counts.get("api_error"):
            print("  -> api_error rows were not cached. Rerun to retry them.")
        return out

    def classify_texts(self, texts, variant: str = "zs_cues") -> pd.DataFrame:
        """Classify any list of email texts (e.g. the adversarial set) with the same rendering and cache."""
        df = pd.DataFrame({"row_id": range(len(texts)), "text": list(texts), "label": -1})
        out = self.run_variant(variant, df, desc=f"custom {variant}")
        return out[["row_id", "status", "y_pred", "score", "reason"]]


# ---------------------------------------------------------------- freeze

def git_commit(repo_root) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def freeze_prompts(clf: LLMClassifier, freeze_file, commit: str = "unknown") -> dict:
    record = {"frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "git_commit": commit, "model": clf.settings.model,
              "variants": {v: clf.fingerprint(v) for v in clf.variants}}
    Path(freeze_file).write_text(json.dumps(record, indent=2) + "\n")
    print(f"Froze {len(clf.variants)} variants -> {freeze_file}")
    return record


def check_frozen(clf: LLMClassifier, freeze_file) -> None:
    path = Path(freeze_file)
    if not path.exists():
        raise RuntimeError("Prompts are not frozen yet. Finish Step 2, then run the freeze cell.")
    frozen = json.loads(path.read_text())["variants"]
    changed = [v for v in clf.variants if frozen.get(v) != clf.fingerprint(v)]
    if changed:
        raise RuntimeError(f"These variants changed after the freeze (or were never frozen): {changed}. "
                           "Test results must come from frozen prompts.")
    print(f"All {len(clf.variants)} variants match the freeze file.")


# ---------------------------------------------------------------- metrics

def effective_pred(y_true, y_pred) -> np.ndarray:
    """Failed calls (y_pred == -1) count as wrong."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return np.where(y_pred == -1, 1 - y_true, y_pred)


def _metric_arrays(t: np.ndarray, p: np.ndarray) -> dict:
    """Metrics along the last axis. Like the TF-IDF and DistilBERT notebooks, precision and F1 are
    NaN when a set has no legitimate emails (Nazario, Nigerian)."""
    tp = ((p == 1) & (t == 1)).sum(-1); fp = ((p == 1) & (t == 0)).sum(-1)
    fn = ((p == 0) & (t == 1)).sum(-1); tn = ((p == 0) & (t == 0)).sum(-1)
    has_legit = (fp + tn) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(has_legit, tp / (tp + fp), np.nan)
        f1 = np.where(has_legit, 2 * tp / (2 * tp + fp + fn), np.nan)
        return {"precision": precision, "recall": tp / (tp + fn), "f1": f1, "fpr": fp / (fp + tn),
                "accuracy": (tp + tn) / (tp + fp + fn + tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics(y_true, y_pred) -> dict:
    t = np.asarray(y_true).astype(int)
    m = {k: v.item() for k, v in _metric_arrays(t, effective_pred(t, y_pred)).items()}
    return {"n": len(t), "n_failed": int((np.asarray(y_pred) == -1).sum()), **m}


def bootstrap_indices(n: int, n_boot: int = 1000, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _percentile_ci(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, seed: int = SEED) -> dict:
    """95% percentile bootstrap CIs for precision, recall, F1 and FPR."""
    t = np.asarray(y_true).astype(int)
    idx = bootstrap_indices(len(t), n_boot, seed)
    m = _metric_arrays(t[idx], effective_pred(t, y_pred)[idx])
    return {k: _percentile_ci(m[k]) for k in ("precision", "recall", "f1", "fpr")}


def paired_bootstrap_diff(y_true, pred_a, pred_b, metric: str = "f1", n_boot: int = 1000, seed: int = SEED):
    """metric(a) - metric(b) on the same resampled rows: point estimate and 95% CI."""
    t = np.asarray(y_true).astype(int)
    a, b = effective_pred(t, pred_a), effective_pred(t, pred_b)
    idx = bootstrap_indices(len(t), n_boot, seed)
    diffs = _metric_arrays(t[idx], a[idx])[metric] - _metric_arrays(t[idx], b[idx])[metric]
    point = _metric_arrays(t, a)[metric] - _metric_arrays(t, b)[metric]
    lo, hi = _percentile_ci(diffs)
    return float(point), lo, hi


def summarize(model: str, test_set: str, y_true, y_pred, n_boot: int = 1000, seed: int = SEED) -> dict:
    m, ci = compute_metrics(y_true, y_pred), bootstrap_ci(y_true, y_pred, n_boot, seed)
    row = {"model": model, "test_set": test_set, "n": m["n"], "n_failed": m["n_failed"],
           "accuracy": m["accuracy"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]}
    for k in ("precision", "recall", "f1", "fpr"):
        row[k] = m[k]
        row[f"{k}_lo"], row[f"{k}_hi"] = ci[k]
    return row


def result_rows(model: str, test_set: str, y_true, y_pred) -> list[dict]:
    """Summary rows; for nazario_paired also recall on its Nazario half (research question 3)."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    rows = [summarize(model, test_set, y_true, y_pred)]
    if test_set == "nazario_paired":
        mask = y_true == 1
        rows.append(summarize(model, "nazario (phishing half)", y_true[mask], y_pred[mask]))
    return rows


# ---------------------------------------------------------------- other models' predictions

def align_predictions(sample: pd.DataFrame, other: pd.DataFrame | None, model: str, test_set: str):
    """Restrict another model's predictions (row_id, y_true, y_pred) to the sampled rows.
    Returns None, with a message, if rows are missing or labels disagree on the same row_id."""
    if other is None:
        return None
    merged = sample[["row_id", "label"]].merge(other, on="row_id", how="left")
    if merged["y_pred"].isna().any():
        print(f"{model}/{test_set}: {int(merged['y_pred'].isna().sum())} sampled rows missing. Skipped.")
        return None
    mismatch = int((merged["label"] != merged["y_true"]).sum())
    if mismatch:
        print(f"{model}/{test_set}: labels disagree on {mismatch} row_ids, so row_id means something "
              "different in that file. Skipped.")
        return None
    print(f"{model}/{test_set}: {len(merged)} sampled rows, labels match on every row")
    return merged[["row_id", "y_true", "y_pred"]].astype({"y_pred": int})


# ---------------------------------------------------------------- cost

def cost_usd(df: pd.DataFrame, prices: dict = PRICES) -> float:
    """Cost of the calls in df (uncached input, cache writes, cache reads, output)."""
    get = lambda c: float(df[c].sum()) if c in df else 0.0
    return (get("input_tokens") * prices["input"] + get("output_tokens") * prices["output"]
            + get("cache_creation_input_tokens") * prices["cache_write_5m"]
            + get("cache_read_input_tokens") * prices["cache_read"]) / 1e6


def project_cost(avg_in: float, avg_out: float, calls: int, prices: dict = PRICES) -> float:
    """Upper-bound cost: every input token at the full price (ignores prompt-caching savings)."""
    return (avg_in * prices["input"] + avg_out * prices["output"]) * calls / 1e6
