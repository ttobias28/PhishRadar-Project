import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from llm_classifier import bootstrap_ci, bootstrap_indices, compute_metrics, paired_bootstrap_diff

RNG = np.random.default_rng(3)
Y = RNG.integers(0, 2, 200)
P = np.where(RNG.random(200) < 0.8, Y, 1 - Y)


def sk(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0), "fpr": fp / (fp + tn)}


def test_metrics_match_sklearn():
    m = compute_metrics(Y, P)
    for k, v in sk(Y, P).items():
        assert m[k] == pytest.approx(v)
    assert m["n"] == 200 and m["n_failed"] == 0


def test_failed_calls_count_as_wrong():
    p = P.copy()
    p[:10] = -1
    m = compute_metrics(Y, p)
    expected = sk(Y, np.where(p == -1, 1 - Y, p))
    assert m["n_failed"] == 10 and m["f1"] == pytest.approx(expected["f1"])
    assert m["accuracy"] == pytest.approx(((np.where(p == -1, 1 - Y, p)) == Y).mean())


def test_single_class_set_reports_recall_only():
    y = np.ones(50, dtype=int)
    m = compute_metrics(y, np.r_[np.ones(40), np.zeros(10)])
    assert m["recall"] == pytest.approx(0.8)
    assert np.isnan(m["precision"]) and np.isnan(m["f1"]) and np.isnan(m["fpr"])


def test_bootstrap_ci_matches_sklearn_on_the_same_resamples():
    ci = bootstrap_ci(Y, P, n_boot=200, seed=42)
    idx = bootstrap_indices(len(Y), 200, 42)
    per = [sk(Y[i], P[i]) for i in idx]
    for k in ("precision", "recall", "f1", "fpr"):
        vals = np.array([d[k] for d in per])
        assert ci[k] == pytest.approx((np.percentile(vals, 2.5), np.percentile(vals, 97.5)))


def test_paired_diff_matches_sklearn():
    other = np.where(RNG.random(200) < 0.7, Y, 1 - Y)
    point, lo, hi = paired_bootstrap_diff(Y, P, other, "f1", n_boot=200, seed=42)
    assert point == pytest.approx(f1_score(Y, P) - f1_score(Y, other))
    idx = bootstrap_indices(len(Y), 200, 42)
    diffs = np.array([f1_score(Y[i], P[i]) - f1_score(Y[i], other[i]) for i in idx])
    assert (lo, hi) == pytest.approx((np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)))
