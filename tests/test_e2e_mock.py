import pandas as pd

from helpers import make_classifier, toy_emails
from llm_classifier import align_predictions, fixed_sample, load_split, result_rows


def test_mock_run_end_to_end(tmp_path, repo_root, capsys):
    split = tmp_path / "test_toy.csv"
    toy_emails(80).to_csv(split, index=False)
    df = load_split(split)
    assert df["row_id"].tolist() == list(range(80))
    sample = fixed_sample(df, "toy", 40, tmp_path / "samples")

    clf = make_classifier(tmp_path, repo_root)
    out_dir = tmp_path / "mock_predictions"
    out_dir.mkdir()
    for v in clf.variants:
        out = clf.run_variant(v, sample, progress=False)
        out[["row_id", "y_true", "y_pred", "score"]].to_csv(out_dir / f"llm_{v}_toy.csv", index=False)
        assert set(out["y_pred"]) <= {-1, 0, 1}
        failed = out["y_pred"] == -1
        assert (out.loc[failed, "status"] != "ok").all() and (out.loc[~failed, "status"] == "ok").all()

    files = sorted(p.name for p in out_dir.iterdir())
    assert files == sorted(f"llm_{v}_toy.csv" for v in clf.variants)
    pred = pd.read_csv(out_dir / "llm_zs_cues_toy.csv")
    assert list(pred.columns) == ["row_id", "y_true", "y_pred", "score"]
    assert pred["row_id"].tolist() == sample["row_id"].tolist()

    # rerun: everything from the cache, identical predictions
    again = clf.run_variant("zs_cues", sample, progress=False)
    assert again["cache_hit"].all() and again["y_pred"].tolist() == pred["y_pred"].tolist()

    # another model's full-file predictions, restricted to the sampled rows
    other = df.rename(columns={"label": "y_true"}).assign(y_pred=lambda d: d["y_true"])
    aligned = align_predictions(sample, other[["row_id", "y_true", "y_pred"]], "tfidf", "toy")
    assert len(aligned) == len(sample)
    bad = other.assign(y_true=1 - other["y_true"])[["row_id", "y_true", "y_pred"]]
    assert align_predictions(sample, bad, "tfidf", "toy") is None

    rows = result_rows("llm_zs_cues", "toy", pred["y_true"], pred["y_pred"])
    assert rows[0]["n"] == len(sample) and rows[0]["n_failed"] == int((pred["y_pred"] == -1).sum())
