import json

import pytest

from helpers import make_classifier, toy_emails
from llm_classifier import ResponseCache, Settings, cache_key, check_frozen, freeze_prompts


def test_cache_key_changes_with_any_part_of_the_request():
    s = Settings()
    tool = {"name": "classify_email", "input_schema": {"type": "object"}}
    base = cache_key("anthropic", s, "system", "user", tool)
    assert base == cache_key("anthropic", s, "system", "user", dict(tool))
    assert base != cache_key("anthropic", s, "system!", "user", tool)                    # prompt text
    assert base != cache_key("anthropic", s, "system", "user2", tool)                    # email text
    assert base != cache_key("anthropic", s, "system", "user", {**tool, "name": "x"})    # tool schema
    assert base != cache_key("anthropic", Settings(model="other"), "system", "user", tool)
    assert base != cache_key("mock", s, "system", "user", tool)


def test_editing_a_prompt_file_changes_its_requests(tmp_path, repo_root):
    clf = make_classifier(tmp_path, repo_root)
    system, _, tool = clf.build_variant("zs_cues")
    before = cache_key("mock", clf.settings, system, "u", tool)
    f = clf.prompts_dir / "zs_cues.txt"
    f.write_text(f.read_text().replace("Judge the email as a whole.", "Judge the whole email."))
    system2, _, _ = clf.build_variant("zs_cues")
    assert cache_key("mock", clf.settings, system2, "u", tool) != before


def test_freeze_guard_catches_an_edited_prompt(tmp_path, repo_root, capsys):
    clf = make_classifier(tmp_path, repo_root)
    freeze = tmp_path / "freeze.json"
    with pytest.raises(RuntimeError, match="not frozen"):
        check_frozen(clf, freeze)
    freeze_prompts(clf, freeze)
    check_frozen(clf, freeze)                                   # unchanged: passes
    f = clf.prompts_dir / "zs_definition.txt"
    f.write_text(f.read_text() + "\nExtra sentence.\n")
    with pytest.raises(RuntimeError, match="zs_definition"):
        check_frozen(clf, freeze)


def test_freeze_guard_catches_changed_fewshot_examples(tmp_path, repo_root):
    clf = make_classifier(tmp_path, repo_root)
    freeze = tmp_path / "freeze.json"
    freeze_prompts(clf, freeze)
    clf.fewshot = toy_emails(6, seed=1).assign(row_id=range(6))
    with pytest.raises(RuntimeError, match="fs_balanced"):
        check_frozen(clf, freeze)


def test_cache_persists_and_skips_corrupt_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    c = ResponseCache(path)
    c.put("k1", {"status": "ok"})
    with path.open("a") as f:
        f.write('{"key": "k2", "result": {"sta')            # cut off mid-write
    c2 = ResponseCache(path)
    assert c2.get("k1") == {"status": "ok"} and len(c2) == 1 and c2.skipped_lines == 1


def test_identical_requests_are_sent_once(tmp_path, repo_root):
    clf = make_classifier(tmp_path, repo_root)
    calls = []
    real = clf.backend.classify
    clf.backend.classify = lambda *a: calls.append(1) or real(*a)
    df = toy_emails(10).assign(row_id=range(10))
    df = __import__("pandas").concat([df, df.assign(row_id=df["row_id"] + 100)])   # every email twice
    out = clf.run_variant("zs_plain", df, progress=False)
    assert len(calls) == 10 and len(out) == 20
    clf.run_variant("zs_plain", df, progress=False)
    assert len(calls) == 10                                     # second run: all from cache


def test_api_errors_are_not_cached_and_fatal_errors_stop_the_run(tmp_path, repo_root):
    clf = make_classifier(tmp_path, repo_root)
    df = toy_emails(4).assign(row_id=range(4))

    def flaky(*a):
        raise TimeoutError("network")
    clf.backend.classify = flaky
    out = clf.run_variant("zs_plain", df, progress=False)
    assert (out["status"] == "api_error").all() and (out["y_pred"] == -1).all() and len(clf.cache) == 0

    class Fatal(Exception):
        pass

    def fatal(*a):
        raise Fatal("bad request")
    clf.backend.classify = fatal
    clf.backend.fatal_errors = (Fatal,)
    with pytest.raises(Fatal):
        clf.run_variant("zs_plain", df, progress=False)
