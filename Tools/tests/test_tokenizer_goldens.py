import json
import os
import subprocess
import sys

import pytest
from tokenizers import Tokenizer

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)

import fetch_tokenizers as ft  # noqa: E402

import _paths  # noqa: E402

MODELS = os.path.join(_paths.dev_dir(), "models", "tokenizers")
GOLDENS = os.path.join(_paths.dev_dir(), "goldens", "tokenizers")
STRINGS = os.path.join(TOOLS, "data", "tokenizer_strings.json")
NAMES = ["bert-base-uncased", "gpt2", "t5-small"]

need_models = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(MODELS, n, "SOURCE.json")) for n in NAMES),
    reason="tokenizer files not fetched; run Tools/fetch_tokenizers.py")
need_goldens = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(GOLDENS, n + ".json")) for n in NAMES),
    reason="goldens not generated; run Tools/make_tokenizer_goldens.py")


def load(name):
    with open(os.path.join(GOLDENS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def tok_path(g):
    return os.path.normpath(os.path.join(GOLDENS, g["tokenizer"]["file"]))


@need_models
def test_fetch_layout_and_sha():
    for n in NAMES:
        d = os.path.join(MODELS, n)
        with open(os.path.join(d, "SOURCE.json"), encoding="utf-8") as f:
            src = json.load(f)
        assert src["files"]
        for fn, sha in src["files"].items():
            assert ft.sha256_file(os.path.join(d, fn)) == sha


@need_models
def test_fetch_second_run_downloads_nothing(monkeypatch, tmp_path, capsys):
    # offline: API is faked, download must never be called
    lic = {m["repo"]: m["license"] for m in ft.MODELS}
    monkeypatch.setattr(ft, "fetch_api", lambda repo, rev: {"cardData": {"license": lic[repo]}})

    def boom(*a, **k):
        raise AssertionError("download attempted")
    monkeypatch.setattr(ft, "download_file", boom)
    assert ft.main(["--out", MODELS]) == 0
    assert "Nothing downloaded" in capsys.readouterr().out


def test_license_guard_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(ft, "fetch_api", lambda repo, rev: {"cardData": {"license": "llama2"}})

    def boom(*a, **k):
        raise AssertionError("download attempted")
    monkeypatch.setattr(ft, "download_file", boom)
    assert ft.main(["--out", str(tmp_path)]) != 0
    assert list(tmp_path.iterdir()) == []


def test_license_guard_single_model(monkeypatch, tmp_path):
    def api(repo, rev):
        return {"cardData": {"license": "llama2" if repo == "openai-community/gpt2" else
                             next(m["license"] for m in ft.MODELS if m["repo"] == repo)}}
    monkeypatch.setattr(ft, "fetch_api", api)
    monkeypatch.setattr(ft, "download_file", lambda r, v, f: b"x")
    assert ft.main(["--out", str(tmp_path)]) != 0
    assert not (tmp_path / "gpt2").exists()


@need_goldens
def test_golden_files_shape():
    for n in NAMES:
        g = load(n)
        assert g["schema"] == "pipelines-tokenizer-goldens/1"
        assert [c["id"] for c in g["cases"]] == ["s%02d" % i for i in range(24)]
        assert [b["id"] for b in g["batches"]] == ["b00", "b01", "b02", "b03"]
        assert ft.sha256_file(tok_path(g)) == g["tokenizer"]["file_sha256"]


@need_goldens
@pytest.mark.parametrize("name", NAMES)
def test_self_consistency(name):
    g = load(name)
    tok = Tokenizer.from_file(tok_path(g))
    for c in g["cases"]:
        assert bytes.fromhex(c["text_utf8_hex"]).decode("utf-8") == c["text"]
        assert tok.encode(c["text"], add_special_tokens=True).ids == c["ids_with_special"]
        assert tok.encode(c["text"], add_special_tokens=False).ids == c["ids_without_special"]
        ws = c["ids_with_special"]
        if name == "bert-base-uncased":
            assert ws[0] == 101 and ws[-1] == 102
        elif name == "t5-small":
            assert ws[-1] == 1
        else:
            assert ws == c["ids_without_special"]


@need_goldens
def test_bert_spot_values():
    g = load("bert-base-uncased")
    cases = {c["id"]: c for c in g["cases"]}
    assert cases["s00"]["ids_with_special"] == [101, 7592, 1010, 2088, 999, 102]
    assert cases["s22"]["ids_with_special"] == [101, 102]
    assert cases["s22"]["decode_with_special_skip"] == ""


@need_goldens
@pytest.mark.parametrize("name", NAMES)
def test_batches(name):
    g = load(name)
    for b in g["batches"]:
        ids, mask = b["input_ids"], b["attention_mask"]
        assert len({len(r) for r in ids}) == 1
        assert [len(r) for r in mask] == [len(r) for r in ids]
        pad = b["pad_id"]
        for r, m in zip(ids, mask):
            assert set(m) <= {0, 1}
            n = sum(m)
            assert m == [1] * n + [0] * (len(m) - n)
            assert all(x == pad for x in r[n:])
        if b["id"] == "b01":
            assert all(len(r) <= 8 for r in ids)
            if name == "bert-base-uncased":
                for r, m in zip(ids, mask):
                    assert r[sum(m) - 1] == 102
        if b["id"] == "b02":
            assert all(len(r) <= 5 for r in ids)


@need_models
def test_determinism_and_custom_strings(tmp_path):
    script = os.path.join(TOOLS, "make_tokenizer_goldens.py")
    outs = []
    for i in range(2):
        out = tmp_path / ("o%d" % i)
        subprocess.run([sys.executable, script, "--models", MODELS, "--strings", STRINGS,
                        "--out", str(out)], check=True, capture_output=True)
        outs.append({n: (out / (n + ".json")).read_bytes() for n in NAMES})
    assert outs[0] == outs[1]

    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps([{"id": "x00", "text": "hi there"}]), encoding="utf-8")
    out = tmp_path / "oc"
    subprocess.run([sys.executable, script, "--models", MODELS, "--strings", str(custom),
                    "--out", str(out)], check=True, capture_output=True)
    g = json.loads((out / "gpt2.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in g["cases"]] == ["x00"]
