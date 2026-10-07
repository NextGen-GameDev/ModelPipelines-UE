"""WP5.3 (C36): user tools. Network-free: downloads are monkeypatched."""
import hashlib
import importlib.metadata as md
import ast
import io
import json
import os
import re
import sys

import pytest
from PIL import Image

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import fetch_reference_models as frm  # noqa: E402
import fetch_sample_images as fsi  # noqa: E402

REQ = os.path.join(TOOLS, "requirements.txt")
REQ_DEV = os.path.join(TOOLS, "requirements-dev.txt")


def parse_requirements(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([0-9][^\s;]*)", line)
            assert m, "not a pinned requirement: %r" % line
            out[m.group(1).lower()] = m.group(2)
    return out


# ---- requirements ------------------------------------------------------------------------------------------------

def test_requirements_pins_equal_env_versions():
    reqs = parse_requirements(REQ)
    for name, want in reqs.items():
        have = md.version(name).split("+")[0]  # torch 2.14.1+cpu -> 2.14.1
        assert have == want, "%s pinned %s but the env has %s" % (name, want, have)


def test_requirements_cover_the_contract_and_exclude_dev_packages():
    reqs = parse_requirements(REQ)
    for name in ("torch", "transformers", "tokenizers", "onnx", "onnxruntime", "numpy", "optimum", "pillow",
                 "sentencepiece"):
        assert name in reqs
    assert reqs["torch"] == "2.14.1" and reqs["transformers"] == "4.57.6" and reqs["tokenizers"] == "0.22.2"
    assert reqs["onnx"] == "1.23.1" and reqs["onnxruntime"] == "1.30.0" and reqs["numpy"] == "2.4.6"
    assert "pytest" not in reqs
    assert "pytest" in parse_requirements(REQ_DEV)
    text = open(REQ, "r", encoding="utf-8").read()
    assert "https://download.pytorch.org/whl/cpu" in text  # comment line with the CPU torch index


# ---- licence gates -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [("CC0", "CC0-1.0"), ("cc0", "CC0-1.0"), ("CC0-1.0", "CC0-1.0"),
                                      ("Public domain", "PD"), ("PD-self", "PD"), ("PD-USGov", "PD"), ("PD", "PD"),
                                      ("CC BY-SA 4.0", None), ("CC BY 2.0", None), ("CC-BY-SA-3.0", None),
                                      ("fair use", None), ("", None), (None, None), ("GFDL", None),
                                      ("Attribution", None)])
def test_sample_license_gate(raw, want):
    assert fsi.classify_license(raw) == want


@pytest.mark.parametrize("lic,want", [("apache-2.0", "Apache-2.0"), ("mit", "MIT"), ("bsd-3-clause", "BSD-3-Clause"),
                                      ("bsd-2-clause", "BSD-2-Clause"), ("cc-by-4.0", None), ("gpl-3.0", None),
                                      ("openrail", None), ("other", None), ("", None), (None, None), ("bsd", None)])
def test_model_license_gate(lic, want):
    assert frm.license_to_spdx(lic) == want


# ---- sample_images.json schema -----------------------------------------------------------------------------------

def load_doc():
    with open(fsi.DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sample_images_json_is_valid():
    doc = load_doc()
    assert fsi.validate_doc(doc) == []
    imgs = doc["images"]
    assert len(imgs) == 6
    assert sum(e["purpose"] == "classify" for e in imgs) == 4
    assert sum(e["purpose"] == "detect" for e in imgs) == 2
    assert all(e["license"] in ("CC0-1.0", "PD") for e in imgs)
    assert len({e["sha256"] for e in imgs}) == 6 and len({e["name"] for e in imgs}) == 6


def test_validate_doc_rejects_bad_entries():
    doc = load_doc()
    doc["images"][0]["license"] = "CC-BY-SA-4.0"
    assert any("CC0-1.0 or PD" in b for b in fsi.validate_doc(doc))
    doc = load_doc()
    doc["images"][1]["license_source"] = "CC BY-SA 4.0"
    assert any("license_source" in b for b in fsi.validate_doc(doc))
    doc = load_doc()
    doc["images"][2]["sha256"] = "abc"
    assert any("sha256" in b for b in fsi.validate_doc(doc))
    doc = load_doc()
    del doc["images"][3]["expect"]
    assert fsi.validate_doc(doc)
    doc = load_doc()
    doc["images"][4]["purpose"] = "classify"
    assert any("4 classify" in b for b in fsi.validate_doc(doc))
    assert fsi.validate_doc({"schema": "x", "images": []})


def test_expectation_helpers():
    ex = {"top1_any_of": ["Indian elephant", "African elephant"], "min_score": 0.5}
    assert fsi.eval_classify("Indian elephant, Elephas maximus", 0.99, ex)[0]
    assert not fsi.eval_classify("Indian elephant, Elephas maximus", 0.49, ex)[0]
    assert not fsi.eval_classify("tusker", 0.9, ex)[0]
    assert fsi.eval_classify("violin, fiddle", 0.8, {"top1_any_of": ["violin"], "min_score": 0.5})[0]
    d = {"min_detections": 3, "labels_include": ["person", "car"]}
    assert fsi.eval_detect(["person", "person", "car"], d)[0]
    assert not fsi.eval_detect(["person", "person"], d)[0]          # 2 < 3
    assert not fsi.eval_detect(["person", "person", "dog"], d)[0]   # car missing
    assert not fsi.eval_detect(["person"] * 3, {"min_detections": 3, "labels_include": ["person"]})[0]  # 1 distinct


# ---- fetch_sample_images: download path with a patched http_get --------------------------------------------------

def make_jpeg_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def test_fetch_all_resizes_and_verifies(tmp_path, monkeypatch):
    doc = load_doc()
    sizes = [(2000, 1000), (500, 700), (1024, 1024), (1500, 3000), (64, 48), (1025, 10)]
    blobs = {}
    for e in doc["images"]:
        e.pop("crop", None)  # the real crop box does not fit these synthetic sizes (crop has its own test)
    for e, (w, h) in zip(doc["images"], sizes):
        b = make_jpeg_bytes(w, h)
        e["file_url"] = "https://example.invalid/%s.jpg" % e["name"]
        e["sha256"] = hashlib.sha256(b).hexdigest()
        blobs[e["file_url"]] = b
    calls = []
    monkeypatch.setattr(fsi.fti, "http_get", lambda url, tries=5: calls.append(url) or blobs[url])
    out = tmp_path / "not" / "yet" / "there"
    paths = fsi.fetch_all(doc, str(out), str(tmp_path / "cache"))
    assert len(calls) == 6 and len(paths) == 6
    for e, (w, h) in zip(doc["images"], sizes):
        with Image.open(paths[e["name"]]) as im:
            assert im.mode == "RGB" and im.format == "PNG"
            assert max(im.size) == min(max(w, h), 1024)
            assert abs(im.size[0] * h - im.size[1] * w) <= max(w, h)  # aspect kept to a pixel
    fsi.fetch_all(doc, str(out), str(tmp_path / "cache"))
    assert len(calls) == 6, "second run must use the cache"
    # a tampered download is refused
    doc["images"][0]["sha256"] = "0" * 64
    for f in (tmp_path / "cache").iterdir():
        f.unlink()
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        fsi.fetch_all(doc, str(out), str(tmp_path / "cache"))


def test_fetch_all_refuses_bad_license_before_any_download(tmp_path, monkeypatch):
    doc = load_doc()
    doc["images"][0]["license"] = "CC-BY-4.0"
    monkeypatch.setattr(fsi.fti, "http_get", lambda *a, **k: pytest.fail("download attempted"))
    with pytest.raises(SystemExit, match="invalid"):
        fsi.fetch_all(doc, str(tmp_path / "o"), str(tmp_path / "c"))


# ---- fetch_reference_models --out -------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_hf_env(monkeypatch):
    """main() sets HF_* environment variables; register them so every test restores them."""
    for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_DISABLE_SYMLINKS_WARNING"):
        monkeypatch.setenv(k, os.environ.get(k, ""))
        if k not in os.environ or os.environ[k] == "":
            monkeypatch.delenv(k)


def run_main(monkeypatch, argv, fake_process):
    monkeypatch.setattr(frm, "process", fake_process)
    monkeypatch.setattr(sys, "argv", ["fetch_reference_models.py"] + argv)
    frm.main()


def test_out_writes_under_out_and_never_touches_dev(tmp_path, monkeypatch, capsys):
    dev = tmp_path / "dev_must_not_exist"
    monkeypatch.setenv("MODELPIPELINES_DEV_DIR", str(dev))
    seen = []

    def fake(slug, repo, prefix, task, ein, eout, out_root, cache, tmp_root, manifest, force):
        seen.append((slug, out_root, cache, tmp_root, force))
        os.makedirs(os.path.join(out_root, slug))
        manifest[slug] = {"repo": repo, "license": "apache-2.0"}

    out = tmp_path / "a" / "b" / "models"   # parent does not exist
    run_main(monkeypatch, ["--out", str(out), "--only", "resnet-50"], fake)
    assert [s[0] for s in seen] == ["resnet-50"]
    assert seen[0][1] == str(out) and seen[0][2] is None and seen[0][3].startswith(str(out))
    assert (out / "resnet-50").is_dir()
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == {
        "resnet-50": {"repo": "microsoft/resnet-50", "license": "apache-2.0"}}
    assert not dev.exists()
    assert "Import each folder: in the Content Browser right-click the folder Content/Models > Model Pipelines > Import Hugging Face Model Folder... (assets go to /Game/Models; see Docs/ImportModel.md)" \
        in capsys.readouterr().out


def test_out_all_four_slugs_and_cache_option(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELPIPELINES_DEV_DIR", str(tmp_path / "nodev"))
    slugs = []
    caches = []

    def fake(slug, repo, prefix, task, ein, eout, out_root, cache, tmp_root, manifest, force):
        slugs.append(slug)
        caches.append(cache)

    run_main(monkeypatch, ["--out", str(tmp_path / "o"), "--cache", str(tmp_path / "hf")], fake)
    assert slugs == ["all-MiniLM-L6-v2", "distilbert-sst2", "resnet-50", "rtdetr_r18vd"]
    assert set(caches) == {str(tmp_path / "hf")}


def test_default_without_out_is_the_dev_reference_folder(tmp_path, monkeypatch):
    dev = tmp_path / "dev"
    monkeypatch.setenv("MODELPIPELINES_DEV_DIR", str(dev))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    got = []
    run_main(monkeypatch, ["--only", "distilbert-sst2"],
             lambda slug, repo, prefix, task, ein, eout, out_root, cache, tmp_root, manifest, force:
             got.append((out_root, cache, tmp_root)))
    assert got == [(str(dev / "models" / "reference"), str(dev / "cache" / "hf"), str(dev / "tmp"))]
    assert os.environ["HF_HOME"] == str(dev / "cache" / "hf")


def test_out_is_a_file_gives_a_clear_error(tmp_path, monkeypatch):
    f = tmp_path / "afile"
    f.write_text("x")
    monkeypatch.setenv("MODELPIPELINES_DEV_DIR", str(tmp_path / "nodev"))
    with pytest.raises(SystemExit, match="cannot create the output folder"):
        run_main(monkeypatch, ["--out", str(f / "sub")], lambda *a, **k: None)


def test_unknown_slug_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="unknown slug"):
        run_main(monkeypatch, ["--out", str(tmp_path / "o"), "--only", "nope"], lambda *a, **k: None)


# ---- requirements completeness ------------------------------------------------------------------------------------

def imported_modules(script, seen=None):
    """Dotted module names imported anywhere (also inside functions) by `script` and the local modules it imports."""
    seen = set() if seen is None else seen
    mods = set()
    path = os.path.join(TOOLS, script)
    seen.add(script)
    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
            # "from pkg import sub" may name a submodule; those are resolved below only if they are modules
            names += [node.module + "." + a.name for a in node.names if a.name != "*"]
        for n in names:
            top = n.split(".")[0]
            if os.path.isfile(os.path.join(TOOLS, top + ".py")):
                if top + ".py" not in seen:
                    mods |= imported_modules(top + ".py", seen)
                continue
            mods.add(n)
    return mods


def providing_distributions(module):
    """Distributions that ship `module` (dotted). None if stdlib. Namespace packages are resolved by file path."""
    top = module.split(".")[0]
    if top in sys.stdlib_module_names:
        return None
    rel = module.replace(".", "/")
    dists = set()
    for dist_name in md.packages_distributions().get(top, []):
        files = md.distribution(dist_name).files or []
        for f in files:
            fp = f.as_posix()
            if fp == rel + ".py" or fp.startswith(rel + "/"):
                dists.add(dist_name)
                break
    if not dists and "." not in module:
        dists = set(md.packages_distributions().get(top, []))
    return dists


def required_distributions():
    found = {}
    for script in ("fetch_reference_models.py", "fetch_sample_images.py"):
        for mod in imported_modules(script):
            d = providing_distributions(mod)
            if d is None:
                continue
            if not d:
                continue  # "from pkg import name" where name is not a module
            found[mod] = d
    return found


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_requirements_list_every_distribution_the_scripts_import():
    pinned = {norm(n) for n in parse_requirements(REQ)}
    found = required_distributions()
    assert "optimum.exporters.onnx" in found, "the AST walk must see the import inside export_optimum()"
    for mod, dists in sorted(found.items()):
        assert {norm(d) for d in dists} & pinned, "%s is provided by %s, none pinned in requirements.txt" % (
            mod, sorted(dists))
    print("distributions found:", sorted({norm(d) for ds in found.values() for d in ds}))


def test_optimum_exporter_is_provided_by_optimum_onnx():
    assert providing_distributions("optimum.exporters.onnx") == {"optimum-onnx"}


# ---- failed fetch leaves nothing behind ---------------------------------------------------------------------------

def fake_hub(monkeypatch, tmp_path):
    import huggingface_hub
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "README.md").write_text("---\nlicense: apache-2.0\n---\n", encoding="utf-8")
    (snap / "config.json").write_text("{}", encoding="utf-8")

    class Api:
        def list_repo_commits(self, repo):
            class C:
                commit_id = "1110a243" + "0" * 32
            return [C()]

        def list_repo_files(self, repo, revision=None):
            return ["README.md", "config.json", "model.safetensors"]

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(snap / "README.md"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    return snap


MINILM = frm.MODELS[0]


def call_process(out, manifest):
    slug, repo, prefix, task, ein, eout = MINILM
    frm.process(slug, repo, prefix, task, ein, eout, str(out), None, str(out / ".tmp"), manifest, False)


def test_failed_export_leaves_no_folder(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("export exploded")
    monkeypatch.setattr(frm, "export_optimum", boom)
    out = tmp_path / "out"
    out.mkdir()
    manifest = {}
    with pytest.raises(RuntimeError):
        call_process(out, manifest)
    assert not (out / "all-MiniLM-L6-v2").exists()
    assert not (out / ".tmp" / "stage_all-MiniLM-L6-v2").exists()
    assert manifest == {}


def test_failed_export_replaces_nothing_and_removes_stale_empty_folder(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(frm, "export_optimum", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    out = tmp_path / "out"
    (out / "all-MiniLM-L6-v2").mkdir(parents=True)   # the empty folder a failed run used to leave
    with pytest.raises(RuntimeError):
        call_process(out, {})
    assert not (out / "all-MiniLM-L6-v2").exists()


def test_missing_exporter_package_gives_one_clear_line(tmp_path, monkeypatch, capsys):
    fake_hub(monkeypatch, tmp_path)

    def noexp(*a, **k):
        raise ModuleNotFoundError("No module named 'optimum.exporters.onnx'", name="optimum.exporters.onnx")
    monkeypatch.setattr(frm, "export_optimum", noexp)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit) as ei:
        call_process(out, {})
    msg = str(ei.value)
    nl = chr(10)
    assert nl not in msg and "optimum.exporters.onnx" in msg and "pip install -r requirements.txt" in msg
    assert not (out / "all-MiniLM-L6-v2").exists()


def test_successful_export_is_moved_into_place_and_detected_up_to_date(tmp_path, monkeypatch, capsys):
    import onnx
    from onnx import TensorProto, helper
    fake_hub(monkeypatch, tmp_path)

    def tiny_export(repo_dir, task, out_dir):
        ins = [helper.make_tensor_value_info(n, TensorProto.INT64, [1, 4]) for n in frm.MODELS[0][4]]
        out = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Cast", [frm.MODELS[0][4][0]], ["last_hidden_state"], to=TensorProto.FLOAT)
        g = helper.make_graph([node], "g", ins, [out])
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]),
                  os.path.join(out_dir, "model.onnx"))
    monkeypatch.setattr(frm, "export_optimum", tiny_export)
    monkeypatch.setattr(frm, "sanity_check", lambda *a, **k: {"last_hidden_state": 0.0})
    monkeypatch.setattr(frm, "ensure_tokenizer_json", lambda *a, **k: None)
    out = tmp_path / "out"
    out.mkdir()
    manifest = {}
    call_process(out, manifest)
    folder = out / "all-MiniLM-L6-v2"
    assert (folder / "model.onnx").is_file() and (folder / "modelpipelines.json").is_file()
    assert (folder / "config.json").is_file() and not (out / ".tmp" / "stage_all-MiniLM-L6-v2").exists()
    capsys.readouterr()
    call_process(out, manifest)
    assert "up to date" in capsys.readouterr().out
