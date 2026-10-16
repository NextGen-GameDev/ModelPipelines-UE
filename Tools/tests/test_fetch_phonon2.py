"""WP9S.6c offline tests for fetch_phonon2.py: SHA256SUMS, tar safety, path injection, base-repo patterns."""
import hashlib
import io
import os
import sys
import tarfile

import pytest

TOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOOLS)
import fetch_phonon2 as fp  # noqa: E402

HELLO = hashlib.sha256(b"hello").hexdigest()   # computed by hashlib, independent of the code under test


def test_parse_sha256sums():
    txt = "# c\n%s  a.txt\n\n%s *sub/b.bin\n" % (HELLO, "0" * 64)
    assert fp.parse_sha256sums(txt) == {"a.txt": HELLO, "sub/b.bin": "0" * 64}
    with pytest.raises(fp.FetchError):
        fp.parse_sha256sums("deadbeef  a.txt\n")


def test_verify_ok_and_mismatch(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "SHA256SUMS.txt").write_text("%s  a.txt\n" % HELLO)
    assert fp.verify_sums(str(tmp_path)) == ["a.txt"]
    (tmp_path / "a.txt").write_bytes(b"hellO")
    with pytest.raises(fp.FetchError, match="a.txt"):
        fp.verify_sums(str(tmp_path))
    (tmp_path / "a.txt").unlink()
    with pytest.raises(fp.FetchError, match="a.txt"):
        fp.verify_sums(str(tmp_path))


@pytest.mark.parametrize("name", ["/etc/x", "../x", "a/../../x", "C:/x", "a\\..\\x"])
def test_unsafe_members_refused(tmp_path, name):
    with pytest.raises(fp.FetchError):
        fp.check_member(name, str(tmp_path))


def test_safe_member_ok(tmp_path):
    assert fp.check_member("a/b.txt", str(tmp_path)) == os.path.abspath(str(tmp_path / "a" / "b.txt"))


def _tar(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for ti, data in members:
            tf.addfile(ti, io.BytesIO(data) if data is not None else None)
    buf.seek(0)
    return buf


def _reg(name, data):
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    return ti, data


def test_extract_regular_files(tmp_path):
    buf = _tar([_reg("model.fermion", b"abc"), _reg("d/x.json", b"{}")])
    with tarfile.open(fileobj=buf, mode="r|") as tf:
        names = fp.safe_extract_tar(tf, str(tmp_path / "o"))
    assert names == ["model.fermion", "d/x.json"]
    assert (tmp_path / "o" / "model.fermion").read_bytes() == b"abc"


def test_extract_refuses_traversal_and_links(tmp_path):
    for ti, data in (_reg("../evil", b"x"), _reg("/abs", b"x")):
        with tarfile.open(fileobj=_tar([(ti, data)]), mode="r|") as tf:
            with pytest.raises(fp.FetchError):
                fp.safe_extract_tar(tf, str(tmp_path / "o"))
    link = tarfile.TarInfo("l")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    with tarfile.open(fileobj=_tar([(link, None)]), mode="r|") as tf:
        with pytest.raises(fp.FetchError):
            fp.safe_extract_tar(tf, str(tmp_path / "o2"))
    assert not (tmp_path / "evil").exists()


def test_base_patterns_exclude_weights():
    pats = fp.base_allow_patterns()
    assert "config.json" in pats and "tokenizer.json" in pats
    import fnmatch
    for bad in ("model.safetensors", "parakeet-tdt-0.6b-v3.nemo", "pytorch_model.bin"):
        assert not any(fnmatch.fnmatch(bad, p) for p in pats), bad
    for p in pats:
        assert not any(p == f or f.strip("*") in p for f in fp.FORBIDDEN_PATTERNS)
    from stt import phonon2_export as ex
    assert set(ex.BASE_COPY) <= set(pats)
    assert fp.CONTAINER_SHA256 == ex.CONTAINER_SHA256 and fp.PHONON_REV == ex.REVISION and fp.BASE_REV == ex.BASE_REVISION


def test_out_needs_no_dev_dir(tmp_path, monkeypatch):
    """With --out: fake download + fake export; the dev dir is a non-existent path and is never created or read."""
    pytest.importorskip("zstandard", reason="needs the pipelines-stt env")
    import zstandard
    from stt import phonon2_export as ex
    ghost = tmp_path / "no_dev"
    monkeypatch.setenv("MODELPIPELINES_DEV_DIR", str(ghost))

    container = b"fake container"
    monkeypatch.setattr(fp, "CONTAINER_SHA256", hashlib.sha256(container).hexdigest())
    tar_buf = _tar([_reg("model.fermion", container)])
    zst = zstandard.ZstdCompressor().compress(tar_buf.getvalue())

    cache = tmp_path / "snap"
    phon, base = cache / "phonon", cache / "base"
    phon.mkdir(parents=True)
    base.mkdir()
    (phon / "phonon-2.bps.tar.zst").write_bytes(zst)
    (phon / "NOTICE").write_text("n")
    (phon / "LICENSE-WEIGHTS-CC-BY-4.0.txt").write_text("l")
    (phon / "SHA256SUMS.txt").write_text("%s  phonon-2.bps.tar.zst\n%s  NOTICE\n" % (
        hashlib.sha256(zst).hexdigest(), hashlib.sha256(b"n").hexdigest()))
    for f in fp.BASE_FILES:
        (base / f).write_text("x")

    calls = []

    def fake_download(repo, rev, pats, c):
        calls.append((repo, rev, pats))
        return str(phon if repo == fp.PHONON_ID else base)

    seen = []

    def fake_export(variant, dest):
        p = ex.dev_paths()
        seen.append((variant, p))
        assert p["dev"] is None and os.path.isfile(p["container"]) and p["base_dir"] == str(base)
        os.makedirs(dest)
        for f in ("NOTICE", "LICENSE-WEIGHTS-CC-BY-4.0.txt", "SOURCE.json", "modelpipelines.json"):
            open(os.path.join(dest, f), "w").write("x")
        return 0

    out = tmp_path / "out"
    rc = fp.main(["--out", str(out), "--variant", "int8"], download=fake_download, export=fake_export)
    assert rc == 0
    assert [v for v, _ in seen] == ["fp32", "int8"]
    assert (out / "phonon-2-int8" / "NOTICE").is_file()
    assert not (out / "_work").exists()                       # deleted without --keep-fp32
    assert not ghost.exists()                                 # dev dir never touched
    assert calls[0][:2] == (fp.PHONON_ID, fp.PHONON_REV) and calls[1][:2] == (fp.BASE_ID, fp.BASE_REV)
    assert ex.dev_paths()["dev"] is not None                  # override cleared afterwards


def test_checksum_mismatch_exits_nonzero(tmp_path, capsys):
    d = tmp_path / "p"
    d.mkdir()
    (d / "a.tar.zst").write_bytes(b"x")
    (d / "SHA256SUMS.txt").write_text("%s  a.tar.zst\n" % ("1" * 64))
    rc = fp.main(["--out", str(tmp_path / "o")], download=lambda *a: str(d), export=lambda *a: 0)
    assert rc == 1 and "a.tar.zst" in capsys.readouterr().out
