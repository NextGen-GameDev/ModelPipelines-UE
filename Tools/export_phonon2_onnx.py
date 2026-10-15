"""Export Phonon-2 (CC-BY-4.0 container) to ONNX: encoder.onnx + decoder_joint.onnx (WP9S.1a, contract C70).

  python export_phonon2_onnx.py --variant fp32 [--out DIR] [--force]

Default output: <dev>/models/stt/phonon-2-onnx/<variant>/. Offline: reads the container and base config from
<dev>/models/stt/. A rerun is skipped when SOURCE.json hashes match every file. Run with the pipelines-stt env.
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="fp32", choices=["fp32", "fp16w", "int8", "fp16"],
                    help="fp32 = full export; fp16w/int8/fp16 are shipping variants built from fp32/ (WP9S.1b)")
    ap.add_argument("--out", default=None, help="output folder (default <dev>/models/stt/phonon-2-onnx/<variant>)")
    ap.add_argument("--force", action="store_true", help="re-export even if SOURCE.json hashes match")
    a = ap.parse_args(argv)

    try:
        from stt import phonon2_export as ex
    except ImportError as e:
        print("NEEDS: package missing: %s (use the pipelines-stt env)" % e)
        return 2
    p = ex.dev_paths()
    out = os.path.abspath(a.out or os.path.join(p["out_root"], a.variant))
    if not a.force and ex.up_to_date(out):
        print("up to date (SOURCE.json hashes match), skipping: %s" % out)
        _print_hashes(out)
        return 0
    if not os.path.isfile(p["container"]):
        print("container not found: %s" % p["container"])
        return 1
    got = ex.sha256_file(p["container"])
    if got != ex.CONTAINER_SHA256:
        print("container sha256 mismatch: %s" % got)
        return 1

    if a.variant != "fp32":
        return _variant(a, p, out, ex)
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        print("loading model from container ...")
        model, _ = ex.load_model(p["container"], p["base_dir"])
        print("exporting (opset %d) ..." % ex.OPSET)
        ex.export_graphs(model, tmp)
        del model
        ex.copy_support_files(tmp)
        ex.check_onnx(tmp)
        ex.assert_c70(tmp)
        ex.write_source_json(tmp, a.variant)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    print("wrote %s" % out)
    _print_hashes(out)
    return 0


def _variant(a, p, out, ex):
    from stt import phonon2_variants as pv
    fp32 = os.path.join(p["out_root"], "fp32")
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        model = None
        if a.variant == "fp16":
            model, _ = ex.load_model(p["container"], p["base_dir"])
        notes = pv.build_variant(a.variant, fp32, tmp, model)
        del model
        s1 = pv.check_s1(tmp)
        print("S1 check: %s" % s1)
        pv.write_variant_source(tmp, a.variant, fp32, notes)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    print("wrote %s" % out)
    _print_hashes(out)
    return 0


def _print_hashes(folder):
    from stt import phonon2_export as ex
    for f in ("encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx"):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp):
            print("%-20s %12d bytes  sha256 %s" % (f, os.path.getsize(fp), ex.sha256_file(fp)))


if __name__ == "__main__":
    sys.exit(main())
