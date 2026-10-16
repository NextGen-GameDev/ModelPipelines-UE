"""Write the modelpipelines.json sidecar (contract C78) into Phonon-2 ONNX folders.

The sidecar tells the editor importer the task (SpeechToText) and the CC-BY-4.0 attribution. Pure standard library
(no torch/transformers) so it runs in any env. Usage: python write_phonon2_sidecar.py <folder> [<folder> ...]
"""
import argparse
import json
import os

SIDECAR = {
    "task": "SpeechToText",
    "license": {
        "spdx_id": "CC-BY-4.0",
        "source_url": "https://huggingface.co/FermionResearch/Phonon-2",
        "attribution": "Phonon-2 by Fermion Research (CC-BY-4.0), derived from NVIDIA parakeet-tdt-0.6b-v3 (CC-BY-4.0); "
                       "converted to ONNX.",
    },
}


def sidecar_bytes():
    return (json.dumps(SIDECAR, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def write_sidecar(folder):
    """Write (or overwrite) <folder>/modelpipelines.json; returns its path."""
    path = os.path.join(folder, "modelpipelines.json")
    with open(path, "wb") as f:
        f.write(sidecar_bytes())
    return path


def main():
    ap = argparse.ArgumentParser(description="Write the Phonon-2 modelpipelines.json sidecar into folders.")
    ap.add_argument("folders", nargs="+", help="Phonon ONNX folders (must exist)")
    args = ap.parse_args()
    for folder in args.folders:
        if not os.path.isdir(folder):
            raise SystemExit("not a folder: " + folder)
        print("wrote " + write_sidecar(folder))


if __name__ == "__main__":
    main()
