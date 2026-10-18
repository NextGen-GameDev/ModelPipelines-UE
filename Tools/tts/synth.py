#!/usr/bin/env python
"""Synthesizes chosen corpus lines with Kokoro v1.0 (ONNX, voice af_heart) from three G2P outputs: the misaki
reference, larroy as-is and the misaki lexicon port, in that order with 0.7 s gaps, one WAV per line in
<dev>/tts/samples/. Prints each clip's audio length and synthesis time.
Needs setup_g2p_bench.py --with-model and compare.py first.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_g2p_bench as setup  # noqa: E402

PICKS = {
    "1_dialogue_stress": "I wouldn't go into those woods after dark if I were you.",
    "2_dialogue_stress": "The blacksmith needs more iron ore before he can finish your sword.",
    "3_number": "That will be 1,250 gold coins.",
    "4_compound_name": "Old Grimbald runs the tavern in Oakhollow.",
    "5_compound_name": "Speak with Elder Morwen in Thistlewood.",
    "6_homograph": "I read that book last winter, and I'll read it again this winter.",
    "7_interjection": "Shh, they'll hear us.",
}
SYSTEMS = [("larroy", "larroy (CMUdict + g2p_en)"), ("port", "misaki lexicon port + g2p_en OOV")]
SAMPLE_RATE = 24000


def main():
    import numpy as np
    import soundfile as sf
    from kokoro_onnx import Kokoro

    with open(os.path.join(setup.tts_dir(), "report.json"), encoding="utf-8") as f:
        report = json.load(f)
    kokoro = Kokoro(os.path.join(setup.model_dir(), "kokoro-v1.0.onnx"),
                    os.path.join(setup.model_dir(), "voices-v1.0.bin"))
    out = os.path.join(setup.tts_dir(), "samples")
    os.makedirs(out, exist_ok=True)
    gap = np.zeros(int(SAMPLE_RATE * 0.7), dtype=np.float32)
    for fname, text in PICKS.items():
        i = report["texts"].index(text)
        variants = [("misaki", report["ref"][i])]
        variants += [(label, report["systems"][name]["lines"][i]["hyp"]) for label, name in SYSTEMS]
        clips = []
        for label, ps in variants:
            t = time.time()
            audio, sr = kokoro.create(ps, voice="af_heart", is_phonemes=True)
            assert sr == SAMPLE_RATE, sr
            print(f"{fname:20s} {label:7s} {len(audio) / sr:4.1f}s audio in {time.time() - t:4.2f}s  {ps}")
            clips += [audio, gap]
        sf.write(os.path.join(out, f"{fname}.wav"), np.concatenate(clips), SAMPLE_RATE)
    print("wrote", out)


if __name__ == "__main__":
    main()
