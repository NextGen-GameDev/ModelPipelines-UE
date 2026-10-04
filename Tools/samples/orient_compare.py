"""Compare the --orient-check captures with the source photos under the 8 flips/rotations.

Run with the Pillow-capable Python (not the editor's):
  python orient_compare.py --captures <dir> --images <dir>
Prints, per photo, the normalized correlation of the capture's board region with each transform of the source PNG.
The board is expected centred in the capture (768x432, FOV 90, board 350 uu away): 1 uu = 384/350 px.
The identity transform must win on every photo.
"""
import argparse, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sample_maps as B

T = {
    "identity": lambda im: im,
    "mirror_h": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "flip_v": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    "rot180": lambda im: im.transpose(Image.ROTATE_180),
    "rot90": lambda im: im.transpose(Image.ROTATE_90),
    "rot270": lambda im: im.transpose(Image.ROTATE_270),
    "transpose": lambda im: im.transpose(Image.TRANSPOSE),
    "transverse": lambda im: im.transpose(Image.TRANSVERSE),
}

def gray(im, size):
    a = np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.float64)
    return a

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--captures", required=True); p.add_argument("--images", required=True)
    a = p.parse_args()
    ok = True
    for m, _c, kind, photos in B.MAPS:
        if kind != "vision": continue
        for i, name in enumerate(photos):
            cap = Image.open(os.path.join(a.captures, "orient_%s_%d.png" % (m, i))).convert("RGB")
            src = Image.open(os.path.join(a.images, name + ".png")).convert("RGB")
            wpx, hpx = B.png_size(os.path.join(a.images, name + ".png"))
            w_uu, h_uu = B.board_size(wpx, hpx)
            k = (cap.width / 2.0) / B.BOARD_DISTANCE
            cw, ch = int(w_uu * k * 0.96), int(h_uu * k * 0.96)   # 4% inset: skip the board edge
            x0, y0 = (cap.width - cw) // 2, (cap.height - ch) // 2
            crop = cap.crop((x0, y0, x0 + cw, y0 + ch))
            size = (64, max(8, int(64 * ch / cw)))
            c = gray(crop, size)
            scores = {n: ncc(c, gray(f(src), size)) for n, f in T.items()}
            best = max(scores, key=scores.get)
            ok &= best == "identity"
            print("%-24s best=%-10s " % (name, best) + " ".join("%s=%.3f" % kv for kv in scores.items()))
    print("ORIENT OK" if ok else "ORIENT MISMATCH")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
