import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fetch_sample_images as fsi  # noqa: E402


def test_validate_crop_ok_and_none():
    assert fsi.validate_crop(None) == []
    assert fsi.validate_crop([1, 2, 10, 20], (10, 20)) == []


@pytest.mark.parametrize("crop", [[1, 2, 3], [1, 2, 3, 4.0], ["a", 0, 1, 1], [0, 0, True, 5], "abcd"])
def test_validate_crop_shape_errors(crop):
    assert fsi.validate_crop(crop)


@pytest.mark.parametrize("crop", [[5, 0, 5, 10], [6, 0, 5, 10], [0, 5, 10, 5], [-1, 0, 5, 5]])
def test_validate_crop_order_errors(crop):
    assert fsi.validate_crop(crop)


def test_validate_crop_outside_source():
    assert fsi.validate_crop([0, 0, 11, 20], (10, 20))
    assert fsi.validate_crop([0, 0, 10, 21], (10, 20))


def test_resize_to_png_applies_crop(tmp_path):
    from PIL import Image
    im = Image.new("RGB", (10, 8), (0, 0, 0))
    for x in range(4, 10):
        for y in range(2, 8):
            im.putpixel((x, y), (200, 100, 50))
    src, dst = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    im.save(src)
    assert fsi.resize_to_png(src, dst, [4, 2, 10, 8]) == (6, 6)
    out = Image.open(dst)
    assert out.size == (6, 6)
    assert out.getpixel((0, 0)) == (200, 100, 50) and out.getpixel((5, 5)) == (200, 100, 50)
    assert fsi.resize_to_png(src, dst) == (10, 8)


def test_resize_to_png_bad_crop_exits(tmp_path):
    from PIL import Image
    src = str(tmp_path / "a.png")
    Image.new("RGB", (4, 4)).save(src)
    with pytest.raises(SystemExit):
        fsi.resize_to_png(src, str(tmp_path / "b.png"), [0, 0, 5, 4])


def test_validate_doc_reports_bad_crop():
    doc = {"schema": fsi.SCHEMA, "images": [{"name": "x", "crop": [3, 3, 2, 9]}]}
    assert any("crop" in b for b in fsi.validate_doc(doc))
