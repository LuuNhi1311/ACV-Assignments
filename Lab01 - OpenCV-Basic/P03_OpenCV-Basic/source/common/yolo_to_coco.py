"""Convert an Ultralytics-YOLO dataset split into a COCO-detection JSON.

The marine-detect datasets (MegaFauna, FishInv) ship in Ultralytics YOLO format:
a ``data.yaml`` pointing at image folders, and per-image label files holding
``class cx cy w h`` (all normalised to [0, 1]). Ultralytics derives the label path
from the image path by swapping the ``images`` segment for ``labels`` and the
extension for ``.txt``.

YOLOv8 and RT-DETR consume this format directly, but DETR (HuggingFace) and the
shared evaluator (:mod:`common.coco_metrics`) need COCO JSON, so we convert here.
Category ids are kept 0-indexed to match the YOLO class ids / ``names`` order, so
every model reports metrics over identical category ids.
"""
import argparse
import json
import os

import yaml
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):                       # {0: 'a', 1: 'b'} -> ['a', 'b']
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def _resolve_split_dir(cfg, data_yaml, split):
    """Return the absolute image directory for a split from a data.yaml entry."""
    root = cfg.get("path", os.path.dirname(os.path.abspath(data_yaml)))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
    entry = cfg.get(split)
    if entry is None:
        raise KeyError("split '%s' not found in %s" % (split, data_yaml))
    img_dir = entry if os.path.isabs(entry) else os.path.join(root, entry)
    return os.path.normpath(img_dir)


def _label_path(img_path):
    """Ultralytics rule: .../images/... -> .../labels/...  with .txt extension."""
    parts = img_path.replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):           # swap the LAST 'images' segment
        if parts[i] == "images":
            parts[i] = "labels"
            break
    base = "/".join(parts)
    return os.path.splitext(base)[0] + ".txt"


def convert(data_yaml, split, out_path=None):
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = load_names(data_yaml)
    img_dir = _resolve_split_dir(cfg, data_yaml, split)

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    images, annotations = [], []
    ann_id = 1
    for img_id, fname in enumerate(files, start=1):
        img_path = os.path.join(img_dir, fname)
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({"id": img_id, "file_name": fname, "width": W, "height": H})

        lbl = _label_path(img_path)
        if not os.path.exists(lbl):
            continue
        for line in open(lbl, "r", encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            cls, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            bw, bh = w * W, h * H
            x, y = (cx * W) - bw / 2.0, (cy * H) - bh / 2.0
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cls,
                "bbox": [x, y, bw, bh], "area": float(bw * bh), "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(names)],
    }
    out_path = out_path or os.path.join(img_dir, "_annotations.coco.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print("[coco] %s [%s]: %d images, %d boxes, %d classes -> %s"
          % (data_yaml, split, len(images), len(annotations), len(names), out_path))
    return out_path


def get_args():
    p = argparse.ArgumentParser(description="Ultralytics YOLO split -> COCO json")
    p.add_argument("data_yaml", help="path to data.yaml")
    p.add_argument("split", help="split key in data.yaml (train/val/test)")
    p.add_argument("-o", "--out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert(args.data_yaml, args.split, args.out)
