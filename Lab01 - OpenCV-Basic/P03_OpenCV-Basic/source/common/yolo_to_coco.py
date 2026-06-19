"""Ultralytics-YOLO split -> COCO json (0-indexed categories, matching names order)."""
import argparse, json, os
import yaml
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_names(data_yaml):
    n = yaml.safe_load(open(data_yaml, encoding="utf-8"))["names"]
    return [n[k] for k in sorted(n, key=int)] if isinstance(n, dict) else list(n)


def _resolve_split_dir(cfg, data_yaml, split):
    root = cfg.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    if not os.path.isabs(root):
        root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
    e = cfg[split]
    return os.path.normpath(e if os.path.isabs(e) else os.path.join(root, e))


def _label_path(img_path):
    parts = img_path.replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"; break
    return os.path.splitext("/".join(parts))[0] + ".txt"


def convert(data_yaml, split, out_path=None):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    names = load_names(data_yaml)
    img_dir = _resolve_split_dir(cfg, data_yaml, split)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
    images, anns, aid = [], [], 1
    for iid, fn in enumerate(files, 1):
        with Image.open(os.path.join(img_dir, fn)) as im:
            W, H = im.size
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        lbl = _label_path(os.path.join(img_dir, fn))
        if not os.path.exists(lbl):
            continue
        for line in open(lbl, encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            bw, bh = w * W, h * H
            if bw > 0 and bh > 0:
                anns.append({"id": aid, "image_id": iid, "category_id": c, "iscrowd": 0,
                             "bbox": [cx * W - bw / 2, cy * H - bh / 2, bw, bh], "area": bw * bh})
                aid += 1
    out = out_path or os.path.join(img_dir, "_annotations.coco.json")
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": i, "name": n} for i, n in enumerate(names)]},
              open(out, "w", encoding="utf-8"))
    print("[coco] %s [%s]: %d imgs, %d boxes -> %s" % (data_yaml, split, len(images), len(anns), out))
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("data_yaml"); a.add_argument("split"); a.add_argument("-o", default=None)
    g = a.parse_args(); convert(g.data_yaml, g.split, g.o)
