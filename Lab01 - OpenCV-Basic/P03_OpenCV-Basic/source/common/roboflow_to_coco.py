"""Convert a Roboflow *yolov4pytorch* split into a COCO-detection JSON.

The Roboflow chess export stores each split (``train`` / ``valid`` / ``test``) as
a folder containing the images plus two text files:

* ``_annotations.txt`` — one line per image::

      image.jpg x1,y1,x2,y2,class_id  x1,y1,x2,y2,class_id  ...

  Boxes are absolute **pixel xyxy** coordinates and ``class_id`` is 0-indexed.
* ``_classes.txt`` — one class name per line, in class-id order.

Both DETR (HuggingFace) and Faster R-CNN (torchvision) consume COCO JSON, and the
shared evaluator (:mod:`common.coco_metrics`) needs the ground truth in the same
format, so we convert once here.

Category ids are kept **0-indexed** to match the YOLO class ids and ``_classes.txt``
order, so the three models report metrics over identical category ids.
"""
import argparse
import json
import os

from PIL import Image


def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def convert(split_dir, out_path=None):
    ann_txt = os.path.join(split_dir, "_annotations.txt")
    classes = load_classes(os.path.join(split_dir, "_classes.txt"))

    with open(ann_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    images, annotations = [], []
    ann_id = 1
    for img_id, line in enumerate(lines, start=1):
        parts = line.split(" ")
        fname = parts[0]
        with Image.open(os.path.join(split_dir, fname)) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        for box in parts[1:]:
            if not box:
                continue
            x1, y1, x2, y2, cls = (int(float(v)) for v in box.split(","))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [x1, y1, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(classes)],
    }
    out_path = out_path or os.path.join(split_dir, "_annotations.coco.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print("[coco] %s: %d images, %d boxes, %d classes -> %s"
          % (split_dir, len(images), len(annotations), len(classes), out_path))
    return out_path


def get_args():
    p = argparse.ArgumentParser(description="Roboflow yolov4pytorch -> COCO json")
    p.add_argument("split_dir", help="folder with _annotations.txt + _classes.txt + images")
    p.add_argument("-o", "--out", default=None, help="output json (default: <split>/_annotations.coco.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert(args.split_dir, args.out)
