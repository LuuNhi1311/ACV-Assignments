"""Evaluate a trained YOLOv8 / RT-DETR checkpoint with the SHARED COCO evaluator.

Rather than relying on each framework's own metric, we run inference, dump
predictions in COCO format and score them with the same ``pycocotools`` wrapper
used for DETR. This guarantees all three models are compared under one protocol.
Also reports parameter count (M) and inference speed (FPS).
"""
import argparse
import os
import sys
import time

import torch
from pycocotools.coco import COCO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.yolo_to_coco import convert, _resolve_split_dir
from common.coco_metrics import evaluate_coco, per_class_ap

import yaml


def load_model(arch, weights):
    from ultralytics import YOLO, RTDETR
    return RTDETR(weights) if arch == "rtdetr" else YOLO(weights)


def get_args():
    p = argparse.ArgumentParser(description="COCO evaluation for YOLOv8 / RT-DETR")
    p.add_argument("-arch", choices=["yolov8", "rtdetr"], required=True)
    p.add_argument("-weights", required=True, help="trained checkpoint best.pt")
    p.add_argument("-data", required=True, help="data.yaml")
    p.add_argument("-split", default="test")
    p.add_argument("-imgsz", type=int, default=640)
    p.add_argument("-conf", type=float, default=0.001, help="low conf -> full PR curve for AP")
    p.add_argument("-device", default="0")
    p.add_argument("-out-json", dest="out_json", default=None)
    return p.parse_args()


@torch.no_grad()
def run(model, img_dir, coco, device, imgsz, conf):
    name2id = {im["file_name"]: im["id"] for im in coco.dataset["images"]}
    predictions = []
    n_imgs, total_t = 0, 0.0
    for fname, img_id in name2id.items():
        path = os.path.join(img_dir, fname)
        t0 = time.perf_counter()
        res = model.predict(path, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        total_t += time.perf_counter() - t0
        n_imgs += 1
        b = res.boxes
        if b is None or b.shape[0] == 0:
            continue
        xyxy = b.xyxy.cpu().numpy()
        scores = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), s, c in zip(xyxy, scores, cls):
            predictions.append({
                "image_id": img_id,
                "category_id": int(c),                # YOLO cls == 0-indexed COCO cat id
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(s),
            })
    fps = n_imgs / total_t if total_t else 0.0
    return predictions, fps


def main():
    args = get_args()
    model = load_model(args.arch, args.weights)
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    gt_json = convert(args.data, args.split)
    coco = COCO(gt_json)
    with open(args.data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    img_dir = _resolve_split_dir(cfg, args.data, args.split)

    print("=" * 70)
    print("Evaluating %s: %s" % (args.arch, args.weights))
    print("  data   : %s [%s]  (%d images)" % (args.data, args.split, len(coco.imgs)))
    print("  params : %.1f M" % n_params)
    print("=" * 70)

    predictions, fps = run(model, img_dir, coco, args.device, args.imgsz, args.conf)
    evaluate_coco(gt_json, predictions, out_json=args.out_json, model_name=args.arch)
    print("\nInference speed: %.2f FPS  |  Params: %.1f M" % (fps, n_params))

    print("\nPer-class AP@[0.50:0.95]:")
    for name, ap in per_class_ap(gt_json, predictions).items():
        print("  %-18s %.4f" % (name, ap))


if __name__ == "__main__":
    main()
