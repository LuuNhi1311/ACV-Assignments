"""Shared COCO-detection evaluator used by every model in this lab.

Wrapping ``pycocotools`` here (the same library the YOLOv4 ``evaluate_metrics.py``
uses) guarantees DETR, Faster R-CNN and YOLO are scored with an **identical**
protocol, so the numbers in the report are directly comparable.

``evaluate_coco`` takes the ground-truth COCO json and a list of predictions::

    [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]

and prints the canonical 12-line COCO summary, returning the metrics as a dict.
"""
import contextlib
import io
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# (display label, short key) for the 12 COCO bbox statistics, in COCOeval order.
METRICS = [
    ("AP @[IoU=0.50:0.95 | all   ]", "mAP"),
    ("AP @[IoU=0.50      | all   ]", "mAP_50"),
    ("AP @[IoU=0.75      | all   ]", "mAP_75"),
    ("AP @[IoU=0.50:0.95 | small ]", "mAP_small"),
    ("AP @[IoU=0.50:0.95 | medium]", "mAP_medium"),
    ("AP @[IoU=0.50:0.95 | large ]", "mAP_large"),
    ("AR @[IoU=0.50:0.95 | maxDet=  1]", "AR_1"),
    ("AR @[IoU=0.50:0.95 | maxDet= 10]", "AR_10"),
    ("AR @[IoU=0.50:0.95 | maxDet=100]", "AR_100"),
    ("AR @[IoU=0.50:0.95 | small ]", "AR_small"),
    ("AR @[IoU=0.50:0.95 | medium]", "AR_medium"),
    ("AR @[IoU=0.50:0.95 | large ]", "AR_large"),
]


def evaluate_coco(gt_json, predictions, out_json=None, model_name="model"):
    """Score ``predictions`` against ``gt_json`` and return a metric dict."""
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(predictions, f)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_json)

    if not predictions:
        print("[warn] no predictions produced — all metrics are 0.0")
        return {key: 0.0 for _, key in METRICS}

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats
    print("\n================= SUMMARY (COCO bbox: %s) =================" % model_name)
    for (label, _), val in zip(METRICS, stats):
        print("  %-34s = %.4f" % (label, val))
    print("=" * (44 + len(model_name)))
    print("mAP(0.50:0.95) = %.4f    mAP@0.50 = %.4f" % (stats[0], stats[1]))
    return {key: float(val) for (_, key), val in zip(METRICS, stats)}


def per_class_ap(gt_json, predictions, class_names=None):
    """Return {class_name or id: AP@[0.50:0.95]} for a per-class breakdown table."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_json)
        if not predictions:
            return {}
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

    # precision dims: [T(iou), R(recall), K(class), A(area), M(maxDet)]
    precision = coco_eval.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    out = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        ap = float(p.mean()) if p.size else float("nan")
        name = coco_gt.cats[cat_id]["name"] if class_names is None else class_names[k]
        out[name] = ap
    return out
