"""Shared pycocotools evaluator so all 3 models are scored identically."""
import contextlib, io, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

METRICS = [("AP@[.50:.95|all]", "mAP"), ("AP@[.50|all]", "mAP_50"), ("AP@[.75|all]", "mAP_75"),
           ("AP@[small]", "mAP_small"), ("AP@[medium]", "mAP_medium"), ("AP@[large]", "mAP_large"),
           ("AR@[maxDet=1]", "AR_1"), ("AR@[maxDet=10]", "AR_10"), ("AR@[maxDet=100]", "AR_100"),
           ("AR@[small]", "AR_small"), ("AR@[medium]", "AR_medium"), ("AR@[large]", "AR_large")]


def evaluate_coco(gt_json, predictions, out_json=None, model_name="model"):
    if out_json:
        json.dump(predictions, open(out_json, "w", encoding="utf-8"))
    if not predictions:
        print("[warn] no predictions"); return {k: 0.0 for _, k in METRICS}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json); dt = gt.loadRes(predictions)
        e = COCOeval(gt, dt, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    print("\n==== COCO bbox: %s ====" % model_name)
    for (lab, _), v in zip(METRICS, e.stats):
        print("  %-18s = %.4f" % (lab, v))
    return {k: float(v) for (_, k), v in zip(METRICS, e.stats)}


def per_class_ap(gt_json, predictions, class_names=None):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        if not predictions:
            return {}
        dt = gt.loadRes(predictions); e = COCOeval(gt, dt, "bbox"); e.evaluate(); e.accumulate()
    prec, out = e.eval["precision"], {}
    for k, cid in enumerate(gt.getCatIds()):
        p = prec[:, :, k, 0, -1]; p = p[p > -1]
        name = gt.cats[cid]["name"] if class_names is None else class_names[k]
        out[name] = float(p.mean()) if p.size else float("nan")
    return out
