"""Aggregate saved prediction JSONs into per-dataset Markdown tables for the report."""
import argparse, contextlib, io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.yolo_to_coco import convert
from common.coco_metrics import evaluate_coco

ARCHES = [("yolov8", "YOLOv8"), ("rtdetr", "RT-DETR"), ("detr", "DETR (R50)")]
COLS = [("mAP", "mAP@[.5:.95]"), ("mAP_50", "mAP@0.50"), ("mAP_75", "mAP@0.75"),
        ("mAP_small", "AP(s)"), ("mAP_medium", "AP(m)"), ("mAP_large", "AP(l)"), ("AR_100", "AR@100")]


def main():
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("-runs", default=os.path.join(src, "runs"))
    p.add_argument("-data-dir", dest="data_dir", default=os.path.join(src, "data"))
    p.add_argument("-datasets", nargs="+", default=["megafauna", "fishinv"]); p.add_argument("-split", default="test")
    a = p.parse_args()
    for ds in a.datasets:
        dy = os.path.join(a.data_dir, "%s.yaml" % ds)
        if not os.path.exists(dy):
            print("\n## %s — skip (no %s)" % (ds, dy)); continue
        with contextlib.redirect_stdout(io.StringIO()):
            gt = convert(dy, a.split)
        print("\n### %s (%s)\n" % (ds, a.split))
        print("| Model | " + " | ".join(h for _, h in COLS) + " |")
        print("|" + "---|" * (len(COLS) + 1))
        for arch, label in ARCHES:
            pred = os.path.join(a.runs, "%s_%s_pred_%s.json" % (arch, ds, a.split))
            if not os.path.exists(pred):
                print("| %s | %s |" % (label, " | ".join("_n/a_" for _ in COLS))); continue
            with contextlib.redirect_stdout(io.StringIO()):
                m = evaluate_coco(gt, json.load(open(pred, encoding="utf-8")), model_name=arch)
            print("| %s | %s |" % (label, " | ".join("%.4f" % m[k] for k, _ in COLS)))
    print("\n(Params/FPS: see runs/logs/)")


if __name__ == "__main__":
    main()
