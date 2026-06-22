"""Confusion matrix + loss curves for every (model, dataset), unified style.

CFM is built from the saved COCO predictions (works for all 3 models). Loss
curves come from Ultralytics results.csv (yolov8/rtdetr) or the DETR train log.
Outputs -> results/figures/<model>_<dataset>_cfm.png / _loss.png

    python common/plots.py            # all available
    python common/plots.py -conf 0.25
"""
import argparse, csv, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.yolo_to_coco import convert, load_names

ARCHES = ["yolov8", "rtdetr", "detr"]


def _xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]


def _iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a, b = np.array(a, float), np.array(b, float)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    ar = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    br_ = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar[:, None] + br_[None, :] - inter + 1e-9)


def build_matrix(gt_json, pred_json, nc, conf=0.25, iou_thr=0.45):
    gt = json.load(open(gt_json, encoding="utf-8"))
    gts = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append((a["category_id"], _xyxy(a["bbox"])))
    preds = {}
    for p in json.load(open(pred_json, encoding="utf-8")):
        if p["score"] >= conf:
            preds.setdefault(p["image_id"], []).append((p["category_id"], p["score"], _xyxy(p["bbox"])))
    M = np.zeros((nc + 1, nc + 1))            # rows=pred, cols=gt; index nc = background
    for iid in set(gts) | set(preds):
        g = gts.get(iid, []); pr = sorted(preds.get(iid, []), key=lambda x: -x[1])
        iou = _iou([x[2] for x in pr], [x[1] for x in g])
        used = set()
        for i in range(len(pr)):
            j, best = -1, iou_thr
            for jj in range(len(g)):
                if jj not in used and iou[i, jj] >= best:
                    best, j = iou[i, jj], jj
            if j >= 0:
                M[pr[i][0], g[j][0]] += 1; used.add(j)
            else:
                M[pr[i][0], nc] += 1          # false positive (predicted, bg)
        for jj in range(len(g)):
            if jj not in used:
                M[nc, g[jj][0]] += 1          # missed gt (predicted background)
    return M


def plot_cfm(gt_json, pred_json, names, out_png, conf=0.25):
    nc = len(names)
    M = build_matrix(gt_json, pred_json, nc, conf=conf)
    col = M.sum(0, keepdims=True); col[col == 0] = 1
    Mn = M / col
    labels = list(names) + ["background"]
    fig, ax = plt.subplots(figsize=(max(8, nc * 0.7), max(7, nc * 0.65)))
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(nc + 1)); ax.set_yticks(range(nc + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix (col-normalised)")
    for i in range(nc + 1):
        for j in range(nc + 1):
            if Mn[i, j] > 0.005:
                ax.text(j, i, "%.2f" % Mn[i, j], ha="center", va="center",
                        fontsize=7, color="white" if Mn[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print("saved", out_png)


def plot_loss_ultra(results_csv, out_png, title):
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(results_csv, encoding="utf-8"))]
    if not rows:
        return
    ep = [float(r["epoch"]) for r in rows]
    col = lambda n: [float(r[n]) for r in rows] if n in rows[0] else None
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for c, lab in [("train/box_loss", "box"), ("train/cls_loss", "cls"), ("train/dfl_loss", "dfl"),
                   ("train/giou_loss", "giou"), ("train/l1_loss", "l1")]:
        y = col(c)
        if y:
            axs[0].plot(ep, y, label=lab)
    axs[0].set_title("%s — train loss" % title); axs[0].set_xlabel("epoch"); axs[0].legend(); axs[0].grid(alpha=.3)
    for c, lab in [("metrics/mAP50(B)", "mAP@0.50"), ("metrics/mAP50-95(B)", "mAP@[.5:.95]")]:
        y = col(c)
        if y:
            axs[1].plot(ep, y, label=lab)
    axs[1].set_title("%s — val mAP" % title); axs[1].set_xlabel("epoch"); axs[1].legend(); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print("saved", out_png)


def plot_loss_detr(log_file, out_png, title):
    ep, tr, va = [], [], []
    for line in open(log_file, encoding="utf-8"):
        m = re.search(r"\[ep (\d+)\] train ([\d.]+) val ([\d.]+)", line)
        if m:
            ep.append(int(m.group(1))); tr.append(float(m.group(2))); va.append(float(m.group(3)))
    if not ep:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, tr, label="train"); ax.plot(ep, va, label="val")
    ax.set_title("%s — Hungarian loss" % title); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print("saved", out_png)


def main():
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proj = os.path.dirname(src)
    p = argparse.ArgumentParser()
    p.add_argument("-runs", default=os.path.join(src, "runs"))
    p.add_argument("-data-dir", dest="data_dir", default=os.path.join(src, "data"))
    p.add_argument("-out", default=os.path.join(proj, "results", "figures"))
    p.add_argument("-datasets", nargs="+", default=["megafauna", "fishinv"])
    p.add_argument("-split", default="test"); p.add_argument("-conf", type=float, default=0.25)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logs = os.path.join(os.path.dirname(a.out), "logs")
    for ds in a.datasets:
        dy = os.path.join(a.data_dir, "%s.yaml" % ds)
        if not os.path.exists(dy):
            continue
        names = load_names(dy)
        gt = convert(dy, a.split)
        for arch in ARCHES:
            pred = os.path.join(a.runs, "%s_%s_pred_%s.json" % (arch, ds, a.split))
            if os.path.exists(pred):
                plot_cfm(gt, pred, names, os.path.join(a.out, "%s_%s_cfm.png" % (arch, ds)), conf=a.conf)
            if arch == "detr":
                lg = os.path.join(logs, "train_detr_%s.log" % ds)
                if os.path.exists(lg):
                    plot_loss_detr(lg, os.path.join(a.out, "detr_%s_loss.png" % ds), "DETR %s" % ds)
            else:
                rc = os.path.join(a.runs, "%s_%s" % (arch, ds), "results.csv")
                if os.path.exists(rc):
                    plot_loss_ultra(rc, os.path.join(a.out, "%s_%s_loss.png" % (arch, ds)), "%s %s" % (arch, ds))


if __name__ == "__main__":
    main()
