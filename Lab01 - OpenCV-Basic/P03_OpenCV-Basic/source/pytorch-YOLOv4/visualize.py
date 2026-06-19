import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import Yolov4
from tool.darknet2pytorch import Darknet
from tool.yolo_layer import YoloLayer


def get_args():
    p = argparse.ArgumentParser(description="Grad-CAM + t-SNE visualization for trained YOLOv4")
    p.add_argument("--mode", choices=["all", "tsne", "gradcam"], default="all")
    p.add_argument("-weights", required=True, type=str)
    p.add_argument("-dir", dest="data_dir", required=True, type=str, help="image dir for the split")
    p.add_argument("-gt", dest="gt_label", required=True, type=str, help="annotation txt (img x1,y1,x2,y2,id ...)")
    p.add_argument("-classes", type=int, default=12)
    p.add_argument("-data", dest="data_yaml", type=str, default=None, help="data .yaml (nc overrides -classes)")
    p.add_argument("-cfg", "--cfg", dest="darknet_cfg", type=str, default=None,
                   help="darknet .cfg -> build Darknet model (for csp-x checkpoints)")
    p.add_argument("-names", type=str, default=None, help="class names file (one per line)")
    p.add_argument("-g", "--gpu", type=str, default="0")
    p.add_argument("-out", dest="out_dir", default="visualizations")
    p.add_argument("--size", type=int, default=608, help="network input size (multiple of 32)")
    p.add_argument("--crop-size", type=int, default=160, help="per-box crop size for t-SNE (multiple of 32)")
    p.add_argument("--max-boxes", type=int, default=1500, help="max GT boxes for t-SNE")
    p.add_argument("--gradcam-n", type=int, default=8, help="number of images for the Grad-CAM grid")
    return p.parse_args()


def read_yaml_nc(path):
    nc = None
    for line in open(path, "r", encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line.startswith("nc:"):
            nc = int(line.split(":", 1)[1].strip())
    return nc


def make_darknet_cfg(src_cfg, n_classes):
    with open(src_cfg, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.replace("classes=80", "classes=%d" % n_classes)
    text = text.replace("filters=255", "filters=%d" % ((n_classes + 5) * 3))
    dst = os.path.splitext(src_cfg)[0] + "_%dcls.cfg" % n_classes
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    return dst


def pick_feature_layer(model):
    if hasattr(model, "down5"):
        return model.down5
    layers = list(model.models)
    yolo_i = next((i for i, m in enumerate(layers)
                   if any(isinstance(x, YoloLayer) for x in m.modules())), len(layers))
    for i in range(yolo_i - 2, -1, -1):
        if any(isinstance(x, nn.Conv2d) for x in layers[i].modules()):
            return layers[i]
    return layers[-1]


def load_annotations(gt_label):
    items = []
    with open(gt_label, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if not parts or parts[0] == "":
                continue
            name = parts[0]
            boxes = []
            for tok in parts[1:]:
                if not tok:
                    continue
                vals = [int(float(v)) for v in tok.split(",")]
                if len(vals) == 5:
                    boxes.append(vals)
            items.append((name, np.array(boxes, dtype=np.int32) if boxes else np.zeros((0, 5), np.int32)))
    return items


def load_names(names_file, n_classes):
    if names_file and os.path.exists(names_file):
        with open(names_file, "r", encoding="utf-8") as f:
            names = [l.strip() for l in f if l.strip()]
        if len(names) >= n_classes:
            return names[:n_classes]
    return [f"class_{i}" for i in range(n_classes)]


@torch.no_grad()
def run_tsne(model, feat_layer, items, data_dir, names, device, args):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    model = model.to(device).eval()
    act = {}
    h = feat_layer.register_forward_hook(lambda m, i, o: act.__setitem__("v", o.detach()))
    feats, labels = [], []
    cs = args.crop_size
    n_boxes = sum(len(b) for _, b in items)
    stride = max(1, n_boxes // args.max_boxes)
    print(f"[t-SNE] extracting features from up to {args.max_boxes} GT crops ...")

    counter = 0
    for name, boxes in items:
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        for (x1, y1, x2, y2, cid) in boxes:
            counter += 1
            if counter % stride != 0:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (cs, cs), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(crop.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
            model(t)
            f = act["v"]
            v = f.mean(dim=(2, 3)).squeeze(0)
            feats.append(v.cpu().numpy())
            labels.append(int(cid))
    h.remove()

    if len(feats) < 5:
        print("[t-SNE] not enough boxes to plot, skipping.")
        return
    feats = np.stack(feats)
    labels = np.array(labels)
    print(f"[t-SNE] {feats.shape[0]} vectors of dim {feats.shape[1]} -> running TSNE ...")

    feats = StandardScaler().fit_transform(feats)
    if feats.shape[1] > 50:
        feats = PCA(n_components=50, random_state=0).fit_transform(feats)
    perp = min(30, max(5, feats.shape[0] // 4))
    emb = TSNE(n_components=2, init="pca", perplexity=perp, learning_rate="auto",
               random_state=0).fit_transform(feats)

    _plot_tsne(emb, labels, names, os.path.join(args.out_dir, "tsne_features.png"))


def _plot_tsne(emb, labels, names, out_path):
    n_classes = len(names)

    bold = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#525252', '#bcbd22', '#17becf', '#393b79', '#b5179e']
    if n_classes <= len(bold):
        colors = bold[:n_classes]
    else:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % 20) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(7, 7))
    for c in range(n_classes):
        m = labels == c
        if m.sum() == 0:
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=22, color=colors[c], label=names[c],
                   alpha=1.0, edgecolors="white", linewidths=0.4)
    lim = np.abs(emb).max() * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("t-SNE of CNN features (test set)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[t-SNE] saved -> {out_path}")


def _gradcam_score(outputs):
    cls_conf = outputs[1][0].max(dim=1)[0]
    k = min(64, cls_conf.numel())
    return torch.topk(cls_conf, k)[0].sum()


def choose_gradcam_layer(model, t, device):
    if not hasattr(model, "models"):
        return pick_feature_layer(model)

    layers = list(model.models)
    yolo_i = next((i for i, m in enumerate(layers)
                   if any(isinstance(x, YoloLayer) for x in m.modules())), len(layers))
    cands = [(i, layers[i]) for i in range(max(1, yolo_i // 3), yolo_i)
             if any(isinstance(x, nn.Conv2d) for x in layers[i].modules())]
    if len(cands) > 24:
        cands = cands[:: max(1, len(cands) // 24)]

    acts, grads, handles = {}, {}, []
    for i, blk in cands:
        handles.append(blk.register_forward_hook(
            lambda m, inp, o, i=i: acts.__setitem__(i, o.detach())))
        handles.append(blk.register_full_backward_hook(
            lambda m, gi, go, i=i: grads.__setitem__(i, go[0].detach())))
    model.zero_grad()
    _gradcam_score(model(t)).backward()
    for h in handles:
        h.remove()

    best, best_s = None, -1.0
    for i, blk in cands:
        g, a = grads.get(i), acts.get(i)
        if g is None or a is None:
            continue
        s = g.norm().item() * (a.shape[2] * a.shape[3]) ** 0.5
        if s > best_s:
            best_s, best = s, (i, blk)
    if best is None:
        return pick_feature_layer(model)
    print(f"[Grad-CAM] auto-selected backbone block idx {best[0]} "
          f"(act {tuple(acts[best[0]].shape)}, score {best_s:.3g})")
    return best[1]


def run_gradcam(model, feat_layer, items, data_dir, names, device, args):
    model = model.to(device).eval()
    acts, grads = {}, {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o.detach()

    def bwd_hook(_m, gi, go):
        grads["v"] = go[0].detach()

    sel = [it for it in items if len(it[1]) > 0][: args.gradcam_n]
    if not sel:
        print("[Grad-CAM] no annotated images, skipping.")
        return

    sz = args.size
    probe_img = cv2.cvtColor(cv2.imread(os.path.join(data_dir, sel[0][0])), cv2.COLOR_BGR2RGB)
    probe_in = cv2.resize(probe_img, (sz, sz), interpolation=cv2.INTER_LINEAR)
    probe_t = torch.from_numpy(probe_in.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
    target_layer = choose_gradcam_layer(model, probe_t, device)

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    overlays = []
    print(f"[Grad-CAM] computing heatmaps for {len(sel)} images ...")
    for name, boxes in sel:
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        net_in = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(net_in.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(device)
        t.requires_grad_(False)

        model.zero_grad()
        score = _gradcam_score(model(t))
        try:
            score.backward()
            g = grads["v"]
            a = acts["v"]
            w = g.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((w * a).sum(dim=1, keepdim=True))[0, 0]
        except Exception as e:
            print(f"  [warn] backward failed ({e}); using activation magnitude")
            a = acts["v"]
            cam = a[0].abs().mean(dim=0)

        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = np.power(cam, 0.7)
        cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.55 * rgb + 0.45 * heat)
        overlays.append((overlay, name, boxes))

    h1.remove(); h2.remove()
    _plot_gradcam(overlays, names, os.path.join(args.out_dir, "gradcam.png"))
    run_detection_montage(model, sel, data_dir, names, device,
                          os.path.join(args.out_dir, "detections_sample.png"))


def run_detection_montage(model, sel, data_dir, names, device, out_path, conf=0.5, nms=0.6):
    from demo import detect_image
    model = model.to(device).eval()
    drawn = []
    print(f"[detect] drawing boxes for the {len(sel)} Grad-CAM images ...")
    for name, _ in sel:
        img = cv2.imread(os.path.join(data_dir, name))
        if img is None:
            continue
        vis, nb = detect_image(model, img, names, conf, nms, cuda=(device.type == "cuda"))
        drawn.append((cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), name, nb))
    if not drawn:
        return
    n = len(drawn)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (im, name, nb) in zip(axes, drawn):
        ax.imshow(im)
        ax.set_title("%s  (%d box)" % (name[:14], nb), fontsize=8)
    fig.suptitle("YOLOv4 detections on test set (random samples)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[detect] saved -> {out_path}")


def _plot_gradcam(overlays, names, out_path):
    n = len(overlays)
    if n == 0:
        return
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (ov, name, boxes) in zip(axes, overlays):
        ax.imshow(ov)
        cls = sorted(set(int(b[4]) for b in boxes))
        title = ", ".join(names[c] for c in cls[:3]) + ("..." if len(cls) > 3 else "")
        ax.set_title(title, fontsize=9)
    fig.suptitle("Grad-CAM (backbone feature activations)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Grad-CAM] saved -> {out_path}")


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    n_classes = args.classes
    if args.data_yaml:
        nc = read_yaml_nc(args.data_yaml)
        if nc:
            n_classes = nc

    names = load_names(args.names, n_classes)
    items = load_annotations(args.gt_label)
    print(f"loaded {len(items)} images from {args.gt_label}")

    if args.darknet_cfg:
        model = Darknet(make_darknet_cfg(args.darknet_cfg, n_classes), inference=True)
    else:
        model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    feat_layer = pick_feature_layer(model)

    if args.mode in ("all", "tsne"):
        run_tsne(model, feat_layer, items, args.data_dir, names, device, args)
    if args.mode in ("all", "gradcam"):
        run_gradcam(model, feat_layer, items, args.data_dir, names, device, args)


if __name__ == "__main__":
    main()
