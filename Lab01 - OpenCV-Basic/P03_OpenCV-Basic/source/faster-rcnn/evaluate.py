"""Evaluate a Faster R-CNN checkpoint with the shared COCO metrics + FPS/params."""
import argparse
import os
import sys
import time

import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.roboflow_to_coco import convert
from common.coco_metrics import evaluate_coco, per_class_ap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ChessDetection, collate_fn
from train import build_model


def get_args():
    p = argparse.ArgumentParser(description="COCO evaluation for Faster R-CNN")
    p.add_argument("-weights", required=True, help="checkpoint .pt")
    p.add_argument("-split-dir", dest="split_dir", required=True, help="data/<split> folder")
    p.add_argument("-batch", type=int, default=4)
    p.add_argument("-workers", type=int, default=4)
    p.add_argument("-out-json", dest="out_json", default=None)
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    predictions = []
    n_imgs, total_t = 0, 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_t += time.perf_counter() - t0
        n_imgs += len(images)

        for target, out in zip(targets, outputs):
            img_id = int(target["image_id"].item())
            for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
                x1, y1, x2, y2 = box.tolist()
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label) - 1,        # undo background shift -> 0-indexed
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
    fps = n_imgs / total_t if total_t else 0.0
    return predictions, fps


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = build_model(ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    gt_json = convert(args.split_dir)
    ds = ChessDetection(args.split_dir, gt_json, train=False)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    print("=" * 70)
    print("Evaluating Faster R-CNN: %s" % args.weights)
    print("  split  : %s (%d images)" % (args.split_dir, len(ds)))
    print("  params : %.1f M    device: %s" % (n_params, device))
    print("=" * 70)

    predictions, fps = run(model, loader, device)
    evaluate_coco(gt_json, predictions, out_json=args.out_json, model_name="Faster R-CNN")
    print("\nInference speed: %.2f FPS  |  Params: %.1f M" % (fps, n_params))

    print("\nPer-class AP@[0.50:0.95]:")
    for name, ap in per_class_ap(gt_json, predictions).items():
        print("  %-14s %.4f" % (name, ap))


if __name__ == "__main__":
    main()
