"""Evaluate a fine-tuned DETR checkpoint with the shared COCO metrics.

Also reports the two efficiency numbers used in the cross-model comparison:
parameter count (millions) and inference speed (FPS, single image, no batching).
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image
from pycocotools.coco import COCO
from transformers import DetrForObjectDetection, DetrImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.roboflow_to_coco import convert
from common.coco_metrics import evaluate_coco, per_class_ap


def get_args():
    p = argparse.ArgumentParser(description="COCO evaluation for DETR")
    p.add_argument("-weights", required=True, help="checkpoint dir (save_pretrained output)")
    p.add_argument("-split-dir", dest="split_dir", required=True, help="data/<split> folder")
    p.add_argument("-conf", type=float, default=0.5, help="confidence threshold for FPS/demo (eval uses 0.0)")
    p.add_argument("-out-json", dest="out_json", default=None, help="dump predictions json")
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


@torch.no_grad()
def run(model, processor, coco, img_dir, device, score_thresh=0.0):
    model.eval()
    predictions = []
    n_imgs, total_t = 0, 0.0
    for img_id in sorted(coco.imgs.keys()):
        info = coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_t += time.perf_counter() - t0
        n_imgs += 1

        target_sizes = torch.tensor([[info["height"], info["width"]]], device=device)
        res = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=score_thresh)[0]
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            x1, y1, x2, y2 = box.tolist()
            predictions.append({
                "image_id": img_id,
                "category_id": int(label),          # DETR labels are 0-indexed == COCO cat ids
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    fps = n_imgs / total_t if total_t else 0.0
    return predictions, fps


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    processor = DetrImageProcessor.from_pretrained(args.weights)
    model = DetrForObjectDetection.from_pretrained(args.weights).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    gt_json = convert(args.split_dir)
    coco = COCO(gt_json)

    print("=" * 70)
    print("Evaluating DETR: %s" % args.weights)
    print("  split  : %s (%d images)" % (args.split_dir, len(coco.imgs)))
    print("  params : %.1f M    device: %s" % (n_params, device))
    print("=" * 70)

    # threshold 0.0 -> COCO AP integrates over the full PR curve (standard protocol)
    predictions, fps = run(model, processor, coco, args.split_dir, device, score_thresh=0.0)

    metrics = evaluate_coco(gt_json, predictions, out_json=args.out_json, model_name="DETR")
    print("\nInference speed: %.2f FPS  |  Params: %.1f M" % (fps, n_params))

    print("\nPer-class AP@[0.50:0.95]:")
    for name, ap in per_class_ap(gt_json, predictions).items():
        print("  %-14s %.4f" % (name, ap))


if __name__ == "__main__":
    main()
