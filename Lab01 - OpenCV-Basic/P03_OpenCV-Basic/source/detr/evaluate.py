"""Evaluate a fine-tuned DETR checkpoint with the shared COCO evaluator + FPS/params."""
import argparse
import os
import sys
import time

import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from transformers import DetrForObjectDetection, DetrImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.yolo_to_coco import convert, _resolve_split_dir
from common.coco_metrics import evaluate_coco, per_class_ap


def get_args():
    p = argparse.ArgumentParser(description="COCO evaluation for DETR")
    p.add_argument("-weights", required=True, help="checkpoint dir (save_pretrained output)")
    p.add_argument("-data", required=True, help="data.yaml")
    p.add_argument("-split", default="test")
    p.add_argument("-out-json", dest="out_json", default=None)
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
        res = processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                      threshold=score_thresh)[0]
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            x1, y1, x2, y2 = box.tolist()
            predictions.append({
                "image_id": img_id, "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score),
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

    gt_json = convert(args.data, args.split)
    coco = COCO(gt_json)
    cfg = yaml.safe_load(open(args.data, "r", encoding="utf-8"))
    img_dir = _resolve_split_dir(cfg, args.data, args.split)

    print("=" * 70)
    print("Evaluating DETR: %s" % args.weights)
    print("  data   : %s [%s]  (%d images)" % (args.data, args.split, len(coco.imgs)))
    print("  params : %.1f M    device: %s" % (n_params, device))
    print("=" * 70)

    predictions, fps = run(model, processor, coco, img_dir, device, score_thresh=0.0)
    evaluate_coco(gt_json, predictions, out_json=args.out_json, model_name="DETR")
    print("\nInference speed: %.2f FPS  |  Params: %.1f M" % (fps, n_params))

    print("\nPer-class AP@[0.50:0.95]:")
    for name, ap in per_class_ap(gt_json, predictions).items():
        print("  %-18s %.4f" % (name, ap))


if __name__ == "__main__":
    main()
