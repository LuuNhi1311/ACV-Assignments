"""Evaluate a DETR checkpoint with the shared COCO evaluator (+ params, FPS)."""
import argparse, os, sys, time
import torch, yaml
from PIL import Image
from pycocotools.coco import COCO
from transformers import DetrForObjectDetection, DetrImageProcessor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.yolo_to_coco import convert, _resolve_split_dir
from common.coco_metrics import evaluate_coco, per_class_ap


@torch.no_grad()
def run(model, proc, coco, img_dir, dev):
    preds, n, t = [], 0, 0.0
    for iid in sorted(coco.imgs):
        info = coco.loadImgs(iid)[0]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        inp = proc(images=img, return_tensors="pt").to(dev)
        if dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); out = model(**inp)
        if dev.type == "cuda": torch.cuda.synchronize()
        t += time.perf_counter() - t0; n += 1
        sz = torch.tensor([[info["height"], info["width"]]], device=dev)
        r = proc.post_process_object_detection(out, target_sizes=sz, threshold=0.0)[0]
        for box, s, l in zip(r["boxes"], r["scores"], r["labels"]):
            x1, y1, x2, y2 = box.tolist()
            preds.append({"image_id": iid, "category_id": int(l), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    return preds, (n / t if t else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-weights", required=True); p.add_argument("-data", required=True)
    p.add_argument("-split", default="test"); p.add_argument("-out-json", dest="out_json", default=None); p.add_argument("-g", "--gpu", default="0")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if a.gpu != "-1" and torch.cuda.is_available() else "cpu")
    proc = DetrImageProcessor.from_pretrained(a.weights)
    model = DetrForObjectDetection.from_pretrained(a.weights).to(dev).eval()
    npar = sum(x.numel() for x in model.parameters()) / 1e6
    gt = convert(a.data, a.split); coco = COCO(gt)
    img_dir = _resolve_split_dir(yaml.safe_load(open(a.data, encoding="utf-8")), a.data, a.split)
    print("Eval DETR: %s [%s] %d imgs, %.1fM params" % (a.weights, a.split, len(coco.imgs), npar))
    preds, fps = run(model, proc, coco, img_dir, dev)
    evaluate_coco(gt, preds, out_json=a.out_json, model_name="DETR")
    print("FPS %.2f | Params %.1fM" % (fps, npar))
    print("Per-class AP:")
    for k, v in per_class_ap(gt, preds).items():
        print("  %-18s %.4f" % (k, v))


if __name__ == "__main__":
    main()
