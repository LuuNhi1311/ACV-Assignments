"""Evaluate YOLOv8 / RT-DETR with the shared COCO evaluator (+ params, FPS)."""
import argparse, os, sys, time
import torch, yaml
from pycocotools.coco import COCO
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.yolo_to_coco import convert, _resolve_split_dir
from common.coco_metrics import evaluate_coco, per_class_ap


@torch.no_grad()
def run(model, img_dir, coco, device, imgsz, conf):
    preds, n, t = [], 0, 0.0
    for im in coco.dataset["images"]:
        t0 = time.perf_counter()
        r = model.predict(os.path.join(img_dir, im["file_name"]), imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        t += time.perf_counter() - t0; n += 1
        b = r.boxes
        if b is None or b.shape[0] == 0:
            continue
        for (x1, y1, x2, y2), s, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
            preds.append({"image_id": im["id"], "category_id": int(c),
                          "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)})
    return preds, (n / t if t else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-arch", choices=["yolov8", "rtdetr"], required=True); p.add_argument("-weights", required=True)
    p.add_argument("-data", required=True); p.add_argument("-split", default="test"); p.add_argument("-imgsz", type=int, default=640)
    p.add_argument("-conf", type=float, default=0.001); p.add_argument("-device", default="0")
    p.add_argument("-out-json", dest="out_json", default=None)
    a = p.parse_args()
    from ultralytics import YOLO, RTDETR
    model = RTDETR(a.weights) if a.arch == "rtdetr" else YOLO(a.weights)
    npar = sum(x.numel() for x in model.model.parameters()) / 1e6
    gt = convert(a.data, a.split); coco = COCO(gt)
    img_dir = _resolve_split_dir(yaml.safe_load(open(a.data, encoding="utf-8")), a.data, a.split)
    print("Eval %s: %s [%s] %d imgs, %.1fM params" % (a.arch, a.weights, a.split, len(coco.imgs), npar))
    preds, fps = run(model, img_dir, coco, a.device, a.imgsz, a.conf)
    evaluate_coco(gt, preds, out_json=a.out_json, model_name=a.arch)
    print("FPS %.2f | Params %.1fM" % (fps, npar))
    print("Per-class AP:")
    for k, v in per_class_ap(gt, preds).items():
        print("  %-18s %.4f" % (k, v))


if __name__ == "__main__":
    main()
