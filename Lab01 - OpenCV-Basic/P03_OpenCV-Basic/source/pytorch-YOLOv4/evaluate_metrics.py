import argparse
import os

import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from cfg import Cfg
from dataset import Yolo_dataset
from models import Yolov4
from tool.darknet2pytorch import Darknet
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator


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


def get_args():
    p = argparse.ArgumentParser(description="COCO-metric evaluation for pytorch-YOLOv4")
    p.add_argument("-weights", required=True, type=str, help="trained checkpoint (.pt/.pth)")
    p.add_argument("-dir", dest="data_dir", required=True, type=str)
    p.add_argument("-gt", dest="gt_label", required=True, type=str)
    p.add_argument("-classes", type=int, default=12)
    p.add_argument("-data", dest="data_yaml", type=str, default=None, help="data .yaml (nc overrides -classes)")
    p.add_argument("-cfg", "--cfg", dest="darknet_cfg", type=str, default=None,
                   help="darknet .cfg -> build Darknet model (for csp-x checkpoints)")
    p.add_argument("-g", "--gpu", type=str, default="0")
    p.add_argument("-batch", type=int, default=4)
    p.add_argument("-conf", type=float, default=0.005, help="confidence threshold")
    p.add_argument("-nms", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("-max-det", dest="max_det", type=int, default=100, help="max detections/image")
    return p.parse_args()


@torch.no_grad()
def evaluate_with_nms(model, data_loader, cfg, device, conf_thresh=0.005, nms_thresh=0.5, max_det=100):

    model.eval()
    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"], bbox_fmt='coco')

    for images, targets in data_loader:
        model_input = np.concatenate([[cv2.resize(img, (cfg.w, cfg.h))] for img in images], axis=0)
        model_input = torch.from_numpy(model_input.transpose(0, 3, 1, 2)).div(255.0).to(device)
        outputs = model(model_input)            # [boxes (B,N,1,4), confs (B,N,C)]

        res = {}
        for img, target, b, c in zip(images, targets, outputs[0], outputs[1]):
            h, w = img.shape[:2]
            b = b.squeeze(1)                    # [N,4] xyxy normalized
            scores, labels = c.max(dim=1)       # [N]
            keep = scores > conf_thresh
            b, scores, labels = b[keep], scores[keep], labels[keep]
            boxes = b.clone()                   # normalized xyxy -> pixel xyxy
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
            if boxes.numel():
                k = torchvision.ops.batched_nms(boxes, scores, labels, nms_thresh)[:max_det]
                boxes, scores, labels = boxes[k], scores[k], labels[k]
            boxes_xywh = boxes.clone()          # pixel xyxy -> coco xywh
            boxes_xywh[:, 2] -= boxes_xywh[:, 0]
            boxes_xywh[:, 3] -= boxes_xywh[:, 1]
            res[target["image_id"].item()] = {
                "boxes": boxes_xywh.unsqueeze(1).cpu(),
                "scores": scores.cpu(),
                "labels": labels.cpu(),
            }
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    n_classes = args.classes
    if args.data_yaml:
        nc = read_yaml_nc(args.data_yaml)
        if nc:
            n_classes = nc

    cfg = Cfg
    cfg.dataset_dir = args.data_dir
    cfg.classes = n_classes

    dataset = Yolo_dataset(args.gt_label, cfg, train=False)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=4, pin_memory=True, drop_last=False,
                        collate_fn=val_collate)

    if args.darknet_cfg:
        model = Darknet(make_darknet_cfg(args.darknet_cfg, n_classes), inference=True)
    else:
        model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print("=" * 70)
    print("Evaluating: %s" % args.weights)
    print("  images : %s" % args.data_dir)
    print("  labels : %s  (%d images)" % (args.gt_label, len(dataset)))
    print("  device : %s" % device)
    print("=" * 70)

    evaluator = evaluate_with_nms(model, loader, cfg, device,
                                  conf_thresh=args.conf, nms_thresh=args.nms, max_det=args.max_det)
    stats = evaluator.coco_eval["bbox"].stats
    names = [
        "AP @[IoU=0.50:0.95 | all   ]",
        "AP @[IoU=0.50      | all   ]",
        "AP @[IoU=0.75      | all   ]",
        "AP @[IoU=0.50:0.95 | small ]",
        "AP @[IoU=0.50:0.95 | medium]",
        "AP @[IoU=0.50:0.95 | large ]",
        "AR @[IoU=0.50:0.95 | maxDet=  1]",
        "AR @[IoU=0.50:0.95 | maxDet= 10]",
        "AR @[IoU=0.50:0.95 | maxDet=100]",
        "AR @[IoU=0.50:0.95 | small ]",
        "AR @[IoU=0.50:0.95 | medium]",
        "AR @[IoU=0.50:0.95 | large ]",
    ]
    print("\n================= SUMMARY (COCO bbox) =================")
    for name, val in zip(names, stats):
        print("  %-32s = %.4f" % (name, val))
    print("======================================================")
    print("mAP(0.50:0.95) = %.4f    mAP@0.50 = %.4f" % (stats[0], stats[1]))


if __name__ == "__main__":
    main()
