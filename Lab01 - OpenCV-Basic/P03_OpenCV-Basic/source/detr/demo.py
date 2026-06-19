"""Draw DETR predictions on a folder of images (or a single image)."""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

COLORS = [(255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 56),
          (255, 56, 255), (56, 255, 255), (255, 153, 56), (153, 56, 255),
          (56, 153, 255), (153, 255, 56), (255, 56, 153), (56, 255, 153),
          (200, 120, 0), (0, 120, 200)]


def get_args():
    p = argparse.ArgumentParser(description="DETR bounding-box demo")
    p.add_argument("-weights", required=True, help="checkpoint dir")
    p.add_argument("-source", required=True, help="image file or directory")
    p.add_argument("-outdir", default="runs/demo_detr")
    p.add_argument("-conf", type=float, default=0.5)
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    processor = DetrImageProcessor.from_pretrained(args.weights)
    model = DetrForObjectDetection.from_pretrained(args.weights).to(device).eval()
    id2label = model.config.id2label

    if os.path.isdir(args.source):
        files = sorted(glob.glob(os.path.join(args.source, "*.jpg")) +
                       glob.glob(os.path.join(args.source, "*.png")))
    else:
        files = [args.source]
    os.makedirs(args.outdir, exist_ok=True)

    for path in files:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        target_sizes = torch.tensor([[img.height, img.width]], device=device)
        res = processor.post_process_object_detection(outputs, target_sizes=target_sizes,
                                                      threshold=args.conf)[0]
        canvas = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            c = COLORS[int(label) % len(COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
            txt = "%s %.2f" % (id2label[int(label)], float(score))
            cv2.putText(canvas, txt, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        out = os.path.join(args.outdir, "detr_" + os.path.basename(path))
        cv2.imwrite(out, canvas)
        print("saved %s (%d boxes)" % (out, len(res["scores"])))


if __name__ == "__main__":
    main()
