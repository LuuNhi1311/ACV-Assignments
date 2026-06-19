"""Draw Faster R-CNN predictions on a folder of images (or a single image)."""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model

COLORS = [(255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 56),
          (255, 56, 255), (56, 255, 255), (255, 153, 56), (153, 56, 255),
          (56, 153, 255), (153, 255, 56), (255, 56, 153), (56, 255, 153)]


def get_args():
    p = argparse.ArgumentParser(description="Faster R-CNN bounding-box demo")
    p.add_argument("-weights", required=True, help="checkpoint .pt")
    p.add_argument("-imgfile", required=True, help="image file or directory")
    p.add_argument("-names", required=True, help="_classes.txt for label text")
    p.add_argument("-outdir", default="visualizations")
    p.add_argument("-conf", type=float, default=0.5)
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    names = [ln.strip() for ln in open(args.names, encoding="utf-8") if ln.strip()]
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = build_model(ckpt["num_classes"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    if os.path.isdir(args.imgfile):
        files = sorted(glob.glob(os.path.join(args.imgfile, "*.jpg")))
    else:
        files = [args.imgfile]
    os.makedirs(args.outdir, exist_ok=True)

    for path in files:
        img = Image.open(path).convert("RGB")
        tensor = F.to_tensor(img).to(device)
        out = model([tensor])[0]

        canvas = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        n = 0
        for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
            if float(score) < args.conf:
                continue
            n += 1
            x1, y1, x2, y2 = (int(v) for v in box.tolist())
            cls = int(label) - 1                     # undo background shift
            c = COLORS[cls % len(COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 2)
            txt = "%s %.2f" % (names[cls], float(score))
            cv2.putText(canvas, txt, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

        outp = os.path.join(args.outdir, "fasterrcnn_" + os.path.basename(path))
        cv2.imwrite(outp, canvas)
        print("saved %s (%d boxes)" % (outp, n))


if __name__ == "__main__":
    main()
