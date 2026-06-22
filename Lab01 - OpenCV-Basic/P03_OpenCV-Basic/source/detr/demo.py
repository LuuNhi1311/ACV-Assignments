"""Draw DETR predictions on an image or folder."""
import argparse, glob, os
import cv2, numpy as np, torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

COLORS = [(255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 255, 56), (255, 56, 255), (56, 255, 255),
          (255, 153, 56), (153, 56, 255), (56, 153, 255), (153, 255, 56), (255, 56, 153), (56, 255, 153)]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-weights", required=True); p.add_argument("-source", required=True)
    p.add_argument("-outdir", default="runs/demo_detr"); p.add_argument("-conf", type=float, default=0.5); p.add_argument("-g", "--gpu", default="0")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if a.gpu != "-1" and torch.cuda.is_available() else "cpu")
    proc = DetrImageProcessor.from_pretrained(a.weights)
    model = DetrForObjectDetection.from_pretrained(a.weights).to(dev).eval()
    id2label = model.config.id2label
    files = sorted(glob.glob(os.path.join(a.source, "*.jpg")) + glob.glob(os.path.join(a.source, "*.png"))) if os.path.isdir(a.source) else [a.source]
    os.makedirs(a.outdir, exist_ok=True)
    for path in files:
        img = Image.open(path).convert("RGB")
        out = model(**proc(images=img, return_tensors="pt").to(dev))
        sz = torch.tensor([[img.height, img.width]], device=dev)
        r = proc.post_process_object_detection(out, target_sizes=sz, threshold=a.conf)[0]
        canvas = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h, w = canvas.shape[:2]
        lw = max(2, round((h + w) / 700))                 # line width scales with image
        fs = max(0.5, lw * 0.35)                           # font scale
        ft = max(1, lw - 1)                                # font thickness
        for box, s, l in zip(r["boxes"], r["scores"], r["labels"]):
            x1, y1, x2, y2 = (int(v) for v in box.tolist()); c = COLORS[int(l) % len(COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), c, lw, cv2.LINE_AA)
            txt = "%s %.2f" % (id2label[int(l)], float(s))
            (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
            ytop = max(0, y1 - th - bl - 4)
            cv2.rectangle(canvas, (x1, ytop), (x1 + tw + 4, y1), c, -1)   # filled label bg
            cv2.putText(canvas, txt, (x1 + 2, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, (255, 255, 255), ft, cv2.LINE_AA)
        outp = os.path.join(a.outdir, "detr_" + os.path.basename(path)); cv2.imwrite(outp, canvas)
        print("saved %s (%d)" % (outp, len(r["scores"])))


if __name__ == "__main__":
    main()
