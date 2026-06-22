"""Train YOLOv8 / RT-DETR (Ultralytics). Same API for both, selected by -arch."""
import argparse, os

DEFAULT_WEIGHTS = {"yolov8": "yolov8l.pt", "rtdetr": "rtdetr-l.pt"}


def disable_pin():
    # WSL: pinned-memory thread can raise 'CUDA out of memory' even with VRAM free.
    if os.getenv("PIN_MEMORY", "true").lower() != "false":
        return
    try:
        import ultralytics.data.build as b; b.PIN_MEMORY = False
    except Exception:
        pass
    try:
        import torch; torch.Tensor.pin_memory = lambda self, *a, **k: self
    except Exception:
        pass
    print("[info] pin_memory disabled")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-arch", choices=["yolov8", "rtdetr"], required=True)
    p.add_argument("-data", required=True); p.add_argument("-weights", default=None)
    p.add_argument("-epochs", type=int, default=200); p.add_argument("-batch", type=int, default=16)
    p.add_argument("-imgsz", type=int, default=640); p.add_argument("-workers", type=int, default=8)
    p.add_argument("-device", default="0"); p.add_argument("-project", default="runs"); p.add_argument("-name", default=None)
    a = p.parse_args()
    disable_pin()
    from ultralytics import YOLO, RTDETR
    w = a.weights or DEFAULT_WEIGHTS[a.arch]
    name = a.name or "%s_%s" % (a.arch, os.path.splitext(os.path.basename(a.data))[0])
    model = RTDETR(w) if a.arch == "rtdetr" else YOLO(w)
    model.train(data=a.data, epochs=a.epochs, batch=a.batch, imgsz=a.imgsz, workers=a.workers,
                device=a.device, project=a.project, name=name, exist_ok=True)
    print("Best: %s/%s/weights/best.pt" % (a.project, name))


if __name__ == "__main__":
    main()
