"""Fine-tune a YOLOv8 or RT-DETR detector with Ultralytics.

Both models share the exact same Ultralytics API and data format (a ``data.yaml``
plus YOLO ``.txt`` labels), so one script trains either via ``--arch``.
"""
import argparse
import os


def load_model(arch, weights):
    from ultralytics import YOLO, RTDETR
    if arch == "rtdetr":
        return RTDETR(weights)
    return YOLO(weights)


DEFAULT_WEIGHTS = {"yolov8": "yolov8s.pt", "rtdetr": "rtdetr-l.pt"}


def get_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 / RT-DETR (Ultralytics)")
    p.add_argument("-arch", choices=["yolov8", "rtdetr"], required=True)
    p.add_argument("-data", required=True, help="path to data.yaml")
    p.add_argument("-weights", default=None, help="pretrained weights (default per arch)")
    p.add_argument("-epochs", type=int, default=100)
    p.add_argument("-batch", type=int, default=16)
    p.add_argument("-imgsz", type=int, default=640)
    p.add_argument("-device", default="0", help="cuda id, 'cpu', or '0,1'")
    p.add_argument("-project", default="runs", help="output root")
    p.add_argument("-name", default=None, help="run name (default <arch>_<dataset>)")
    return p.parse_args()


def main():
    args = get_args()
    weights = args.weights or DEFAULT_WEIGHTS[args.arch]
    dataset_tag = os.path.splitext(os.path.basename(args.data))[0]
    name = args.name or "%s_%s" % (args.arch, dataset_tag)

    print("=" * 70)
    print("Training %s | data=%s | weights=%s | device=%s"
          % (args.arch, args.data, weights, args.device))
    print("=" * 70)

    model = load_model(args.arch, weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=name,
        exist_ok=True,
    )
    print("Done. Best checkpoint: %s/%s/weights/best.pt" % (args.project, name))


if __name__ == "__main__":
    main()
