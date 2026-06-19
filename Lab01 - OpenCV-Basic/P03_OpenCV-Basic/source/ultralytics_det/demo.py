"""Draw YOLOv8 / RT-DETR predictions on a folder (or single image) via Ultralytics."""
import argparse


def load_model(arch, weights):
    from ultralytics import YOLO, RTDETR
    return RTDETR(weights) if arch == "rtdetr" else YOLO(weights)


def get_args():
    p = argparse.ArgumentParser(description="YOLOv8 / RT-DETR demo")
    p.add_argument("-arch", choices=["yolov8", "rtdetr"], required=True)
    p.add_argument("-weights", required=True)
    p.add_argument("-source", required=True, help="image file or directory")
    p.add_argument("-imgsz", type=int, default=640)
    p.add_argument("-conf", type=float, default=0.25)
    p.add_argument("-device", default="0")
    p.add_argument("-project", default="runs", help="output root")
    p.add_argument("-name", default=None)
    return p.parse_args()


def main():
    args = get_args()
    model = load_model(args.arch, args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name or ("demo_%s" % args.arch),
        exist_ok=True,
        verbose=True,
    )
    print("Saved annotated images under %s/%s" % (args.project, args.name or ("demo_%s" % args.arch)))


if __name__ == "__main__":
    main()
