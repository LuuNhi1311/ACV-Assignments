"""Draw YOLOv8 / RT-DETR predictions (Ultralytics saves annotated images)."""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-arch", choices=["yolov8", "rtdetr"], required=True); p.add_argument("-weights", required=True)
    p.add_argument("-source", required=True); p.add_argument("-imgsz", type=int, default=640)
    p.add_argument("-conf", type=float, default=0.25); p.add_argument("-device", default="0")
    p.add_argument("-line-width", dest="line_width", type=int, default=3, help="box thickness")
    p.add_argument("-project", default="runs"); p.add_argument("-name", default=None)
    a = p.parse_args()
    from ultralytics import YOLO, RTDETR
    model = RTDETR(a.weights) if a.arch == "rtdetr" else YOLO(a.weights)
    model.predict(source=a.source, imgsz=a.imgsz, conf=a.conf, device=a.device, save=True,
                  line_width=a.line_width, show_conf=True, show_labels=True,
                  project=a.project, name=a.name or "demo_%s" % a.arch, exist_ok=True)


if __name__ == "__main__":
    main()
