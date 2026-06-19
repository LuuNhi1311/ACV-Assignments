"""Fine-tune torchvision Faster R-CNN (ResNet-50 FPN) on Chess Pieces.

Starts from COCO-pretrained weights and replaces the box predictor head with one
sized for ``num_classes = 12 + 1`` (background). Plain SGD loop with the model's
own multi-task loss (RPN + ROI classification/regression).
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.roboflow_to_coco import convert

# `faster-rcnn` is not an importable package name (hyphen) -> import siblings directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import ChessDetection, collate_fn, load_class_names


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on Chess Pieces")
    p.add_argument("-data", dest="data_root", required=True, help="dataset root with train/ valid/")
    p.add_argument("-epochs", type=int, default=30)
    p.add_argument("-batch", type=int, default=4)
    p.add_argument("-lr", type=float, default=5e-3)
    p.add_argument("-momentum", type=float, default=0.9)
    p.add_argument("-wd", dest="weight_decay", type=float, default=5e-4)
    p.add_argument("-workers", type=int, default=4)
    p.add_argument("-out", default="checkpoints")
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    class_names = load_class_names(os.path.join(args.data_root, "train"))
    num_classes = len(class_names) + 1   # + background

    train_ann = convert(os.path.join(args.data_root, "train"))
    val_ann = convert(os.path.join(args.data_root, "valid"))
    train_ds = ChessDetection(os.path.join(args.data_root, "train"), train_ann, train=True)
    val_ds = ChessDetection(os.path.join(args.data_root, "valid"), val_ann, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    model = build_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    print("=" * 70)
    print("Training Faster R-CNN | classes=%d (+bg) | device=%s" % (len(class_names), device))
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 20 == 0:
                print("  epoch %d  step %d/%d  loss %.4f"
                      % (epoch, step, len(train_loader), running / step))
        scheduler.step()

        # validation loss (model returns losses only in train mode)
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                val_loss += sum(model(images, targets).values()).item()
        val_loss /= max(1, len(val_loader))
        print("[epoch %d] train_loss=%.4f  val_loss=%.4f"
              % (epoch, running / max(1, len(train_loader)), val_loss))

        torch.save({"model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": class_names},
                   os.path.join(args.out, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "num_classes": num_classes,
                        "class_names": class_names},
                       os.path.join(args.out, "best.pt"))
            print("  -> saved best (val_loss=%.4f)" % best_val)

    print("Done. Best val_loss=%.4f  checkpoints in %s" % (best_val, args.out))


if __name__ == "__main__":
    main()
