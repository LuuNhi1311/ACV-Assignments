"""Fine-tune DETR (facebook/detr-resnet-50) on a marine YOLO dataset.

Transparent manual training loop (no Trainer) for clarity and version-robustness.
The backbone uses a 10x smaller LR than the transformer/heads, as in DETR's recipe.
Data is read from a YOLO ``data.yaml`` and converted to COCO on the fly.
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.yolo_to_coco import convert, load_names, _resolve_split_dir

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # detr/
from dataset import CocoDetection, make_collate_fn


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune DETR on a marine YOLO dataset")
    p.add_argument("-data", required=True, help="path to data.yaml")
    p.add_argument("-model", default="facebook/detr-resnet-50")
    p.add_argument("-epochs", type=int, default=50)
    p.add_argument("-batch", type=int, default=4)
    p.add_argument("-lr", type=float, default=1e-4)
    p.add_argument("-lr-backbone", dest="lr_backbone", type=float, default=1e-5)
    p.add_argument("-wd", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("-workers", type=int, default=4)
    p.add_argument("-train-split", dest="train_split", default="train")
    p.add_argument("-val-split", dest="val_split", default="val")
    p.add_argument("-out", default="checkpoints")
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


def build_loader(data_yaml, split, processor, batch, workers, train):
    cfg = yaml.safe_load(open(data_yaml, "r", encoding="utf-8"))
    img_dir = _resolve_split_dir(cfg, data_yaml, split)
    ann = convert(data_yaml, split)
    ds = CocoDetection(img_dir, ann, processor, train=train)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers,
                      collate_fn=make_collate_fn(processor), pin_memory=True)


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    class_names = load_names(args.data)
    n_classes = len(class_names)

    processor = DetrImageProcessor.from_pretrained(args.model)
    model = DetrForObjectDetection.from_pretrained(
        args.model, num_labels=n_classes, ignore_mismatched_sizes=True,
        id2label={i: n for i, n in enumerate(class_names)},
        label2id={n: i for i, n in enumerate(class_names)},
    ).to(device)

    train_loader = build_loader(args.data, args.train_split, processor, args.batch, args.workers, True)
    val_loader = build_loader(args.data, args.val_split, processor, args.batch, args.workers, False)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": heads, "lr": args.lr}, {"params": backbone, "lr": args.lr_backbone}],
        weight_decay=args.weight_decay)

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    print("=" * 70)
    print("Training DETR | data=%s | classes=%d | device=%s" % (args.data, n_classes, device))
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
            loss = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running += loss.item()
            if step % 50 == 0:
                print("  epoch %d  step %d/%d  loss %.4f"
                      % (epoch, step, len(train_loader), running / step))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                val_loss += model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels).loss.item()
        val_loss /= max(1, len(val_loader))
        print("[epoch %d] train_loss=%.4f  val_loss=%.4f"
              % (epoch, running / max(1, len(train_loader)), val_loss))

        model.save_pretrained(os.path.join(args.out, "last"))
        processor.save_pretrained(os.path.join(args.out, "last"))
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(os.path.join(args.out, "best"))
            processor.save_pretrained(os.path.join(args.out, "best"))
            print("  -> saved best (val_loss=%.4f)" % best_val)

    print("Done. Best val_loss=%.4f  checkpoints in %s" % (best_val, args.out))


if __name__ == "__main__":
    main()
