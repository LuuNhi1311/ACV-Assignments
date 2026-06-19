"""Fine-tune DETR (facebook/detr-resnet-50) on the Chess Pieces dataset.

Transparent manual training loop (no Trainer) so the optimisation is easy to read
and robust across ``transformers`` versions. The backbone uses a 10x smaller LR
than the transformer/heads, as in the original DETR recipe.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # source/
from common.roboflow_to_coco import convert

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # detr/
from dataset import ChessCocoDetection, make_collate_fn, load_class_names


def get_args():
    p = argparse.ArgumentParser(description="Fine-tune DETR on Chess Pieces")
    p.add_argument("-data", dest="data_root", required=True, help="dataset root with train/ valid/")
    p.add_argument("-model", default="facebook/detr-resnet-50")
    p.add_argument("-epochs", type=int, default=50)
    p.add_argument("-batch", type=int, default=4)
    p.add_argument("-lr", type=float, default=1e-4)
    p.add_argument("-lr-backbone", dest="lr_backbone", type=float, default=1e-5)
    p.add_argument("-wd", dest="weight_decay", type=float, default=1e-4)
    p.add_argument("-workers", type=int, default=4)
    p.add_argument("-out", default="checkpoints", help="output dir for checkpoints")
    p.add_argument("-g", "--gpu", type=str, default="0")
    return p.parse_args()


def build_loader(split_dir, processor, batch, workers, train):
    ann = convert(split_dir)  # writes/refreshes _annotations.coco.json
    ds = ChessCocoDetection(split_dir, ann, processor, train=train)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers,
                      collate_fn=make_collate_fn(processor), pin_memory=True)


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if (args.gpu != "-1" and torch.cuda.is_available()) else "cpu")

    class_names = load_class_names(os.path.join(args.data_root, "train"))
    n_classes = len(class_names)

    processor = DetrImageProcessor.from_pretrained(args.model)
    model = DetrForObjectDetection.from_pretrained(
        args.model,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
        id2label={i: n for i, n in enumerate(class_names)},
        label2id={n: i for i, n in enumerate(class_names)},
    ).to(device)

    train_loader = build_loader(os.path.join(args.data_root, "train"), processor,
                                args.batch, args.workers, train=True)
    val_loader = build_loader(os.path.join(args.data_root, "valid"), processor,
                              args.batch, args.workers, train=False)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": heads, "lr": args.lr},
         {"params": backbone, "lr": args.lr_backbone}],
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    print("=" * 70)
    print("Training DETR (%s) | classes=%d | device=%s" % (args.model, n_classes, device))
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running += loss.item()
            if step % 20 == 0:
                print("  epoch %d  step %d/%d  loss %.4f"
                      % (epoch, step, len(train_loader), running / step))

        # quick validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask = batch["pixel_mask"].to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
                val_loss += model(pixel_values=pixel_values, pixel_mask=pixel_mask,
                                  labels=labels).loss.item()
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
