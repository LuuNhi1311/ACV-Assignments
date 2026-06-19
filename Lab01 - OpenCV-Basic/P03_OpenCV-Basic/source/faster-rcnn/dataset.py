"""COCO-detection dataset for torchvision Faster R-CNN.

Returns ``(image_tensor, target)`` where ``target`` holds ``boxes`` (xyxy pixels)
and ``labels``. torchvision detectors reserve label ``0`` for background, so the
0-indexed COCO category ids (0..11) are shifted to ``1..12`` here; predictions are
shifted back to 0..11 in ``evaluate.py``/``demo.py`` before scoring.
"""
import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import functional as F


class ChessDetection(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, train=False):
        self.img_dir = img_dir
        self.train = train
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        img = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"] + 1)          # +1: reserve 0 for background

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
        }
        img = F.to_tensor(img)
        if self.train and torch.rand(1).item() < 0.5:        # horizontal flip augmentation
            img = F.hflip(img)
            w = img.shape[-1]
            if boxes.numel():
                target["boxes"] = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) \
                    + torch.tensor([w, 0, w, 0])
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def load_class_names(split_dir):
    with open(os.path.join(split_dir, "_classes.txt"), "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
