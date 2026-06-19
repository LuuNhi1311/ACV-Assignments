"""COCO-detection dataset wrapper for DETR (HuggingFace).

Reads the COCO json produced by :mod:`common.roboflow_to_coco` and lets the
``DetrImageProcessor`` turn each (image, annotations) pair into the
``pixel_values`` / ``labels`` tensors DETR expects. Category ids are 0-indexed
(0..11), which is exactly DETR's ``class_labels`` range for ``num_labels=12``.
"""
import os

import torch
from PIL import Image
from pycocotools.coco import COCO


class ChessCocoDetection(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, processor, train=False):
        self.img_dir = img_dir
        self.processor = processor
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
        target = {"image_id": img_id, "annotations": anns}

        enc = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = enc["pixel_values"][0]
        labels = enc["labels"][0]
        return pixel_values, labels


def make_collate_fn(processor):
    def collate_fn(batch):
        pixel_values = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }
    return collate_fn


def load_class_names(split_dir):
    with open(os.path.join(split_dir, "_classes.txt"), "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]
