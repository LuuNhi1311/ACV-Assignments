"""COCO-detection dataset wrapper for DETR (HuggingFace).

Reads a COCO json (produced by :mod:`common.yolo_to_coco`) plus its image folder,
and lets ``DetrImageProcessor`` turn each (image, annotations) pair into the
``pixel_values`` / ``labels`` tensors DETR expects. Category ids are 0-indexed
(matching the YOLO ``names`` order), i.e. exactly DETR's ``class_labels`` range.
"""
import os

import torch
from PIL import Image
from pycocotools.coco import COCO


class CocoDetection(torch.utils.data.Dataset):
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
        return enc["pixel_values"][0], enc["labels"][0]


def make_collate_fn(processor):
    def collate_fn(batch):
        pixel_values = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        enc = processor.pad(pixel_values, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"],
                "pixel_mask": enc["pixel_mask"], "labels": labels}
    return collate_fn
