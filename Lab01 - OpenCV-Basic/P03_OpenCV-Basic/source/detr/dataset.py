"""COCO dataset wrapper for DETR (HuggingFace), 0-indexed labels."""
import os
import torch
from PIL import Image
from pycocotools.coco import COCO


class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, processor, train=False):
        self.img_dir, self.processor, self.train = img_dir, processor, train
        self.coco = COCO(ann_file); self.ids = sorted(self.coco.imgs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]; info = self.coco.loadImgs(iid)[0]
        img = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid))
        enc = self.processor(images=img, annotations={"image_id": iid, "annotations": anns}, return_tensors="pt")
        return enc["pixel_values"][0], enc["labels"][0]


def make_collate_fn(processor):
    def collate(batch):
        enc = processor.pad([b[0] for b in batch], return_tensors="pt")
        return {"pixel_values": enc["pixel_values"], "pixel_mask": enc["pixel_mask"], "labels": [b[1] for b in batch]}
    return collate
