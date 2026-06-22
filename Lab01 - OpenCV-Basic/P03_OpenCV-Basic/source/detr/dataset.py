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


def make_collate_fn(processor=None):
    # Pad variable-size images to the batch max + build pixel_mask. Manual (torch)
    # so it works across transformers versions (processor.pad() args changed).
    def collate(batch):
        pvs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        h = max(p.shape[1] for p in pvs); w = max(p.shape[2] for p in pvs)
        padded = pvs[0].new_zeros(len(pvs), pvs[0].shape[0], h, w)
        mask = torch.zeros(len(pvs), h, w, dtype=torch.long)
        for i, p in enumerate(pvs):
            padded[i, :, : p.shape[1], : p.shape[2]] = p
            mask[i, : p.shape[1], : p.shape[2]] = 1
        return {"pixel_values": padded, "pixel_mask": mask, "labels": labels}
    return collate
