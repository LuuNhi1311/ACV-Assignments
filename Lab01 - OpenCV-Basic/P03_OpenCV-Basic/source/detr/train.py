"""Fine-tune DETR (facebook/detr-resnet-50) on a marine YOLO dataset (-> COCO)."""
import argparse, os, sys
import torch, yaml
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.yolo_to_coco import convert, load_names, _resolve_split_dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CocoDetection, make_collate_fn


def loader(data_yaml, split, proc, batch, workers, train):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    ds = CocoDetection(_resolve_split_dir(cfg, data_yaml, split), convert(data_yaml, split), proc, train)
    pin = os.getenv("PIN_MEMORY", "true").lower() == "true"
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers,
                      collate_fn=make_collate_fn(proc), pin_memory=pin)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-data", required=True); p.add_argument("-model", default="facebook/detr-resnet-50")
    p.add_argument("-epochs", type=int, default=200); p.add_argument("-batch", type=int, default=4)
    p.add_argument("-lr", type=float, default=1e-4); p.add_argument("-lr-backbone", dest="lrb", type=float, default=1e-5)
    p.add_argument("-wd", type=float, default=1e-4); p.add_argument("-workers", type=int, default=4)
    p.add_argument("-train-split", dest="tr", default="train"); p.add_argument("-val-split", dest="va", default="val")
    p.add_argument("-out", default="checkpoints"); p.add_argument("-g", "--gpu", default="0")
    a = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    dev = torch.device("cuda" if a.gpu != "-1" and torch.cuda.is_available() else "cpu")
    names = load_names(a.data)
    proc = DetrImageProcessor.from_pretrained(a.model)
    model = DetrForObjectDetection.from_pretrained(
        a.model, num_labels=len(names), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)}).to(dev)
    tl = loader(a.data, a.tr, proc, a.batch, a.workers, True)
    vl = loader(a.data, a.va, proc, a.batch, a.workers, False)
    bb = [q for n, q in model.named_parameters() if "backbone" in n and q.requires_grad]
    hd = [q for n, q in model.named_parameters() if "backbone" not in n and q.requires_grad]
    opt = torch.optim.AdamW([{"params": hd, "lr": a.lr}, {"params": bb, "lr": a.lrb}], weight_decay=a.wd)
    os.makedirs(a.out, exist_ok=True); best = float("inf")
    print("Train DETR | classes=%d | device=%s" % (len(names), dev))
    for ep in range(1, a.epochs + 1):
        model.train(); run = 0.0
        for step, b in enumerate(tl, 1):
            lab = [{k: v.to(dev) for k, v in t.items()} for t in b["labels"]]
            loss = model(pixel_values=b["pixel_values"].to(dev), pixel_mask=b["pixel_mask"].to(dev), labels=lab).loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1); opt.step()
            run += loss.item()
            if step % 50 == 0:
                print("  ep %d %d/%d loss %.4f" % (ep, step, len(tl), run / step))
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for b in vl:
                lab = [{k: v.to(dev) for k, v in t.items()} for t in b["labels"]]
                vloss += model(pixel_values=b["pixel_values"].to(dev), pixel_mask=b["pixel_mask"].to(dev), labels=lab).loss.item()
        vloss /= max(1, len(vl))
        print("[ep %d] train %.4f val %.4f" % (ep, run / max(1, len(tl)), vloss))
        model.save_pretrained(a.out + "/last"); proc.save_pretrained(a.out + "/last")
        if vloss < best:
            best = vloss; model.save_pretrained(a.out + "/best"); proc.save_pretrained(a.out + "/best")
            print("  saved best %.4f" % best)
    print("Done. best val %.4f" % best)


if __name__ == "__main__":
    main()
