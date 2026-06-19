"""Normalise an extracted marine-detect dataset into a clean Ultralytics data.yaml.

The downloaded zips (MegaFauna / FishInv) are in Ultralytics YOLO format and ship
their own ``data.yaml`` (with the authoritative class names + order). This script
locates that yaml inside the extracted folder, rewrites ``path`` to an absolute
path and makes sure ``train`` / ``val`` / ``test`` keys resolve, then writes a
clean copy to ``<out>`` for the training/eval scripts to use.
"""
import argparse
import os

import yaml

SPLIT_CANDIDATES = {
    "train": ["images/train", "train/images", "train"],
    "val": ["images/val", "val/images", "valid/images", "images/valid", "val", "valid"],
    "test": ["images/test", "test/images", "test"],
}


def find_source_yaml(root):
    hits = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".yaml", ".yml")):
                path = os.path.join(dirpath, f)
                try:
                    cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(cfg, dict) and "names" in cfg:
                    hits.append(path)
    return hits


def detect_split(base, split):
    for cand in SPLIT_CANDIDATES[split]:
        if os.path.isdir(os.path.join(base, cand)):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description="Normalise marine dataset -> clean data.yaml")
    ap.add_argument("root", help="extracted dataset folder (e.g. data/megafauna)")
    ap.add_argument("-o", "--out", required=True, help="output data.yaml path")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    yamls = find_source_yaml(root)
    if not yamls:
        raise SystemExit(
            "No data.yaml with 'names' found under %s.\n"
            "Inspect the extracted folder and create one manually." % root)
    src = yamls[0]
    cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
    base = os.path.dirname(os.path.abspath(src))

    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]

    out_cfg = {"path": base, "names": {i: n for i, n in enumerate(names)}}
    for split in ("train", "val", "test"):
        rel = cfg.get(split)
        if rel and os.path.isdir(os.path.join(base, rel) if not os.path.isabs(rel) else rel):
            out_cfg[split] = rel
        else:
            det = detect_split(base, split)
            if det:
                out_cfg[split] = det
    if "val" not in out_cfg and "test" in out_cfg:
        out_cfg["val"] = out_cfg["test"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, allow_unicode=True, sort_keys=False)
    print("[prepare] source yaml : %s" % src)
    print("[prepare] %d classes  : %s" % (len(names), names))
    print("[prepare] splits      : %s"
          % {k: out_cfg[k] for k in ("train", "val", "test") if k in out_cfg})
    print("[prepare] wrote       : %s" % args.out)


if __name__ == "__main__":
    main()
