"""Normalise an extracted marine dataset into a clean data.yaml (absolute path)."""
import argparse, os
import yaml

CANDS = {"train": ["images/train", "train/images", "train"],
         "val": ["images/val", "val/images", "valid/images", "images/valid", "val", "valid"],
         "test": ["images/test", "test/images", "test"]}


def find_yaml(root):
    hits = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".yaml", ".yml")):
                try:
                    c = yaml.safe_load(open(os.path.join(dp, f), encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(c, dict) and "names" in c:
                    hits.append(os.path.join(dp, f))
    return hits


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    ys = find_yaml(os.path.abspath(a.root))
    if not ys:
        raise SystemExit("No data.yaml with 'names' under %s" % a.root)
    cfg = yaml.safe_load(open(ys[0], encoding="utf-8")); base = os.path.dirname(ys[0])
    names = cfg["names"]; names = [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)
    out = {"path": base, "names": dict(enumerate(names))}
    for sp in ("train", "val", "test"):
        rel = cfg.get(sp)
        if rel and os.path.isdir(rel if os.path.isabs(rel) else os.path.join(base, rel)):
            out[sp] = rel
        else:
            for cand in CANDS[sp]:
                if os.path.isdir(os.path.join(base, cand)):
                    out[sp] = cand; break
    if "val" not in out and "test" in out:
        out["val"] = out["test"]
    yaml.safe_dump(out, open(a.out, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print("[prepare] %d classes %s ; splits %s -> %s"
          % (len(names), names, {k: out[k] for k in ("train", "val", "test") if k in out}, a.out))


if __name__ == "__main__":
    main()
