import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DATASET_PY = os.path.join(REPO, "dataset.py")
CFG_PY = os.path.join(REPO, "cfg.py")
UTILS_PY = os.path.join(REPO, "tool", "utils.py")

NEW_GET_IMAGE_ID = '''def get_image_id(filename:str) -> int:
    # PATCHED_IMAGE_ID
    import hashlib, os
    base = os.path.basename(filename)
    return int(hashlib.md5(base.encode("utf-8")).hexdigest()[:15], 16)
'''


def patch_dataset():
    with open(DATASET_PY, "r", encoding="utf-8") as f:
        text = f.read()
    if "PATCHED_IMAGE_ID" in text:
        return "dataset.py get_image_id: already patched"
    pattern = re.compile(
        r"def get_image_id\(filename:str\) -> int:.*?\n    return id\n",
        re.DOTALL,
    )
    new_text, n = pattern.subn(NEW_GET_IMAGE_ID, text)
    if n != 1:
        raise RuntimeError("could not locate get_image_id() to patch in dataset.py")
    with open(DATASET_PY, "w", encoding="utf-8") as f:
        f.write(new_text)
    return "dataset.py get_image_id: PATCHED"


def patch_cfg():
    with open(CFG_PY, "r", encoding="utf-8") as f:
        text = f.read()
    changes = []

    new_text, n = re.subn(
        r"Cfg\.use_darknet_cfg\s*=\s*True",
        "Cfg.use_darknet_cfg = False",
        text,
    )
    if n:
        changes.append("use_darknet_cfg -> False")
    text = new_text

    epochs = os.environ.get("EPOCHS")
    if epochs:
        text, n = re.subn(r"Cfg\.TRAIN_EPOCHS\s*=\s*\d+",
                          "Cfg.TRAIN_EPOCHS = %d" % int(epochs), text)
        if n:
            changes.append("TRAIN_EPOCHS -> %s" % epochs)

    batch = os.environ.get("BATCH")
    if batch:
        text, n = re.subn(r"(?m)^Cfg\.batch\s*=\s*\d+",
                          "Cfg.batch = %d" % int(batch), text)
        if n:
            changes.append("batch -> %s" % batch)

    subdiv = os.environ.get("SUBDIV")
    if subdiv:
        text, n = re.subn(r"(?m)^Cfg\.subdivisions\s*=\s*\d+",
                          "Cfg.subdivisions = %d" % int(subdiv), text)
        if n:
            changes.append("subdivisions -> %s" % subdiv)

    burn_in = os.environ.get("BURN_IN")
    if burn_in:
        text, n = re.subn(r"(?m)^Cfg\.burn_in\s*=\s*\d+",
                          "Cfg.burn_in = %d" % int(burn_in), text)
        if n:
            changes.append("burn_in -> %s" % burn_in)

    with open(CFG_PY, "w", encoding="utf-8") as f:
        f.write(text)
    return "cfg.py: " + (", ".join(changes) if changes else "no change needed")


def patch_numpy():


    with open(DATASET_PY, "r", encoding="utf-8") as f:
        text = f.read()
    repl = {"dtype=np.float)": "dtype=np.float64)",
            "dtype=np.int)": "dtype=np.int64)"}
    n = 0
    for old, new in repl.items():
        c = text.count(old)
        if c:
            text = text.replace(old, new)
            n += c
    with open(DATASET_PY, "w", encoding="utf-8") as f:
        f.write(text)
    return "dataset.py numpy aliases: %s" % ("patched %d" % n if n else "already patched")


def patch_utils():
    with open(UTILS_PY, "r", encoding="utf-8") as f:
        text = f.read()
    if "np.float32(c3[0])" not in text and "np.float32(c1[1] - 2)" not in text:
        return "tool/utils.py plot_boxes_cv2: already patched"
    text = text.replace(
        "c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)",
        "c3 = (int(c1[0] + t_size[0]), int(c1[1] - t_size[1] - 3))")
    text = text.replace(
        "cv2.rectangle(img, (x1,y1), (np.float32(c3[0]), np.float32(c3[1])), rgb, -1)",
        "cv2.rectangle(img, (x1,y1), (c3[0], c3[1]), rgb, -1)")
    text = text.replace(
        "cv2.putText(img, msg, (c1[0], np.float32(c1[1] - 2))",
        "cv2.putText(img, msg, (c1[0], int(c1[1] - 2))")
    with open(UTILS_PY, "w", encoding="utf-8") as f:
        f.write(text)
    return "tool/utils.py plot_boxes_cv2: PATCHED"


if __name__ == "__main__":
    print("[patch_repo] " + patch_dataset())
    print("[patch_repo] " + patch_numpy())
    print("[patch_repo] " + patch_cfg())
    print("[patch_repo] " + patch_utils())
    sys.exit(0)
