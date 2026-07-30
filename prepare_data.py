"""
Instructor-side, run ONCE. Converts the raw Kaggle LGG download into a single
compact .npz that students can fetch with no Kaggle account and no auth.

Usage:  python prepare_data.py --src path/to/kaggle_3m --out lgg_128.npz --size 128
"""
import argparse, os, glob, re
import numpy as np
from PIL import Image

def build(src, out, size):
    pat_dirs = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d))
    if not pat_dirs:
        raise SystemExit(f"No patient folders found in {src}")
    imgs, msks, pids, sidx = [], [], [], []
    for d in pat_dirs:
        pid = os.path.basename(d)
        slices = glob.glob(os.path.join(d, "*_mask.tif"))
        # sort by the slice number embedded in the filename, not lexically
        def num(p):
            m = re.search(r"_(\d+)_mask\.tif$", os.path.basename(p))
            return int(m.group(1)) if m else -1
        slices.sort(key=num)
        for mp in slices:
            ip = mp.replace("_mask.tif", ".tif")
            if not os.path.exists(ip):
                continue
            im = Image.open(ip).convert("RGB").resize((size, size), Image.BILINEAR)
            mk = Image.open(mp).convert("L").resize((size, size), Image.NEAREST)
            imgs.append(np.asarray(im, np.uint8))
            msks.append((np.asarray(mk, np.uint8) > 127).astype(np.uint8))
            pids.append(pid); sidx.append(num(mp))
    imgs = np.stack(imgs); msks = np.stack(msks)
    pids = np.array(pids); sidx = np.array(sidx, np.int16)
    np.savez_compressed(out, images=imgs, masks=msks, patient_ids=pids, slice_index=sidx)
    mb = os.path.getsize(out) / 1e6
    print(f"{len(imgs)} slices from {len(set(pids))} patients -> {out} ({mb:.1f} MB)")
    print(f"  images {imgs.shape} {imgs.dtype} | masks {msks.shape} | tumor slices: "
          f"{(msks.reshape(len(msks),-1).sum(1)>0).mean():.1%}")

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--src", required=True); a.add_argument("--out", default="lgg_128.npz")
    a.add_argument("--size", type=int, default=128)
    g = a.parse_args(); build(g.src, g.out, g.size)
