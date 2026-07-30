"""Shared helpers for the Brain MRI Segmentation seminar."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- data
def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["images"], z["masks"], z["patient_ids"], z["slice_index"]

def split_by_slice(n, val_frac=0.2, seed=0):
    """The TEMPTING split. Shuffles individual slices. Leaks."""
    idx = np.arange(n); np.random.default_rng(seed).shuffle(idx)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:]

def split_by_patient(patient_ids, val_frac=0.2, seed=0):
    """The CORRECT split. A patient is entirely in train or entirely in val."""
    pats = np.unique(patient_ids)
    rng = np.random.default_rng(seed); rng.shuffle(pats)
    cut = int(len(pats) * (1 - val_frac))
    tr_p, va_p = set(pats[:cut]), set(pats[cut:])
    tr = np.array([i for i, p in enumerate(patient_ids) if p in tr_p])
    va = np.array([i for i, p in enumerate(patient_ids) if p in va_p])
    return tr, va

class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, indices, augment=False):
        self.x, self.y, self.idx, self.aug = images, masks, indices, augment
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        img = self.x[j].astype(np.float32) / 255.0      # HWC
        msk = self.y[j].astype(np.float32)              # HW
        if self.aug:
            if np.random.rand() < 0.5: img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5: img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)
            if k: img, msk = np.rot90(img, k, (0, 1)), np.rot90(msk, k, (0, 1))
        img = np.ascontiguousarray(img.transpose(2, 0, 1))   # CHW
        msk = np.ascontiguousarray(msk)[None]                # 1HW
        return torch.from_numpy(img), torch.from_numpy(msk)

# ---------------------------------------------------------------- metric
def dice_score(pred, target, eps=1e-7):
    """Dice for binary arrays. If BOTH are empty -> 1.0 (nothing there, model agreed)."""
    pred = np.asarray(pred).astype(bool).ravel()
    target = np.asarray(target).astype(bool).ravel()
    inter = np.logical_and(pred, target).sum()
    s = pred.sum() + target.sum()
    if s == 0: return 1.0
    return float((2 * inter + eps) / (s + eps))

def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()

def combo_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)

# ---------------------------------------------------------------- model
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.b(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, f=16):
        super().__init__()
        self.d1, self.d2, self.d3, self.d4 = DoubleConv(in_ch,f), DoubleConv(f,f*2), DoubleConv(f*2,f*4), DoubleConv(f*4,f*8)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(f*8, f*16)
        self.u4 = nn.ConvTranspose2d(f*16, f*8, 2, 2); self.c4 = DoubleConv(f*16, f*8)
        self.u3 = nn.ConvTranspose2d(f*8,  f*4, 2, 2); self.c3 = DoubleConv(f*8,  f*4)
        self.u2 = nn.ConvTranspose2d(f*4,  f*2, 2, 2); self.c2 = DoubleConv(f*4,  f*2)
        self.u1 = nn.ConvTranspose2d(f*2,  f,   2, 2); self.c1 = DoubleConv(f*2,  f)
        self.out = nn.Conv2d(f, out_ch, 1)
    def forward(self, x):
        s1 = self.d1(x);            s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2));s4 = self.d4(self.pool(s3))
        b  = self.bott(self.pool(s4))
        x = self.c4(torch.cat([self.u4(b),  s4], 1))
        x = self.c3(torch.cat([self.u3(x),  s3], 1))
        x = self.c2(torch.cat([self.u2(x),  s2], 1))
        x = self.c1(torch.cat([self.u1(x),  s1], 1))
        return self.out(x)

# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict_all(model, images, indices, bs=32, thr=0.5):
    model.eval(); outs = []
    for k in range(0, len(indices), bs):
        chunk = indices[k:k+bs]
        x = torch.from_numpy(images[chunk].astype(np.float32).transpose(0,3,1,2)/255.).to(DEVICE)
        outs.append((torch.sigmoid(model(x)).cpu().numpy()[:,0] > thr).astype(np.uint8))
    return np.concatenate(outs)

def per_patient_dice(pred, true, pids):
    """Buda-style: pool every pixel of a patient, then one Dice per patient."""
    return {p: dice_score(pred[pids == p], true[pids == p]) for p in np.unique(pids)}

# ---------------------------------------------------------------- capstone
def shape_features(mask2d):
    """Shape descriptors of the largest connected tumor component in one slice."""
    from skimage import measure
    lab = measure.label(mask2d.astype(np.uint8))
    if lab.max() == 0: return None
    props = max(measure.regionprops(lab), key=lambda r: r.area)
    per = props.perimeter if props.perimeter > 0 else 1e-6
    return dict(area=props.area, perimeter=props.perimeter,
                circularity=4*np.pi*props.area/(per**2),
                eccentricity=props.eccentricity, extent=props.extent,
                solidity=props.solidity,
                major_axis=_axis(props, "major"), minor_axis=_axis(props, "minor"))

def _axis(props, which):
    # skimage renamed these in 0.26; support both
    for name in (f"axis_{which}_length", f"{which}_axis_length"):
        if hasattr(props, name): return getattr(props, name)
    return float("nan")

def patient_shape_features(masks, pids):
    """One feature row per patient, taken from that patient's largest tumor slice."""
    rows = []
    for p in np.unique(pids):
        sel = np.where(pids == p)[0]
        areas = masks[sel].reshape(len(sel), -1).sum(1)
        if areas.max() == 0: continue
        f = shape_features(masks[sel[areas.argmax()]])
        if f: rows.append(dict(patient=p, **f))
    return rows

# ---------------------------------------------------------------- data fetch
DATA_URL = ""   # instructor: put a direct-download URL to lgg_128.npz here

def get_data(path="lgg_128.npz", url=None, allow_demo=True):
    """Return (images, masks, patient_ids, slice_index).
    Order: local file -> DATA_URL -> kagglehub -> synthetic demo."""
    import os, urllib.request
    if os.path.exists(path):
        print(f"Using local {path}"); return load_npz(path)
    url = url or DATA_URL
    if url:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, path)
            return load_npz(path)
        except Exception as e:
            print(f"  download failed: {e}")
    try:
        import kagglehub
        print("Trying Kaggle (needs an account + token)...")
        src = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
        for root, dirs, _ in os.walk(src):
            if os.path.basename(root) == "kaggle_3m" or any(d.startswith("TCGA_") for d in dirs):
                build_npz(root, path); return load_npz(path)
    except Exception as e:
        print(f"  kaggle failed: {e}")
    if not allow_demo:
        raise RuntimeError("Could not obtain the dataset.")
    print("\n" + "!"*70 + "\n!! DEMO MODE: using SYNTHETIC data, not real patients.\n"
          "!! Numbers here are NOT meaningful. Ask your instructor for the real URL.\n" + "!"*70 + "\n")
    return make_demo_data()

def build_npz(src, out="lgg_128.npz", size=128):
    """Convert a raw kaggle_3m folder into the compact npz."""
    import glob, os, re as _re
    from PIL import Image
    imgs, msks, pids, sidx = [], [], [], []
    num = lambda p: int(_re.search(r"_(\d+)_mask\.tif$", os.path.basename(p)).group(1))
    for d in sorted(x for x in glob.glob(os.path.join(src, "*")) if os.path.isdir(x)):
        for mp in sorted(glob.glob(os.path.join(d, "*_mask.tif")), key=num):
            ip = mp.replace("_mask.tif", ".tif")
            if not os.path.exists(ip): continue
            imgs.append(np.asarray(Image.open(ip).convert("RGB").resize((size,size), Image.BILINEAR), np.uint8))
            msks.append((np.asarray(Image.open(mp).convert("L").resize((size,size), Image.NEAREST), np.uint8)>127).astype(np.uint8))
            pids.append(os.path.basename(d)); sidx.append(num(mp))
    np.savez_compressed(out, images=np.stack(imgs), masks=np.stack(msks),
                        patient_ids=np.array(pids), slice_index=np.array(sidx, np.int16))
    return out

def make_demo_data(n_patients=40, size=128, seed=0):
    """Synthetic stand-in. Deliberately messy: bright skull, uneven tumor
    intensity, noise -- so thresholding struggles like it does on real MRI."""
    rng = np.random.default_rng(seed)
    I, M, P, S = [], [], [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for p in range(n_patients):
        pid = f"DEMO_{p:03d}"
        n = rng.integers(20, 30); start = rng.integers(3, 10); run = rng.integers(6, 14)
        cx, cy = rng.integers(40,88), rng.integers(40,88); rad = rng.integers(5, 16)
        for s in range(n):
            r2 = ((xx-size/2)/(size*0.37))**2 + ((yy-size/2)/(size*0.43))**2
            brain, skull = r2 < 1.0, (r2 > 0.92) & (r2 < 1.12)
            img = np.zeros((size,size,3), np.float32)
            img[...,1] = brain*(70+40*rng.random()) + skull*(150+60*rng.random())
            img[...,0] = brain*(60+30*rng.random()) + skull*(120+50*rng.random())
            img[...,2] = brain*(65+35*rng.random()) + skull*(130+40*rng.random())
            img[...,1] += brain*rng.normal(0,18,(size,size))   # tissue texture
            mask = np.zeros((size,size), np.uint8)
            if start <= s < start+run:
                t = (s-start)/max(1,run-1); r = rad*np.sin(np.pi*max(t,0.08))
                wob = 1 + 0.3*np.sin(4*np.arctan2(yy-cy, xx-cx) + p)
                blob = ((xx-cx)**2+(yy-cy)**2) < (r*wob)**2
                blob &= brain
                if blob.sum() > 4:
                    mask[blob] = 1
                    img[blob,1] += rng.normal(55, 22, blob.sum())   # overlaps skull range
            img += rng.normal(0, 9, img.shape)
            I.append(np.clip(img,0,255).astype(np.uint8)); M.append(mask); P.append(pid); S.append(s)
    return np.stack(I), np.stack(M), np.array(P), np.array(S, np.int16)

META_URL = ""   # instructor: direct-download URL to the dataset's data.csv

def get_meta(path="lgg_meta.csv", url=None):
    """Per-patient clinical/genomic table (the dataset's data.csv). For capstone track C."""
    import os, urllib.request, pandas as pd
    if not os.path.exists(path):
        url = url or META_URL
        if url:
            try: urllib.request.urlretrieve(url, path)
            except Exception as e: print("meta download failed:", e)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "Patient" in df.columns: df = df.rename(columns={"Patient": "patient_short"})
        return df
    print("!! DEMO MODE: synthetic metadata, correlations here are random by construction.")
    rng = np.random.default_rng(1)
    n = 40
    return pd.DataFrame({"patient_short": [f"DEMO_{i:03d}" for i in range(n)],
                         "RNASeqCluster": rng.integers(1,5,n),
                         "MethylationCluster": rng.integers(1,6,n),
                         "death01": rng.integers(0,2,n),
                         "age_at_initial_pathologic": rng.integers(20,80,n)})

def short_id(pid):
    """TCGA_CS_4941_19960909 -> TCGA_CS_4941   (matches the data.csv Patient column)"""
    parts = str(pid).split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else str(pid)
