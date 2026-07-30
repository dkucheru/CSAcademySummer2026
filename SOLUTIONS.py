"""
INSTRUCTOR SOLUTION KEY — every TODO in notebooks 01-05, filled in.
Do not give this to students. Run it to verify your setup works.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, collections
from seminar import *

# ============ SESSION 1 ============
def session1(images, masks, patient_ids):
    n_patients = len(np.unique(patient_ids))                       # Q1
    n_slices   = len(images)
    has_tumour = masks.reshape(len(masks), -1).sum(1) > 0          # Q2
    tumour_pixel_fraction = masks.mean()                           # Q3
    areas = masks.reshape(len(masks), -1).sum(1)                   # Q4
    tiny = int(((areas > 0) & (areas < 20)).sum())
    institutions = [str(p).split("_")[1] for p in patient_ids]     # Q5
    return dict(n_patients=n_patients, n_slices=n_slices,
                pct_slices_tumour=has_tumour.mean(),
                pct_pixels_tumour=tumour_pixel_fraction,
                tiny_tumour_slices=tiny,
                institutions=collections.Counter(institutions))

# ============ SESSION 2 ============
def my_dice(pred, true):                                            # Q2/Q3
    pred = np.asarray(pred).astype(bool); true = np.asarray(true).astype(bool)
    overlap = np.logical_and(pred, true).sum()
    total   = pred.sum() + true.sum()
    return 1.0 if total == 0 else 2 * overlap / total

def my_segmenter(flair_stack, thresh=None):                         # Q4
    """Threshold + skull removal + small-blob removal + per-slice normalisation."""
    from scipy import ndimage
    from skimage import measure
    s = flair_stack
    # per-slice z-normalisation, so a fixed cutoff means the same thing everywhere
    mu = s.reshape(len(s), -1).mean(1)[:, None, None]
    sd = s.reshape(len(s), -1).std(1)[:, None, None] + 1e-6
    z = (s - mu) / sd
    pred = z > (1.6 if thresh is None else thresh)
    # kill the skull: ignore everything outside a central ellipse
    H, W = s.shape[1:]
    yy, xx = np.mgrid[0:H, 0:W]
    inside = (((xx - W/2)/(W*0.33))**2 + ((yy - H/2)/(H*0.39))**2) < 1.0
    pred &= inside
    # remove blobs smaller than 15 px
    out = np.zeros_like(pred)
    for k in range(len(pred)):
        if not pred[k].any(): continue
        lab = measure.label(pred[k])
        for r in measure.regionprops(lab):
            if r.area >= 15: out[k][lab == r.label] = True
    return out

# ============ SESSION 3 ============
class DoubleConv(nn.Module):                                        # Q1
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.block(x)

class MyUNet(nn.Module):                                            # Q2
    def __init__(self, in_ch=3, out_ch=1, f=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.d1, self.d2 = DoubleConv(in_ch, f), DoubleConv(f, f*2)
        self.d3, self.d4 = DoubleConv(f*2, f*4), DoubleConv(f*4, f*8)
        self.bottleneck  = DoubleConv(f*8, f*16)
        self.u4, self.c4 = nn.ConvTranspose2d(f*16, f*8, 2, 2), DoubleConv(f*16, f*8)
        self.u3, self.c3 = nn.ConvTranspose2d(f*8,  f*4, 2, 2), DoubleConv(f*8,  f*4)
        self.u2, self.c2 = nn.ConvTranspose2d(f*4,  f*2, 2, 2), DoubleConv(f*4,  f*2)
        self.u1, self.c1 = nn.ConvTranspose2d(f*2,  f,   2, 2), DoubleConv(f*2,  f)
        self.out = nn.Conv2d(f, out_ch, 1)
    def forward(self, x):
        s1 = self.d1(x); s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2)); s4 = self.d4(self.pool(s3))
        b  = self.bottleneck(self.pool(s4))
        x = self.c4(torch.cat([self.u4(b), s4], dim=1))
        x = self.c3(torch.cat([self.u3(x), s3], dim=1))
        x = self.c2(torch.cat([self.u2(x), s2], dim=1))
        x = self.c1(torch.cat([self.u1(x), s1], dim=1))
        return self.out(x)

def my_soft_dice_loss(logits, target, eps=1.0):                     # Q3
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1,2,3)) + eps
    den = p.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (1 - num/den).mean()

def my_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + my_soft_dice_loss(logits, target)
