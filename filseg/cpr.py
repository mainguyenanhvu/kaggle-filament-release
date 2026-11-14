"""Curved-planar-reformation (CPR) strip refiner: the guarded second pass of the CPRVOTE_G submission.

Each mask of a finished submission is straightened along its spine into a 65-row strip, a small U-Net
(`StripNet`) predicts a residual on the mask's own logit there, and the strip probability is mapped back
to the image. Overlaps go to the higher probability and each instance keeps its largest component.
The guard ("G"): a mask is replaced only when IoU(refined, original) >= 0.7; any other mask keeps its
original pixels minus those a replaced neighbour took. Pixels outside the strip band keep their state.

  load_strip_net(path, device) -> StripNet
  refine_image(net, gray, rles, device) -> list of COCO run-length masks, same count and order

Geometry: spine sampled at 1 px arc length, extended 0.25 L (at least 24 px) past both tips along the end
tangent; half-height H = clip(3 * max half-width, 16, 64) px; columns at step H/32 (isotropic).
"""

from __future__ import annotations

import heapq

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_util
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROWS, EXT_FRAC, EXT_MIN, MAXCOL = 65, 0.25, 24, 1600
GUARD_IOU = 0.7


def _longest_path(m):
    """Geodesic double sweep inside the mask, then a medial-seeking Dijkstra between the two ends."""
    pts = np.argwhere(m)
    if len(pts) == 1:
        return pts.astype(float)
    nb = [(a, b, 1.414 if a and b else 1.0) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    dt = ndi.distance_transform_edt(m)
    h, w = m.shape

    def run(start, medial):
        dist = {start: 0.0}
        par = {start: None}
        pq = [(0.0, start)]
        far = start
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            far = u
            for a, b, step in nb:
                v = (u[0] + a, u[1] + b)
                if 0 <= v[0] < h and 0 <= v[1] < w and m[v]:
                    c = step * (1 + 25.0 / dt[v] ** 2) if medial else step
                    nd = d + c
                    if nd < dist.get(v, 1e18):
                        dist[v] = nd
                        par[v] = u
                        heapq.heappush(pq, (nd, v))
        return far, par

    a, _ = run(tuple(pts[0]), False)
    b, _ = run(a, False)
    _, par = run(a, True)
    path = []
    while b is not None:
        path.append(b)
        b = par[b]
    return np.array(path, float)


def _resample(p, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < 1e-6:
        p = p + np.outer(np.arange(len(p)), [0, 1.0])
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    t = np.arange(0, d[-1] + step * 0.5, step)
    return np.stack([np.interp(t, d, p[:, i]) for i in (0, 1)], 1)


def spine_of(mask):
    """mask: bool 2D crop -> (N x 2 smoothed spine at 1 px arc length, max half-width)."""
    m = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n > 1:  # bridge fragments through the largest component only
        m = lab == 1 + np.argmax(ndi.sum(m, lab, range(1, n + 1)))
    p = _longest_path(m)
    if len(p) < 2:
        c = np.argwhere(m).mean(0)
        p = np.array([c - [0, 2], c + [0, 2]], float)
    if len(p) >= 9:
        q = np.stack([ndi.gaussian_filter1d(p[:, i], 4, mode="nearest") for i in (0, 1)], 1)
        q[0], q[-1] = p[0], p[-1]
        p = q
    return _resample(p), float(ndi.distance_transform_edt(m).max())


def frame(mask_full, ext_frac=EXT_FRAC):
    """mask_full: bool H x W -> dict with the extended spine, normals, tangents, H, step and core columns."""
    r = np.flatnonzero(mask_full.any(1))
    c = np.flatnonzero(mask_full.any(0))
    r0, r1, c0, c1 = max(r[0] - 1, 0), r[-1] + 2, max(c[0] - 1, 0), c[-1] + 2
    sp, hw = spine_of(mask_full[r0:r1, c0:c1])
    sp = sp + [r0, c0]
    L = len(sp)
    E = int(max(EXT_MIN, ext_frac * L))
    k = min(8, L - 1)
    t0 = sp[0] - sp[k]
    t0 /= max(np.linalg.norm(t0), 1e-6)
    t1 = sp[-1] - sp[-1 - k]
    t1 /= max(np.linalg.norm(t1), 1e-6)
    head = sp[0] + np.outer(np.arange(E, 0, -1), t0)
    tail = sp[-1] + np.outer(np.arange(1, E + 1), t1)
    full = np.vstack([head, sp, tail])
    tan = np.gradient(full, axis=0)
    tan = np.stack([ndi.gaussian_filter1d(tan[:, i], 3, mode="nearest") for i in (0, 1)], 1)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-6)
    nor = np.stack([-tan[:, 1], tan[:, 0]], 1)
    H = float(np.clip(np.ceil(3 * max(hw, 1)), 16, 64))
    step = max(H / 32.0, (len(full) - 1) / (MAXCOL - 1))
    return dict(spine=full, normal=nor, tangent=tan, H=H, step=step, core=(E, E + L))


def grid(fr):
    cols = np.arange(0, len(fr["spine"]) - 1 + 1e-6, fr["step"])
    ts = np.linspace(-fr["H"], fr["H"], ROWS)
    idx = np.arange(len(fr["spine"]))
    si = np.interp(cols, idx, fr["spine"][:, 0])
    sj = np.interp(cols, idx, fr["spine"][:, 1])
    ni = np.interp(cols, idx, fr["normal"][:, 0])
    nj = np.interp(cols, idx, fr["normal"][:, 1])
    return si[None] + ts[:, None] * ni[None], sj[None] + ts[:, None] * nj[None]


def sample(img, fr, order=1):
    gi, gj = grid(fr)
    return ndi.map_coordinates(img.astype(np.float32), [gi, gj], order=order, mode="constant", cval=0)


def back(prob, fr, shape=(2048, 2048)):
    """ROWS x cols strip -> (r0, c0, back-projected probability on the bounding box of the band)."""
    sp, nor, tan = fr["spine"], fr["normal"], fr["tangent"]
    H = fr["H"]
    r0 = max(int(sp[:, 0].min() - H - 1), 0)
    r1 = min(int(sp[:, 0].max() + H + 2), shape[0])
    c0 = max(int(sp[:, 1].min() - H - 1), 0)
    c1 = min(int(sp[:, 1].max() + H + 2), shape[1])
    if r1 <= r0 or c1 <= c0:
        return r0, c0, np.zeros((0, 0), np.float32)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    q = np.stack([rr.ravel(), cc.ravel()], 1).astype(float)
    d, i = cKDTree(sp).query(q, distance_upper_bound=H * 1.5, workers=1)
    ok = np.isfinite(d)
    out = np.zeros(q.shape[0], np.float32)
    qi, ii = q[ok], i[ok]
    v = qi - sp[ii]
    t = (v * nor[ii]).sum(1)
    along = (v * tan[ii]).sum(1)
    keep = (np.abs(t) <= H) & (np.abs(along) <= 1.5)
    col = ii / fr["step"]
    row = (t + H) / (2 * H) * (ROWS - 1)
    vals = ndi.map_coordinates(prob.astype(np.float32), [row[keep], col[keep]], order=1, mode="nearest")
    tmp = np.zeros(qi.shape[0], np.float32)
    tmp[keep] = vals
    out[ok] = tmp
    return r0, c0, out.reshape(rr.shape)


def channels(img, prior, H, core, step):
    """img, prior: float ROWS x cols, prior in 0..1 -> 5 x ROWS x cols float32 network input."""
    med = np.median(img)
    mad = 1.4826 * np.median(np.abs(img - med)) + 1.0
    x = np.clip((img - med) / mad, -8, 8) / 4
    rows, cols = img.shape
    t = np.broadcast_to(np.linspace(-1, 1, rows)[:, None], (rows, cols))
    j = np.arange(cols) * step
    tip = np.maximum(core[0] - j, j - core[1])
    tip = np.broadcast_to(np.clip(tip / H, -4, 4)[None] / 4, (rows, cols))
    h = np.full((rows, cols), H / 64.0)
    return np.stack([x, prior, t, tip, h]).astype(np.float32)


class Block(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.SiLU(),
                               nn.Conv2d(b, b, 3, padding=1, bias=False), nn.GroupNorm(8, b))
        self.p = nn.Conv2d(a, b, 1) if a != b else nn.Identity()

    def forward(self, x):
        return F.silu(self.c(x) + self.p(x))


class StripNet(nn.Module):
    """U-Net on the straightened strip; its output is a residual on the prior logit, zero at init."""

    def __init__(self, c=5, w=32):
        super().__init__()
        ws = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.enc = nn.ModuleList([Block(c if i == 0 else ws[i - 1], ws[i]) for i in range(5)])
        self.dec = nn.ModuleList([Block(ws[i + 1] + ws[i], ws[i]) for i in range(3, -1, -1)])
        self.out = nn.Conv2d(w, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        r, c = x.shape[-2:]
        pr, pc = (-r) % 16, (-c) % 16
        y = F.pad(x, (0, pc, 0, pr), mode="replicate")
        fs = []
        for i, e in enumerate(self.enc):
            y = e(y if i == 0 else F.avg_pool2d(y, 2))
            fs.append(y)
        y = fs[-1]
        for d, f in zip(self.dec, fs[-2::-1]):
            y = d(torch.cat([F.interpolate(y, size=f.shape[-2:], mode="bilinear", align_corners=False), f], 1))
        res = self.out(y)[..., :r, :c]
        p = x[:, 1:2].clamp(0, 1) * 0.9 + 0.05
        return torch.log(p / (1 - p)) + res


def load_strip_net(path, device) -> StripNet:
    net = StripNet().to(device).eval()
    net.load_state_dict(torch.load(path, map_location=device))
    return net


@torch.no_grad()
def strip_prob(net, img, m, fr, device):
    """Mean logit over the four flips of the strip -> probability (ROWS x cols)."""
    im = sample(img, fr)
    pri = sample(m.astype(np.float32), fr)
    cols = im.shape[1]
    step = fr["step"]
    core = fr["core"]
    out = 0.0
    for fc in (0, 1):
        for fr_ in (0, 1):
            a, b, co = im, pri, core
            if fc:
                a, b, co = a[:, ::-1], b[:, ::-1], [(cols - 1) * step - core[1], (cols - 1) * step - core[0]]
            if fr_:
                a, b = a[::-1], b[::-1]
            x = torch.from_numpy(channels(a.copy(), b.copy(), fr["H"], co, step))[None].to(device)
            lg = net(x)[0, 0].cpu().numpy()
            if fr_:
                lg = lg[::-1]
            if fc:
                lg = lg[:, ::-1]
            out = out + lg
    return 1 / (1 + np.exp(-out / 4))


def domain_back(prob, fr, m):
    """Full-image probability: the strip's value inside its band, the mask's own state elsewhere."""
    r0, c0, p = back(prob, fr)
    _, _, d = back(np.ones_like(prob), fr)
    full = m.astype(np.float32).copy()
    sub = full[r0:r0 + p.shape[0], c0:c0 + p.shape[1]]
    sub[d > 0.5] = p[d > 0.5]
    return full


def largest(b):
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    return b if n <= 1 else lab == 1 + int(np.argmax(ndi.sum(b, lab, range(1, n + 1))))


def _enc(b):
    return mask_util.encode(np.asfortranarray(b.astype(np.uint8)))


def refine_image(net, img, rles, device) -> list[dict]:
    """The guarded pass on one image: same number and order of masks as `rles`."""
    if not len(rles):
        return []
    ms = [mask_util.decode(r).astype(bool) for r in rles]
    P = []
    for m in ms:
        fr = frame(m)
        P.append(domain_back(strip_prob(net, img, m, fr, device), fr, m))
    A = np.stack(P)
    idx = A.argmax(0)
    top = A.max(0) >= 0.5
    refined = [largest((idx == i) & top) for i in range(len(ms))]
    keep = [(refined[i] & ms[i]).sum() / max((refined[i] | ms[i]).sum(), 1) >= GUARD_IOU and refined[i].any()
            for i in range(len(ms))]
    taken = np.zeros_like(ms[0], bool)
    for i in range(len(ms)):
        if keep[i]:
            taken |= refined[i]
    out = []
    for i, m in enumerate(ms):
        b = refined[i] if keep[i] else (m & ~taken)
        out.append(_enc(b) if b.any() else rles[i])
    return out


__all__ = ["StripNet", "frame", "load_strip_net", "refine_image"]
