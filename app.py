"""
Mask Embedding Explorer — single-file Streamlit app.

    pip install streamlit plotly pandas numpy pillow scikit-learn umap-learn torch torchvision transformers
    streamlit run app.py --server.maxUploadSize 4000

Add images + COCO files, embed every mask with a model of your choice, explore them on a 2-D map,
click a point to see its crop, and check label quality.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors



# ############################################################################
# COCO PARSING, MASK DECODING, CROPS
# ############################################################################
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class MaskRecord:
    uid: str                 # unique id: "<coco file>::<row>::<ann id>"
    image_key: str           # normalised file stem used to match images <-> annotations
    image_name: str
    coco_source: str
    ann_id: Any
    category: str            # category NAME (ids differ between COCO files, names don't)
    bbox: tuple              # x, y, w, h
    area: float
    segmentation: Any
    img_h: int | None
    img_w: int | None
    iscrowd: int = 0

    @property
    def effective_area(self) -> float:
        return self.area if self.area else float(self.bbox[2] * self.bbox[3])


def image_key(name: str) -> str:
    """'train/imgs/IMG_001.JPG' -> 'img_001' (folder + extension independent)."""
    base = os.path.basename(str(name).replace("\\", "/"))
    return os.path.splitext(base)[0].lower()


# --------------------------------------------------------------------------- COCO
def parse_coco(data: dict, source: str) -> tuple[list[MaskRecord], int]:
    """Return (records, n_skipped). One record per annotation that has a segmentation."""
    if not isinstance(data, dict) or "annotations" not in data or "images" not in data:
        raise ValueError("not a COCO file (needs 'images' and 'annotations')")
    cats = {c["id"]: str(c.get("name", c["id"])) for c in data.get("categories", [])}
    imgs = {im["id"]: im for im in data["images"]}
    records, skipped = [], 0
    for row, ann in enumerate(data["annotations"]):
        im = imgs.get(ann.get("image_id"))
        seg = ann.get("segmentation")
        if im is None or not seg:
            skipped += 1
            continue
        cid = ann.get("category_id")
        bbox = tuple(float(v) for v in (ann.get("bbox") or (0, 0, 0, 0)))
        records.append(MaskRecord(
            uid=f"{source}::{row}::{ann.get('id', row)}",
            image_key=image_key(im.get("file_name", "")),
            image_name=os.path.basename(str(im.get("file_name", ""))),
            coco_source=source,
            ann_id=ann.get("id", row),
            category=cats.get(cid, f"category_{cid}"),
            bbox=bbox,
            area=float(ann.get("area") or 0),
            segmentation=seg,
            img_h=im.get("height"),
            img_w=im.get("width"),
            iscrowd=int(ann.get("iscrowd", 0) or 0),
        ))
    return records, skipped


# --------------------------------------------------------------------------- masks
def _rle_string_to_counts(s) -> list[int]:
    """Port of pycocotools' rleFrString (compressed COCO RLE)."""
    if isinstance(s, str):
        s = s.encode("ascii")
    counts, p, m = [], 0, 0
    while p < len(s):
        x, k, more = 0, 0, 1
        while more:
            c = s[p] - 48
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += counts[m - 2]
        counts.append(x)
        m += 1
    return counts


def _rle_counts_to_mask(counts, h: int, w: int) -> np.ndarray:
    flat = np.zeros(h * w, dtype=bool)
    idx, val = 0, False
    for c in counts:
        c = int(c)
        if val:
            flat[idx:idx + c] = True
        idx += c
        val = not val
    return flat.reshape((h, w), order="F")  # COCO RLE is column-major


def decode_mask(seg, h: int, w: int) -> np.ndarray:
    """Decode a COCO segmentation into a (h, w) boolean mask matching the actual image size."""
    if isinstance(seg, list) and seg and isinstance(seg[0], dict):  # list of RLEs
        out = np.zeros((h, w), bool)
        for s in seg:
            out |= decode_mask(s, h, w)
        return out
    if isinstance(seg, list):  # polygons
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for poly in seg:
            if poly and len(poly) >= 6:
                draw.polygon([float(v) for v in poly], outline=1, fill=1)
        return np.asarray(canvas, dtype=bool)
    if isinstance(seg, dict) and "counts" in seg:
        rh, rw = seg.get("size", [h, w])
        counts = seg["counts"]
        if not isinstance(counts, list):
            counts = _rle_string_to_counts(counts)
        m = _rle_counts_to_mask(counts, int(rh), int(rw))
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        return m
    raise ValueError(f"unsupported segmentation format: {type(seg).__name__}")


# --------------------------------------------------------------------------- images
def load_image(src) -> Image.Image:
    """src = raw bytes (uploads) or a file path (local folder)."""
    im = Image.open(io.BytesIO(src)) if isinstance(src, (bytes, bytearray)) else Image.open(src)
    im.load()
    if im.mode in ("I;16", "I;16B", "I", "F"):  # 16-bit / float images -> stretch to 8-bit
        a = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(a)
    return im.convert("RGB") if im.mode != "RGB" else im


def mask_bbox(mask: np.ndarray, fallback=(0, 0, 0, 0)) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs):
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    x, y, w, h = fallback
    return int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h))


def make_crop(img: Image.Image, mask: np.ndarray, bbox=(0, 0, 0, 0), mode: str = "masked",
              pad: float = 0.1, square: bool = True, bg=(124, 116, 104)) -> Image.Image:
    """mode='masked': pixels outside the mask are filled with `bg`; mode='bbox': raw box crop."""
    W, H = img.size
    x0, y0, x1, y1 = mask_bbox(mask, bbox)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    px, py = int(round(bw * pad)), int(round(bh * pad))
    if square:
        side = max(bw + 2 * px, bh + 2 * py)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        X0, Y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        X1, Y1 = X0 + side, Y0 + side
    else:
        X0, Y0, X1, Y1 = x0 - px, y0 - py, x1 + px, y1 + py
    bg_arr = np.array(bg, dtype=np.uint8)
    out = np.empty((Y1 - Y0, X1 - X0, 3), np.uint8)
    out[:] = bg_arr
    sx0, sy0, sx1, sy1 = max(0, X0), max(0, Y0), min(W, X1), min(H, Y1)
    if sx1 > sx0 and sy1 > sy0:
        region = np.asarray(img)[sy0:sy1, sx0:sx1]
        if mode == "masked":
            region = np.where(mask[sy0:sy1, sx0:sx1, None], region, bg_arr)
        out[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = region
    return Image.fromarray(out)


def _boundary(mask: np.ndarray, t: int) -> np.ndarray:
    er = mask.copy()
    for _ in range(t):
        er = er & np.roll(er, 1, 0) & np.roll(er, -1, 0) & np.roll(er, 1, 1) & np.roll(er, -1, 1)
    return mask & ~er


def overlay(img: Image.Image, mask: np.ndarray, color=(255, 80, 80), alpha: float = 0.4,
            max_side: int = 1000) -> Image.Image:
    """Full image with the mask tinted and outlined, for the detail view."""
    arr = np.asarray(img, dtype=np.float32).copy()
    c = np.array(color, np.float32)
    arr[mask] = arr[mask] * (1 - alpha) + c * alpha
    edge = _boundary(mask, max(2, int(max(img.size) / 350)))
    arr[edge] = c
    out = Image.fromarray(arr.astype(np.uint8))
    out.thumbnail((max_side, max_side))
    return out


# ############################################################################
# EMBEDDING MODELS
# ############################################################################
def get_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class BaseEmbedder:
    name = "base"
    supports_text = False

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def embed_text(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError("This model can't embed text.")


# --------------------------------------------------------------------------- baseline
class QuickBaseline(BaseEmbedder):
    """Colour histogram + gradient orientations + tiny thumbnail. For smoke tests only."""
    name = "Quick test (colour + texture features)"

    def embed(self, images):
        feats = []
        for im in images:
            small = im.convert("RGB").resize((64, 64))
            hsv = np.asarray(small.convert("HSV"), np.float32) / 255
            hist = np.concatenate([np.histogram(hsv[..., c], bins=12, range=(0, 1))[0] for c in range(3)]) / 4096
            g = np.asarray(small.convert("L"), np.float32) / 255
            gy, gx = np.gradient(g)
            hog = np.histogram(np.arctan2(gy, gx) % np.pi, bins=9, range=(0, np.pi), weights=np.hypot(gx, gy))[0]
            thumb = np.asarray(im.convert("L").resize((8, 8)), np.float32).ravel() / 255
            feats.append(np.concatenate([hist, hog / (hog.sum() + 1e-6), 0.3 * thumb]))
        return np.stack(feats).astype(np.float32)


# --------------------------------------------------------------------------- Hugging Face
def _as_tensor(v):
    if hasattr(v, "pooler_output") and v.pooler_output is not None:
        return v.pooler_output
    if hasattr(v, "last_hidden_state"):
        return v.last_hidden_state[:, 0]
    return v


class HFVisionEmbedder(BaseEmbedder):
    """Any vision backbone on the HF hub (DINOv2, ViT, ResNet, ConvNeXt, a local checkpoint...)."""

    def __init__(self, model_id: str, device: str, pooling: str = "pooled"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch, self.device, self.pooling, self.name = torch, device, pooling, model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    def embed(self, images):
        inputs = self.processor(images=[i.convert("RGB") for i in images], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        h = getattr(out, "last_hidden_state", None)
        if self.pooling == "mean" and h is not None:
            v = h[:, 1:].mean(1) if h.dim() == 3 else h.mean(dim=(2, 3))  # tokens or CNN feature map
        elif getattr(out, "pooler_output", None) is not None:
            v = out.pooler_output
        else:
            v = h[:, 0] if h.dim() == 3 else h.mean(dim=(2, 3))
        return v.flatten(1).float().cpu().numpy()


class HFClipEmbedder(BaseEmbedder):
    """CLIP / SigLIP: image embeddings plus text embeddings in the same space (enables text search)."""
    supports_text = True

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device, self.name = torch, device, model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.is_siglip = "siglip" in self.model.config.model_type

    def embed(self, images):
        pv = self.processor(images=[i.convert("RGB") for i in images], return_tensors="pt")["pixel_values"]
        with self.torch.no_grad():
            v = _as_tensor(self.model.get_image_features(pixel_values=pv.to(self.device)))
        return v.float().cpu().numpy()

    def embed_text(self, texts):
        kw = dict(padding="max_length", max_length=64) if self.is_siglip else dict(padding=True)
        inp = self.processor(text=texts, return_tensors="pt", truncation=True, **kw).to(self.device)
        with self.torch.no_grad():
            v = _as_tensor(self.model.get_text_features(**inp))
        return v.float().cpu().numpy()


# --------------------------------------------------------------------------- your own model
class CustomEmbedder(BaseEmbedder):
    def __init__(self, path: str, device: str):
        tag = hashlib.md5(open(path, "rb").read()).hexdigest()[:10]
        spec = importlib.util.spec_from_file_location(f"user_embedder_{tag}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "Embedder"):
            raise ValueError(f"{os.path.basename(path)} must define a class named `Embedder`.")
        try:
            self.obj = mod.Embedder(device=device)
        except TypeError:
            self.obj = mod.Embedder()
        self.name = getattr(self.obj, "name", os.path.basename(path))
        self.supports_text = hasattr(self.obj, "embed_text")

    def embed(self, images):
        return np.asarray(self.obj.embed(images), dtype=np.float32)

    def embed_text(self, texts):
        return np.asarray(self.obj.embed_text(texts), dtype=np.float32)


# --------------------------------------------------------------------------- factory
TEXT_MODEL_TYPES = {"clip", "siglip", "siglip2"}


def build_embedder(backend: str, model_id: str | None = None, pooling: str = "pooled",
                   custom_path: str | None = None, device: str = "auto") -> BaseEmbedder:
    device = get_device(device)
    if backend == "quick":
        return QuickBaseline()
    if backend == "custom":
        if not custom_path or not os.path.exists(custom_path):
            raise ValueError("Upload your embedder .py file (or give its path) in the Model section.")
        return CustomEmbedder(custom_path, device)
    try:
        import torch  # noqa: F401
        from transformers import AutoConfig
    except ImportError as e:
        raise ImportError("This model needs PyTorch and transformers: pip install torch transformers") from e
    if not model_id:
        raise ValueError("Enter a Hugging Face model ID or a local checkpoint folder.")
    cfg = AutoConfig.from_pretrained(model_id)
    if cfg.model_type in TEXT_MODEL_TYPES:
        return HFClipEmbedder(model_id, device)
    return HFVisionEmbedder(model_id, device, pooling)


# ############################################################################
# ANALYSIS: PROJECTION, NEIGHBOURS, LABEL QUALITY
# ############################################################################
def l2n(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def project(X, method="UMAP", n_neighbors=15, min_dist=0.1, perplexity=30.0, seed=42):
    """Returns (Y [n,2], method actually used, note)."""
    n = len(X)
    Xn = l2n(np.asarray(X, np.float32))
    if n < 3:
        return np.column_stack([np.arange(n), np.zeros(n)]).astype(float), "none", "Need at least 3 masks."
    if method == "PCA" or n < 10:
        note = "" if method == "PCA" else "Fewer than 10 masks, so PCA was used."
        dims = min(2, Xn.shape[1])
        Y = PCA(dims, random_state=seed).fit_transform(Xn)
        if dims == 1:
            Y = np.column_stack([Y[:, 0], np.zeros(n)])
        return Y, "PCA", note
    Xr = PCA(50, random_state=seed).fit_transform(Xn) if (Xn.shape[1] > 50 and n > 50) else Xn
    note = ""
    if method == "UMAP":
        try:
            import umap
            Y = umap.UMAP(n_neighbors=int(min(n_neighbors, n - 1)), min_dist=min_dist,
                          random_state=seed).fit_transform(Xr)
            return Y, "UMAP", ""
        except ImportError:
            note = "umap-learn isn't installed, so t-SNE was used (pip install umap-learn)."
    perp = float(min(perplexity, max(2.0, (n - 1) / 3)))
    Y = TSNE(2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(Xr)
    return Y, "t-SNE", note


def knn(X, k=10):
    """Cosine nearest neighbours, self excluded. Returns (indices [n,k], similarities [n,k])."""
    Xn = l2n(X)
    n = len(Xn)
    k = int(min(k, n - 1))
    if k < 1:
        return np.zeros((n, 0), int), np.zeros((n, 0))
    d, idx = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(Xn).kneighbors(Xn)
    out_i, out_s = np.empty((n, k), int), np.empty((n, k))
    for r in range(n):
        keep = idx[r] != r
        if keep.all():
            keep[-1] = False
        out_i[r], out_s[r] = idx[r][keep][:k], 1 - d[r][keep][:k]
    return out_i, out_s


def label_consistency(labels, nbr_i, nbr_s):
    """Similarity-weighted share of neighbours with the same label + the neighbours' majority label."""
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    if nbr_i.shape[1] == 0:
        return np.ones(n), labels.copy(), np.ones(n)
    nl = labels[nbr_i]
    w = np.clip(nbr_s, 0, None) + 1e-6
    consistency = ((nl == labels[:, None]) * w).sum(1) / w.sum(1)
    suggested, conf = np.empty(n, dtype=object), np.empty(n)
    for r in range(n):
        scores: dict = {}
        for lab, ww in zip(nl[r], w[r]):
            scores[lab] = scores.get(lab, 0.0) + ww
        best = max(scores, key=scores.get)
        suggested[r], conf[r] = best, scores[best] / w[r].sum()
    return consistency, suggested, conf


def class_stats(X, labels):
    """Class centroids, centroid similarity matrix, distance-to-own-centroid and per-class z-score."""
    Xn = l2n(X)
    labels = np.asarray(labels, dtype=object)
    classes = sorted(set(labels))
    C = l2n(np.stack([Xn[labels == c].mean(0) for c in classes]))
    ci = {c: i for i, c in enumerate(classes)}
    own = C[[ci[l] for l in labels]]
    dist = 1 - (Xn * own).sum(1)
    z, spread = np.zeros(len(X)), {}
    for c in classes:
        m = labels == c
        d = dist[m]
        spread[c] = float(d.mean())
        sd = d.std()
        z[m] = (d - d.mean()) / (sd if sd > 1e-9 else 1.0)
    return dict(classes=classes, centroids=C, sim=C @ C.T, dist=dist, z=z, spread=spread)


def silhouette_by_class(X, labels, max_n=6000, seed=0):
    from sklearn.metrics import silhouette_samples
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    if len(set(labels)) < 2 or n <= len(set(labels)):
        return {}
    idx = np.arange(n) if n <= max_n else np.random.default_rng(seed).choice(n, max_n, replace=False)
    sub = labels[idx]
    if len(set(sub)) < 2:
        return {}
    s = silhouette_samples(l2n(X[idx]), sub, metric="cosine")
    return {c: float(s[sub == c].mean()) for c in sorted(set(sub))}


def near_duplicates(nbr_i, nbr_s, thr=0.97):
    pairs: dict = {}
    for i in range(len(nbr_i)):
        for j, s in zip(nbr_i[i], nbr_s[i]):
            if s >= thr:
                key = (min(i, int(j)), max(i, int(j)))
                pairs[key] = max(pairs.get(key, 0.0), float(s))
    return sorted(((a, b, s) for (a, b), s in pairs.items()), key=lambda t: -t[2])


def search(X, q, top=12):
    sims = l2n(X) @ l2n(np.asarray(q, np.float32).reshape(1, -1))[0]
    order = np.argsort(-sims)[:top]
    return order, sims[order]


# ############################################################################
# DEMO DATASET
# ############################################################################
SHAPES = ["circle", "square", "triangle", "star"]
DEMO_COLORS = {"circle": (220, 80, 70), "square": (60, 130, 220), "triangle": (70, 180, 90), "star": (230, 180, 40)}


def _poly(kind, cx, cy, r, rot):
    if kind == "circle":
        return [(cx + r * math.cos(2 * math.pi * i / 36), cy + r * math.sin(2 * math.pi * i / 36)) for i in range(36)]
    if kind == "square":
        return [(cx + r * math.cos(rot + math.pi / 4 + i * math.pi / 2), cy + r * math.sin(rot + math.pi / 4 + i * math.pi / 2)) for i in range(4)]
    if kind == "triangle":
        return [(cx + r * math.cos(rot + i * 2 * math.pi / 3), cy + r * math.sin(rot + i * 2 * math.pi / 3)) for i in range(3)]
    return [(cx + (r if i % 2 == 0 else r * 0.45) * math.cos(rot + i * math.pi / 5),
             cy + (r if i % 2 == 0 else r * 0.45) * math.sin(rot + i * math.pi / 5)) for i in range(10)]


def _area(pts):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def make_demo_dataset(n_images=40, seed=0, noise_rate=0.06):
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    bases = [1, 10]  # same names, different ids in the two files
    cocos = [{"images": [], "annotations": [], "categories": [{"id": b + i, "name": s} for i, s in enumerate(SHAPES)]}
             for b in bases]
    ann_ids, images = [1, 1], []
    W, H = 320, 240
    for k in range(n_images):
        fi = k % 2
        bg = nrng.integers(0, 50, (H, W, 3)) + np.linspace(40, 110, W)[None, :, None]
        img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        fname = f"demo_{k:03d}.png"
        cocos[fi]["images"].append({"id": k + 1, "file_name": fname, "width": W, "height": H})
        for _ in range(rng.randint(2, 5)):
            kind = rng.choice(SHAPES)
            r = rng.uniform(16, 42)
            cx, cy = rng.uniform(r, W - r), rng.uniform(r, H - r)
            pts = _poly(kind, cx, cy, r, rng.uniform(0, 2 * math.pi))
            draw.polygon(pts, fill=tuple(int(np.clip(v + rng.gauss(0, 30), 0, 255)) for v in DEMO_COLORS[kind]))
            label = kind if rng.random() >= noise_rate else rng.choice([s for s in SHAPES if s != kind])
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            cocos[fi]["annotations"].append({
                "id": ann_ids[fi], "image_id": k + 1, "category_id": bases[fi] + SHAPES.index(label),
                "segmentation": [[round(v, 2) for p in pts for v in p]],
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": _area(pts), "iscrowd": 0})
            ann_ids[fi] += 1
        buf = io.BytesIO()
        img.save(buf, "PNG")
        images.append((fname, buf.getvalue()))
    return images, [("demo_part1.json", cocos[0]), ("demo_part2.json", cocos[1])]


# ############################################################################
# TEMPLATE FOR YOUR OWN MODEL (offered as a download in the app)
# ############################################################################
CUSTOM_MODEL_TEMPLATE = '"""Template for plugging YOUR model into the Mask Embedding Explorer.\n\nRules:\n  * define a class called `Embedder`\n  * __init__(self, device) loads the model once\n  * embed(self, images) receives a list of PIL RGB crops and returns a (N, D) numpy array\n  * optional: embed_text(self, texts) -> (N, D) array in the same space (enables text search)\n\nTip for segmentation models (Mask R-CNN, Mask2Former, SAM, a U-Net...):\nuse the model\'s backbone / encoder as the feature extractor and global-average-pool its\nlast feature map. That shows you how *your* model sees the masks.\n"""\nimport numpy as np\nimport torch\nimport torchvision\nfrom torchvision import transforms\n\n\nclass Embedder:\n    name = "My model (ResNet-18 example)"\n\n    def __init__(self, device="cpu"):\n        self.device = device\n        # ---- replace this block with your own model -------------------------------\n        # self.model = MyNet(num_classes=...)\n        # self.model.load_state_dict(torch.load("weights.pth", map_location=device))\n        # self.model.head = torch.nn.Identity()          # drop the classifier\n        self.model = torchvision.models.resnet18(weights="DEFAULT")\n        self.model.fc = torch.nn.Identity()\n        # ---------------------------------------------------------------------------\n        self.model.to(device).eval()\n        self.transform = transforms.Compose([\n            transforms.Resize((224, 224)),\n            transforms.ToTensor(),\n            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n        ])\n\n    @torch.no_grad()\n    def embed(self, images):\n        batch = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device)\n        feats = self.model(batch)\n        return feats.flatten(1).cpu().numpy().astype(np.float32)\n'


# ############################################################################
# DASHBOARD
# ############################################################################
st.set_page_config(page_title="Mask Embedding Explorer", page_icon="🧩", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
div[data-testid="stImage"] img {border-radius: 6px;}
</style>""", unsafe_allow_html=True)

THUMB = 256
PALETTE = px.colors.qualitative.Bold + px.colors.qualitative.Pastel + px.colors.qualitative.Dark24
ss = st.session_state
for k, v in dict(result=None, sel_idx=None, sel_group=[], last_sig=None, overlay_cache={}).items():
    ss.setdefault(k, v)

PRESETS = {
    "DINOv2 small — best general visual similarity": ("hf", "facebook/dinov2-small"),
    "DINOv2 base — stronger, slower": ("hf", "facebook/dinov2-base"),
    "CLIP ViT-B/32 — also enables text search": ("hf", "openai/clip-vit-base-patch32"),
    "ResNet-50 (ImageNet)": ("hf", "microsoft/resnet-50"),
    "Any Hugging Face model ID or local checkpoint": ("hf", None),
    "My own model (Python file)": ("custom", None),
    "Quick test — no download, no GPU": ("quick", None),
}


# =========================================================================== helpers
@st.cache_resource(show_spinner=False)
def parse_coco_bytes(data: bytes, name: str):
    return parse_coco(json.loads(data.decode("utf-8")), name)


@st.cache_data(show_spinner=False)
def scan_folder(folder: str):
    found, dups = {}, []
    for p in Path(folder).expanduser().rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            k = image_key(p.name)
            if k in found:
                dups.append(p.name)
            found[k] = str(p)
    return found, dups


@st.cache_data(show_spinner=False)
def demo_dataset(n):
    return make_demo_dataset(n_images=n)


@st.cache_resource(show_spinner="Loading model (the first run downloads the weights)…")
def load_embedder(backend, model_id, pooling, custom_path, device):
    return build_embedder(backend, model_id=model_id, pooling=pooling, custom_path=custom_path, device=device)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def data_uri(img, size=96):
    im = img.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def cat_colors(cats):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(cats))}


# Project folder auto-detection: an *_embedder.py next to app.py, and data/images + data/annotations
APP_DIR = Path(__file__).resolve().parent
LOCAL_EMBEDDERS = sorted(str(p) for p in APP_DIR.glob("*_embedder.py"))
DATA_IMAGES, DATA_ANNS = APP_DIR / "data" / "images", APP_DIR / "data" / "annotations"
HAS_LOCAL_DATA = DATA_IMAGES.is_dir() and DATA_ANNS.is_dir()


def select_one(i):
    ss.sel_idx, ss.sel_group = int(i), []


def select_group(idxs):
    ss.sel_group, ss.sel_idx = [int(i) for i in idxs], None


# =========================================================================== sidebar: data
st.sidebar.markdown("## 🧩 Mask Embedding Explorer")
images: dict = {}
records: list = []
coco_names: list = []
problems_in: list = []
dup_images: list = []

with st.sidebar.expander("① Images & COCO files", expanded=ss.result is None):
    source = st.radio("Where are your files?", ["Upload", "Local folder", "Demo"], horizontal=True,
                      index=1 if HAS_LOCAL_DATA else 0,
                      help="Local folder is best for large datasets (no upload size limit).")
    if source == "Upload":
        up_imgs = st.file_uploader("Images", type=[e.strip(".") for e in IMAGE_EXTS], accept_multiple_files=True)
        up_cocos = st.file_uploader("COCO files (.json) — one file may cover many images", type=["json"],
                                    accept_multiple_files=True)
        for f in up_imgs or []:
            k = image_key(f.name)
            if k in images:
                dup_images.append(f.name)
            images[k] = f.getvalue()
        for f in up_cocos or []:
            try:
                recs, _ = parse_coco_bytes(f.getvalue(), f.name)
                records += recs
                coco_names.append(f.name)
            except Exception as e:
                problems_in.append(f"{f.name}: {e}")
    elif source == "Local folder":
        folder = st.text_input("Images folder (searched recursively)", str(DATA_IMAGES) if HAS_LOCAL_DATA else "",
                               placeholder="/data/project/images")
        coco_txt = st.text_area("COCO files or folders, one per line", str(DATA_ANNS) if HAS_LOCAL_DATA else "",
                                placeholder="/data/project/annotations")
        if folder:
            if os.path.isdir(os.path.expanduser(folder)):
                found, dup_images = scan_folder(folder)
                images = dict(found)
            else:
                problems_in.append(f"Folder not found: {folder}")
        for line in [l.strip() for l in coco_txt.splitlines() if l.strip()]:
            p = Path(line).expanduser()
            paths = sorted(p.rglob("*.json")) if p.is_dir() else [p]
            for jp in paths:
                try:
                    recs, _ = parse_coco_bytes(jp.read_bytes(), jp.name)
                    records += recs
                    coco_names.append(jp.name)
                except Exception as e:
                    problems_in.append(f"{jp}: {e}")
    else:
        n_demo = st.slider("Demo images", 10, 150, 40, 10)
        st.caption("Coloured shapes in two COCO files. About 6% of the labels are deliberately wrong — "
                   "see if the Label quality tab finds them.")
        demo_imgs, demo_cocos = demo_dataset(n_demo)
        images = {image_key(n): b for n, b in demo_imgs}
        for name, coco in demo_cocos:
            recs, _ = parse_coco_bytes(json.dumps(coco).encode(), name)
            records += recs
            coco_names.append(name)

    matched = [r for r in records if r.image_key in images]
    all_cats = sorted({r.category for r in matched})
    selected: list = []
    if matched:
        st.divider()
        cats_sel = st.multiselect("Classes to include", all_cats, default=all_cats)
        c1, c2 = st.columns(2)
        min_area = c1.number_input("Min mask area (px²)", 0, value=0, step=50)
        max_per = c2.number_input("Max masks per class", 0, value=0, step=100, help="0 = keep all. "
                                  "Randomly subsamples big classes for speed and balance.")
        skip_crowd = st.checkbox("Skip crowd annotations (iscrowd = 1)", True)
        by_cat = defaultdict(list)
        for r in matched:
            if r.category in cats_sel and r.effective_area >= min_area and not (skip_crowd and r.iscrowd):
                by_cat[r.category].append(r)
        rng = np.random.default_rng(0)
        for c in sorted(by_cat):
            rs = by_cat[c]
            if max_per and len(rs) > max_per:
                rs = [rs[i] for i in sorted(rng.choice(len(rs), max_per, replace=False))]
            selected += rs

# =========================================================================== sidebar: model + crop
with st.sidebar.expander("② Embedding model", expanded=ss.result is None):
    choice = st.selectbox("Model", list(PRESETS),
                          index=list(PRESETS).index("My own model (Python file)") if LOCAL_EMBEDDERS else 0,
                          help="The model decides what 'similar' means. Use your own model to see "
                               "how *it* organises the masks.")
    backend, model_id = PRESETS[choice]
    custom_path, pooling = None, "pooled"
    if choice.startswith("Any"):
        model_id = st.text_input("Model ID or local folder", "facebook/dinov2-small")
    if backend == "custom":
        st.caption("Your file defines `class Embedder` with `embed(images) -> (N, D) array`.")
        st.download_button("Download template", CUSTOM_MODEL_TEMPLATE, "my_embedder.py", width="stretch")
        up_py = st.file_uploader("Upload your embedder (.py)", type=["py"])
        custom_path = st.text_input("…or its path on this machine",
                                    LOCAL_EMBEDDERS[0] if LOCAL_EMBEDDERS else "") or None
        if custom_path:
            custom_path = os.path.expanduser(custom_path)
        if up_py is not None:
            raw = up_py.getvalue()
            custom_path = os.path.join(tempfile.gettempdir(), f"embedder_{hashlib.md5(raw).hexdigest()[:10]}.py")
            Path(custom_path).write_bytes(raw)
    if backend == "hf":
        pooling = st.radio("Pooling", ["pooled", "mean"], horizontal=True,
                           format_func=lambda p: {"pooled": "CLS / pooled", "mean": "Mean of patches"}[p],
                           help="Ignored for CLIP models.")
    c1, c2 = st.columns(2)
    device = c1.selectbox("Device", ["auto", "cuda", "mps", "cpu"])
    batch_size = c2.number_input("Batch size", 1, 256, 32)

with st.sidebar.expander("③ Crop style"):
    crop_mode = st.radio("What the model sees", ["masked", "bbox"],
                         format_func=lambda m: {"masked": "Mask only (background filled)",
                                                "bbox": "Bounding box (keeps background)"}[m])
    pad = st.slider("Padding around the mask", 0, 100, 10, format="%d%%") / 100
    square = st.checkbox("Pad to a square (no stretching)", True)
    bg_hex = st.color_picker("Fill colour", "#7C7468", help="Default is close to the ImageNet mean colour.")
crop_cfg = dict(mode=crop_mode, pad=pad, square=square, bg=hex_to_rgb(bg_hex))
model_cfg = dict(backend=backend, model_id=model_id, pooling=pooling, custom_path=custom_path, device=device)
settings_sig = json.dumps([model_cfg, crop_cfg, len(selected)], default=str)

run_clicked = st.sidebar.button(f"🚀 Extract embeddings ({len(selected)} masks)", type="primary",
                                width="stretch", disabled=not selected)


# =========================================================================== extraction
def extract(selected, images, embedder, crop_cfg, batch_size):
    groups = defaultdict(list)
    for i, r in enumerate(selected):
        groups[r.image_key].append(i)
    n, done = len(selected), 0
    feats, thumbs, px_area, problems = {}, {}, {}, []
    b_imgs, b_idx = [], []
    bar = st.progress(0.0, text="Starting…")

    def flush():
        if not b_imgs:
            return
        V = np.asarray(embedder.embed(b_imgs), dtype=np.float32)
        if V.ndim != 2 or len(V) != len(b_imgs):
            raise ValueError(f"The model returned shape {V.shape} for {len(b_imgs)} crops; expected (N, D).")
        for j, v in zip(b_idx, V):
            feats[j] = v
        b_imgs.clear()
        b_idx.clear()

    for key, idxs in groups.items():
        try:
            img = load_image(images[key])
        except Exception as e:
            problems.append({"image": key, "annotation": "", "problem": f"can't open image: {e}"})
            done += len(idxs)
            continue
        W, H = img.size
        for i in idxs:
            r = selected[i]
            if r.img_w and r.img_h and (r.img_w, r.img_h) != (W, H):
                problems.append({"image": r.image_name, "annotation": r.ann_id,
                                 "problem": f"COCO says {r.img_w}×{r.img_h}, file is {W}×{H} (masks may be offset)"})
            try:
                mask = decode_mask(r.segmentation, H, W)
                if not mask.any():
                    raise ValueError("mask is empty")
                crop = make_crop(img, mask, r.bbox, **crop_cfg)
            except Exception as e:
                problems.append({"image": r.image_name, "annotation": r.ann_id, "problem": str(e)})
                continue
            th = crop.copy()
            th.thumbnail((THUMB, THUMB))
            thumbs[i], px_area[i] = th, int(mask.sum())
            crop.thumbnail((512, 512))
            b_imgs.append(crop)
            b_idx.append(i)
            if len(b_imgs) >= batch_size:
                flush()
        done += len(idxs)
        bar.progress(min(done / n, 1.0), text=f"Embedding masks… {done}/{n}")
    flush()
    bar.empty()
    keep = sorted(feats)
    return keep, feats, thumbs, px_area, problems


if run_clicked:
    try:
        embedder = load_embedder(**model_cfg)
        keep, feats, thumbs, px_area, problems = extract(selected, images, embedder, crop_cfg, int(batch_size))
        if len(keep) < 2:
            st.error("Fewer than 2 masks could be embedded. Check the problems listed below.")
            st.dataframe(pd.DataFrame(problems))
        else:
            recs = [selected[i] for i in keep]
            df = pd.DataFrame({
                "idx": np.arange(len(keep)),
                "category": [r.category for r in recs],
                "image": [r.image_name for r in recs],
                "coco_file": [r.coco_source for r in recs],
                "ann_id": [str(r.ann_id) for r in recs],
                "area_px": [px_area[i] for i in keep],
                "uid": [r.uid for r in recs],
            })
            ss.result = dict(df=df, X=np.stack([feats[i] for i in keep]), thumbs=[thumbs[i] for i in keep],
                             records=recs, images=images, crop_cfg=crop_cfg, model_cfg=model_cfg,
                             model_name=embedder.name, supports_text=embedder.supports_text,
                             problems=problems, sig=settings_sig, proj={}, ana={})
            ss.sel_idx, ss.sel_group, ss.last_sig, ss.overlay_cache = None, [], None, {}
            st.rerun()
    except Exception as e:
        st.error(f"Extraction failed: {e}")

res = ss.result

# =========================================================================== landing page
if res is None:
    st.title("🧩 Mask Embedding Explorer")
    st.write("Add images and their COCO files, pick a model, then press **Extract embeddings**. "
             "Every mask becomes one point on the map; click a point to see the mask.")
    m = st.columns(5)
    m[0].metric("Images", len(images))
    m[1].metric("COCO files", len(coco_names))
    m[2].metric("Masks matched", len(matched))
    m[3].metric("Masks without image", len(records) - len(matched))
    m[4].metric("Classes", len(all_cats))
    for p in problems_in:
        st.error(p)
    if dup_images:
        st.warning(f"{len(dup_images)} images share a file name with another image; only one of each is used: "
                   + ", ".join(dup_images[:5]))
    if records and not matched:
        st.warning("None of the annotations match the images. Images are matched by file name "
                   "(folder and extension are ignored), e.g. `train/IMG_1.jpg` in COCO ↔ `IMG_1.png` uploaded.")
    if not records and not images:
        st.info("👈 Start in the sidebar — or pick **Demo** to try it with synthetic data.")
    if selected:
        left, right = st.columns([1, 2])
        with left:
            cnt = pd.Series([r.category for r in selected]).value_counts().rename_axis("class").reset_index(name="masks")
            fig = px.bar(cnt, x="masks", y="class", orientation="h", color="class",
                         color_discrete_map=cat_colors(cnt["class"]), height=max(220, 28 * len(cnt)))
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
            st.markdown("**Masks per class**")
            st.plotly_chart(fig, width="stretch")
        with right:
            st.markdown("**Preview of what the model will see** (updates with the crop settings)")
            buckets = defaultdict(list)
            for r in selected:
                buckets[r.category].append(r)
            picks, i = [], 0
            while len(picks) < 8 and any(len(b) > i for b in buckets.values()):
                picks += [b[i] for b in buckets.values() if len(b) > i]
                i += 1
            cols = st.columns(4)
            for n, r in enumerate(picks[:8]):
                try:
                    img = load_image(images[r.image_key])
                    mask = decode_mask(r.segmentation, img.size[1], img.size[0])
                    cols[n % 4].image(make_crop(img, mask, r.bbox, **crop_cfg), caption=r.category,
                                      width="stretch")
                except Exception as e:
                    cols[n % 4].warning(f"{r.image_name}: {e}")
    st.stop()

# =========================================================================== analysis settings
with st.sidebar.expander("④ Map & analysis", expanded=True):
    method = st.radio("Projection", ["UMAP", "t-SNE", "PCA"], horizontal=True,
                      help="UMAP keeps both local and global structure; t-SNE focuses on local clusters; "
                           "PCA is linear and fastest.")
    n_nb, min_dist, perp = 15, 0.1, 30.0
    if method == "UMAP":
        n_nb = st.slider("UMAP neighbours", 2, 100, 15, help="Higher = more global structure.")
        min_dist = st.slider("UMAP min distance", 0.0, 1.0, 0.1, 0.05, help="Lower = tighter clusters.")
    elif method == "t-SNE":
        perp = st.slider("Perplexity", 2.0, 100.0, 30.0, 1.0)
    k_nn = st.slider("Neighbours used for analysis (k)", 3, 30, 10)
    thr = st.slider("Flag as suspicious below agreement", 0.0, 1.0, 0.4, 0.05,
                    help="Share of a mask's nearest neighbours that carry the same label.")

if res["sig"] != settings_sig:
    st.info("Your model, crop or data settings changed since the last run. Press **Extract embeddings** to update.")

X, df0 = res["X"], res["df"]
pkey = (method, n_nb, min_dist, perp)
if pkey not in res["proj"]:
    with st.spinner(f"Computing {method} projection…"):
        res["proj"][pkey] = project(X, method, n_nb, min_dist, perp)
Y, used_method, proj_note = res["proj"][pkey]
if k_nn not in res["ana"]:
    nbr_i, nbr_s = knn(X, k_nn)
    cons, sugg, sconf = label_consistency(df0["category"].values, nbr_i, nbr_s)
    res["ana"][k_nn] = dict(nbr_i=nbr_i, nbr_s=nbr_s, cons=cons, sugg=sugg, sconf=sconf,
                            cls=class_stats(X, df0["category"].values),
                            sil=silhouette_by_class(X, df0["category"].values))
A = res["ana"][k_nn]

df = df0.copy()
df["x"], df["y"] = Y[:, 0], Y[:, 1]
df["agreement"] = A["cons"]
df["suggested"] = A["sugg"]
df["outlier_z"] = A["cls"]["z"]
df["suspicious"] = df["agreement"] < thr
cats = sorted(df["category"].unique())
CMAP = cat_colors(cats)


def get_overlay(i):
    uid = df.at[i, "uid"]
    if uid not in ss.overlay_cache:
        r = res["records"][i]
        img = load_image(res["images"][r.image_key])
        mask = decode_mask(r.segmentation, img.size[1], img.size[0])
        col = hex_to_rgb(CMAP[r.category]) if CMAP[r.category].startswith("#") else (255, 80, 80)
        if len(ss.overlay_cache) > 60:
            ss.overlay_cache.pop(next(iter(ss.overlay_cache)))
        ss.overlay_cache[uid] = overlay(img, mask, color=col)
    return ss.overlay_cache[uid]


def gallery(idxs, prefix, ncols=4, sims=None, ref_cat=None, limit=48):
    cols = st.columns(ncols)
    for n, j in enumerate(list(idxs)[:limit]):
        with cols[n % ncols]:
            st.image(res["thumbs"][j], width="stretch")
            cap = df.at[j, "category"]
            if ref_cat is not None:
                cap = ("🟢 " if cap == ref_cat else "🔴 ") + cap
            if sims is not None:
                cap += f" · {sims[n]:.2f}"
            st.caption(cap)
            st.button("Inspect", key=f"{prefix}_{j}_{n}", on_click=select_one, args=(j,), width="stretch")


def detail(i, prefix):
    row = df.loc[i]
    st.markdown(f"#### {row.category}")
    if row.suspicious and row.suggested != row.category:
        st.warning(f"Only {row.agreement:.0%} of its nearest neighbours are *{row.category}*. "
                   f"They look more like **{row.suggested}** — worth checking the label.")
    else:
        st.success(f"{row.agreement:.0%} of its nearest neighbours share this label.")
    a, b = st.columns([1, 1.5])
    a.image(res["thumbs"][i], caption="Crop given to the model", width="stretch")
    try:
        b.image(get_overlay(i), caption=f"{row.image}", width="stretch")
    except Exception as e:
        b.warning(f"Can't render the full image: {e}")
    st.caption(f"Annotation {row.ann_id} in {row.coco_file} — {row.area_px:,} px — "
               f"distance to class centre: {A['cls']['dist'][i]:.3f} (z = {row.outlier_z:+.1f})")
    st.markdown("**Most similar masks**")
    gallery(A["nbr_i"][i][:8], f"{prefix}nb{i}", sims=A["nbr_s"][i][:8], ref_cat=row.category)


def group_panel(idxs, prefix):
    sub = df.loc[idxs]
    st.markdown(f"#### {len(sub)} masks selected")
    cnt = sub["category"].value_counts().rename_axis("class").reset_index(name="masks")
    fig = px.bar(cnt, x="masks", y="class", orientation="h", color="class", color_discrete_map=CMAP,
                 height=max(160, 26 * len(cnt)))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), yaxis_title=None)
    st.plotly_chart(fig, width="stretch", key=f"{prefix}_grpbar")
    st.download_button("Download selection (CSV)", sub.drop(columns=["x", "y"]).to_csv(index=False),
                       "selection.csv", key=f"{prefix}_grpdl")
    gallery(idxs, f"{prefix}grp", limit=40)


# =========================================================================== tabs
st.markdown(f"### 🧩 {len(df):,} masks · {len(cats)} classes · {res['model_name']}")
t_map, t_quality, t_classes, t_search, t_export, t_info = st.tabs(
    ["🗺️ Explorer", "🩺 Label quality", "📊 Classes", "🔎 Search", "💾 Export", "ℹ️ Run info"])

# --------------------------------------------------------------------------- explorer
with t_map:
    if proj_note:
        st.caption(proj_note)
    left, right = st.columns([3, 2], gap="large")
    with left:
        c = st.columns([1.1, 2.4, 1])
        color_by = c[0].selectbox("Colour by", ["category", "image", "coco_file", "agreement", "outlier_z", "area_px"])
        vis = c[1].multiselect("Show classes", cats, default=cats)
        ring = c[2].toggle("Ring suspicious", value=False)
        dv = df[df["category"].isin(vis)]
        hover = dict(image=True, ann_id=True, agreement=":.2f", x=False, y=False, idx=False)
        kw = dict(x="x", y="y", custom_data=["idx"], hover_name="category", hover_data=hover, render_mode="webgl")
        if color_by in ("agreement", "outlier_z", "area_px"):
            scale = {"agreement": "RdYlGn", "outlier_z": "Magma_r", "area_px": "Viridis"}[color_by]
            fig = px.scatter(dv, color=color_by, color_continuous_scale=scale, **kw)
        else:
            fig = px.scatter(dv, color=color_by, color_discrete_map=CMAP if color_by == "category" else None, **kw)
        fig.update_traces(marker=dict(size=7, opacity=0.85, line=dict(width=0)))
        if ring:
            fl = dv[dv["suspicious"]]
            fig.add_trace(go.Scattergl(x=fl.x, y=fl.y, mode="markers", name="suspicious", customdata=fl[["idx"]].values,
                                       marker=dict(size=15, symbol="circle-open", color="#E02424", line=dict(width=2)),
                                       hoverinfo="skip"))
        focus = [ss.sel_idx] if ss.sel_idx is not None else ss.sel_group
        if ss.sel_idx is not None:
            nb = A["nbr_i"][ss.sel_idx][:8]
            fig.add_trace(go.Scattergl(x=df.x[nb], y=df.y[nb], mode="markers", name="neighbours",
                                       customdata=np.array(nb)[:, None], hoverinfo="skip",
                                       marker=dict(size=13, symbol="diamond-open", color="#111827", line=dict(width=2))))
        if focus:
            fig.add_trace(go.Scattergl(x=df.x[focus], y=df.y[focus], mode="markers", name="selected",
                                       customdata=np.array(focus)[:, None], hoverinfo="skip",
                                       marker=dict(size=18 if ss.sel_idx is not None else 11, symbol="star",
                                                   color="#FACC15", line=dict(width=1.5, color="#111827"))))
        n_groups = dv[color_by].nunique() if color_by in ("category", "image", "coco_file") else 0
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0), uirevision=str(pkey), dragmode="zoom",
                          legend=dict(orientation="h", y=-0.05, title=None), showlegend=n_groups <= 25)
        fig.update_xaxes(showticklabels=False, title=None, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, title=None, showgrid=False, zeroline=False)
        event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                                selection_mode=("points", "box", "lasso"), key="scatter")
        st.caption(f"{used_method} of {X.shape[1]}-d embeddings. Click a point to inspect it; "
                   "use box or lasso select (top-right toolbar) to grab a group; double-click to reset zoom.")

        picked = []
        pts = (event or {}).get("selection", {}).get("points", []) if isinstance(event, dict) else \
            getattr(getattr(event, "selection", None), "points", []) or []
        for p in pts:
            cd = p.get("customdata")
            if cd is None:
                try:
                    cd = fig.data[p["curve_number"]].customdata[p.get("point_index", p.get("point_number"))]
                except Exception:
                    continue
            picked.append(int(cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd))
        sig = tuple(sorted(set(picked)))
        if sig and sig != ss.last_sig:
            ss.last_sig = sig
            select_one(sig[0]) if len(sig) == 1 else select_group(list(sig))
            st.rerun()

    with right:
        with st.expander("Find a specific mask"):
            opts = df.sort_values(["image", "ann_id"])["idx"].tolist()
            pick = st.selectbox("Mask", [None] + opts, key="find_mask",
                                format_func=lambda j: "—" if j is None else
                                f"{df.at[j, 'image']} · ann {df.at[j, 'ann_id']} · {df.at[j, 'category']}")
            if pick is not None and st.button("Show it", width="stretch"):
                select_one(pick)
                st.rerun()
        if ss.sel_idx is not None:
            detail(ss.sel_idx, "map")
        elif ss.sel_group:
            group_panel(ss.sel_group, "map")
        else:
            st.markdown("#### Click any point")
            st.write("You'll see the crop, where it sits in its image, and its most similar masks. "
                     "Drag a lasso to compare a whole region of the map.")
            q = st.columns(3)
            q[0].metric("kNN label accuracy", f"{(df.suggested == df.category).mean():.0%}",
                        help="How often a mask's neighbours vote for its own label — a quick measure of how well "
                             "this model separates your classes.")
            q[1].metric("Suspicious masks", int(df.suspicious.sum()))
            q[2].metric("Embedding size", X.shape[1])

# --------------------------------------------------------------------------- label quality
with t_quality:
    st.write("A mask whose nearest neighbours mostly carry a different label is often mislabelled, ambiguous "
             "or badly segmented. Fixing these usually improves a model more than adding data.")
    q = st.columns(4)
    q[0].metric("kNN label accuracy", f"{(df.suggested == df.category).mean():.1%}")
    q[1].metric("Suspicious (agreement < %.0f%%)" % (thr * 100), int(df.suspicious.sum()))
    q[2].metric("Mean agreement", f"{df.agreement.mean():.2f}")
    q[3].metric("Class outliers (z > 2.5)", int((df.outlier_z > 2.5).sum()))

    st.markdown("##### Possible label errors")
    flagged = df[df.suspicious].sort_values("agreement").head(300)
    if flagged.empty:
        st.success("Nothing below the threshold. Raise it in ④ Map & analysis to be stricter.")
    else:
        table = pd.DataFrame({
            "crop": [data_uri(res["thumbs"][j]) for j in flagged.idx],
            "label": flagged.category.values, "neighbours say": flagged.suggested.values,
            "agreement": flagged.agreement.values, "image": flagged.image.values,
            "annotation": flagged.ann_id.values})
        ev = st.dataframe(table, hide_index=True, width="stretch", height=380, on_select="rerun",
                          selection_mode="single-row", key="flag_table",
                          column_config={"crop": st.column_config.ImageColumn("Crop", width="small"),
                                         "agreement": st.column_config.ProgressColumn(
                                             "Agreement", min_value=0.0, max_value=1.0, format="%.2f")})
        rows = ev.selection.rows if ev is not None else []
        if rows:
            st.divider()
            detail(int(flagged.idx.iloc[rows[0]]), "qual")
        else:
            st.caption("Select a row to inspect it here.")

    st.markdown("##### Far from their own class")
    st.caption("Masks that are unusually far from their class centre: rare poses, occlusion, bad crops or wrong labels.")
    far = df.sort_values("outlier_z", ascending=False).head(12)
    gallery(far.idx, "far", ncols=6, limit=12)

    st.markdown("##### Near-duplicates")
    dthr = st.slider("Similarity threshold", 0.80, 1.00, 0.97, 0.005)
    dups = near_duplicates(A["nbr_i"], A["nbr_s"], dthr)
    st.caption(f"{len(dups)} pairs. Duplicates across train/test splits inflate your metrics; "
               "duplicates with different labels are contradictions.")
    for a_, b_, s in dups[:12]:
        c = st.columns([1, 1, 3])
        c[0].image(res["thumbs"][a_], width="stretch")
        c[1].image(res["thumbs"][b_], width="stretch")
        same_lbl = df.at[a_, "category"] == df.at[b_, "category"]
        c[2].markdown(f"**{s:.3f}** similarity  \n{df.at[a_, 'image']} ↔ {df.at[b_, 'image']}  \n"
                      + ("same label" if same_lbl else
                         f"⚠️ different labels: {df.at[a_, 'category']} vs {df.at[b_, 'category']}"))

# --------------------------------------------------------------------------- classes
with t_classes:
    cls = A["cls"]
    summary = pd.DataFrame({
        "class": cats,
        "masks": [int((df.category == c).sum()) for c in cats],
        "kNN recall": [float((df.suggested[df.category == c] == c).mean()) for c in cats],
        "mean agreement": [float(df.agreement[df.category == c].mean()) for c in cats],
        "silhouette": [A["sil"].get(c, np.nan) for c in cats],
        "spread": [cls["spread"][c] for c in cats],
    })
    st.markdown("##### How well each class separates")
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "kNN recall": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        "mean agreement": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        "silhouette": st.column_config.NumberColumn(format="%.2f", help="-1…1, higher = better separated"),
        "spread": st.column_config.NumberColumn(format="%.3f", help="Mean cosine distance to the class centre"),
    })
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Class centre similarity")
        st.caption("High off-diagonal values = classes the model sees as alike.")
        fig = px.imshow(cls["sim"], x=cls["classes"], y=cls["classes"], color_continuous_scale="Blues",
                        text_auto=".2f" if len(cats) <= 12 else False, aspect="auto")
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("##### Neighbour confusion")
        st.caption("Row = label, column = what the neighbours vote for (row-normalised).")
        cm = pd.crosstab(pd.Categorical(df.category, cats), pd.Categorical(df.suggested, cats), normalize="index",
                         dropna=False).reindex(index=cats, columns=cats, fill_value=0)
        fig = px.imshow(cm.values, x=cats, y=cats, color_continuous_scale="Purples", zmin=0, zmax=1,
                        text_auto=".2f" if len(cats) <= 12 else False, aspect="auto")
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="neighbours say",
                          yaxis_title="label")
        st.plotly_chart(fig, width="stretch")
    st.markdown("##### Typical vs unusual members")
    ccls = st.selectbox("Class", cats)
    members = df[df.category == ccls].assign(d=cls["dist"][df.category == ccls]).sort_values("d")
    st.caption("Closest to the class centre")
    gallery(members.idx.head(6), "typ", ncols=6, limit=6)
    st.caption("Furthest from the class centre")
    gallery(members.idx.tail(6)[::-1], "atyp", ncols=6, limit=6)

# --------------------------------------------------------------------------- search
with t_search:
    st.markdown("##### Search by example image")
    st.caption("Upload a crop of an object and find the most similar masks. It's embedded with the same model.")
    q_img = st.file_uploader("Query image", type=[e.strip(".") for e in IMAGE_EXTS], key="q_img")
    top = st.slider("Results", 4, 48, 12, 4)
    try:
        search_emb = load_embedder(**res["model_cfg"])
    except Exception as e:
        search_emb = None
        st.warning(f"Can't load the model for search: {e}")
    if q_img is not None and search_emb is not None:
        qi = load_image(q_img.getvalue())
        c = st.columns([1, 5])
        c[0].image(qi, caption="Query", width="stretch")
        with c[1]:
            order, sims = search(X, search_emb.embed([qi])[0], top)
            st.button("Highlight these on the map", on_click=select_group, args=(list(order),), key="hl_img")
            gallery(order, "qimg", ncols=6, sims=sims, limit=top)
    st.divider()
    st.markdown("##### Search by text")
    if res["supports_text"]:
        text = st.text_input("Describe what you're looking for", placeholder="a partly occluded red object")
        if text and search_emb is not None:
            order, sims = search(X, search_emb.embed_text([text])[0], top)
            st.button("Highlight these on the map", on_click=select_group, args=(list(order),), key="hl_txt")
            gallery(order, "qtxt", ncols=6, sims=sims, limit=top)
    else:
        st.caption("Available with CLIP or SigLIP models (or a custom model with `embed_text`).")

# --------------------------------------------------------------------------- export
with t_export:
    st.write("Everything computed here, ready for notebooks or your labelling tool.")
    out = df.drop(columns=["suspicious"]).rename(columns={"x": f"{used_method}_x", "y": f"{used_method}_y"})
    c = st.columns(2)
    c[0].download_button("Masks + map coordinates + scores (CSV)", out.to_csv(index=False), "masks.csv",
                         width="stretch")
    c[0].download_button("Suspicious labels only (CSV)", out[df.suspicious].to_csv(index=False),
                         "suspicious_labels.csv", width="stretch")
    buf = io.BytesIO()
    np.savez_compressed(buf, embeddings=X, uid=df.uid.values.astype(str), category=df.category.values.astype(str),
                        projection=Y)
    c[1].download_button("Embeddings (.npz)", buf.getvalue(), "embeddings.npz", width="stretch")
    if c[1].button("Prepare crops as ZIP (one folder per class)", width="stretch"):
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for j in df.idx:
                ib = io.BytesIO()
                res["thumbs"][j].save(ib, "PNG")
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in df.at[j, "category"])
                z.writestr(f"{safe}/{image_key(df.at[j, 'image'])}_ann{df.at[j, 'ann_id']}.png", ib.getvalue())
        c[1].download_button("Download crops.zip", zbuf.getvalue(), "crops.zip", width="stretch")

# --------------------------------------------------------------------------- run info
with t_info:
    mc = res["model_cfg"]
    st.markdown(f"**Model:** {res['model_name']} ({mc['backend']}, pooling: {mc['pooling']}, "
                f"device: {get_device(mc['device'])})  \n"
                f"**Embedding size:** {X.shape[1]}  \n**Masks embedded:** {len(df):,}  \n"
                f"**Crop:** {res['crop_cfg']['mode']}, padding {res['crop_cfg']['pad']:.0%}, "
                f"square: {res['crop_cfg']['square']}")
    if res["problems"]:
        st.markdown(f"**{len(res['problems'])} issues while cropping**")
        st.dataframe(pd.DataFrame(res["problems"]), hide_index=True, width="stretch")
    else:
        st.success("Every selected mask was embedded without issues.")
