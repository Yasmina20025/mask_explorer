"""
Embedder for c2t_resnet50.pt, for the Mask Embedding Explorer
(sidebar → Embedding model → "My own model (Python file)").

Check it on its own first:   python c2t_embedder.py

Works whether the .pt file is a TorchScript model, a whole pickled model,
or a state dict / training checkpoint of a ResNet-50.
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---- settings you may need to change -----------------------------------------------------
WEIGHTS = ""   # leave empty to search automatically (see _find_weights), or give a full path
IMG_SIZE = 224                             # input size used in training
MEAN = [0.485, 0.456, 0.406]               # normalisation used in training (ImageNet default)
STD = [0.229, 0.224, 0.225]
SPATIAL_SIZE = 448    # input size for the difference heatmaps in "Whole images" mode
                      # (448 -> 14×14 grid; larger = finer heatmap, but slower)
FEATURES = "pooled"   # "pooled": 2048-d vector just before the final layer (recommended)
                      # "output": the model's final output (class scores, or its own embedding head)
# ------------------------------------------------------------------------------------------


def _from_state_dict(ckpt):
    sd = ckpt
    for key in ("state_dict", "model_state_dict", "model", "net", "network", "weights"):
        if isinstance(sd.get(key), dict):
            sd = sd[key]
            break
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    anchors = [k for k in sd if k.endswith("conv1.weight") and "layer" not in k]
    if not anchors:
        raise RuntimeError("No ResNet weights found in the file. First keys: " + ", ".join(list(sd)[:8]))
    prefix = min(anchors, key=len)[: -len("conv1.weight")]   # e.g. "module." or "backbone."
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    net = models.resnet50(weights=None)
    note = ""
    if "fc.weight" in sd:
        w = sd["fc.weight"]
        net.fc = nn.Linear(w.shape[1], w.shape[0], bias="fc.bias" in sd)
        note = f", head: {w.shape[0]} outputs"
    else:
        net.fc = nn.Identity()
        note = ", custom head not loaded (pooled features only)"
    if sd["conv1.weight"].shape[1] != 3:
        net.conv1 = nn.Conv2d(sd["conv1.weight"].shape[1], 64, 7, 2, 3, bias=False)
    missing, _ = net.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("fc.")]
    if missing:
        raise RuntimeError(f"The weights don't match a standard ResNet-50 ({len(missing)} layers missing, "
                           f"e.g. {missing[:3]}). Send me the model code and I'll adapt this file.")
    return net, "ResNet-50 state dict" + note


def _load(path):
    try:
        return torch.jit.load(path, map_location="cpu"), "TorchScript model"
    except Exception:
        pass
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except (AttributeError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"The file stores a whole model whose Python class isn't available here ({e}). "
                "At the top of this file add: import sys; sys.path.append('/path/to/your/training/code') "
                "and import the model class.") from e
    if isinstance(obj, nn.Module):
        return obj, f"full model ({type(obj).__name__})"
    if isinstance(obj, dict):
        return _from_state_dict(obj)
    raise RuntimeError(f"Unrecognised file content: {type(obj).__name__}")


def _find_resnet(model):
    """The module that has conv1 ... layer4 (the model itself, or e.g. model.backbone)."""
    for _, m in model.named_modules():
        if all(hasattr(m, a) for a in ("conv1", "bn1", "layer1", "layer4")):
            return m
    return None


def _first_tensor(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        out = list(out.values())
    for o in out:
        if torch.is_tensor(o):
            return o
    raise RuntimeError("The model's output contains no tensor.")


def _find_weights():
    """WEIGHTS if set; otherwise look in model/ next to this file, next to this file, then ~/models/c2t."""
    if WEIGHTS:
        return os.path.expanduser(WEIGHTS)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "model", "c2t_resnet50.pt"), os.path.join(here, "c2t_resnet50.pt"),
              os.path.expanduser("~/models/c2t/c2t_resnet50.pt")):
        if os.path.exists(p):
            return p
    return os.path.join(here, "model", "c2t_resnet50.pt")


class Embedder:
    def __init__(self, device="cpu"):
        path = _find_weights()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found at {path}. Put c2t_resnet50.pt in the model/ folder "
                                    "next to c2t_embedder.py, or set WEIGHTS at the top of c2t_embedder.py.")
        self.weights_path = path
        self.model, self.kind = _load(path)
        self.model.eval()
        self.resnet = _find_resnet(self.model)
        conv1 = getattr(self.resnet, "conv1", None)
        self.gray = conv1 is not None and conv1.weight.shape[1] == 1
        mean, std = ([sum(MEAN) / 3], [sum(STD) / 3]) if self.gray else (MEAN, STD)
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.spatial_transform = transforms.Compose([
            transforms.Resize((SPATIAL_SIZE, SPATIAL_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self.name = f"c2t_resnet50 ({FEATURES} features)"
        self.device = device
        try:                                    # some ops aren't supported on Apple GPUs (mps)
            self.model.to(device)
            self._forward(torch.zeros(1, 1 if self.gray else 3, IMG_SIZE, IMG_SIZE, device=device))
        except Exception:
            self.device = "cpu"
            self.model.to("cpu")

    def _feature_map(self, x):
        r = self.resnet
        act = getattr(r, "relu", None) or getattr(r, "act1")
        x = r.maxpool(act(r.bn1(r.conv1(x))))
        return r.layer4(r.layer3(r.layer2(r.layer1(x))))          # (N, 2048, h, w)

    def _pooled(self, x):
        return self._feature_map(x).mean(dim=(2, 3))

    @torch.no_grad()
    def _forward(self, x):
        if FEATURES == "pooled" and self.resnet is not None:
            return self._pooled(x)
        return _first_tensor(self.model(x)).flatten(1)

    @torch.no_grad()
    def embed(self, images):
        mode = "L" if self.gray else "RGB"
        batch = torch.stack([self.transform(im.convert(mode)) for im in images]).to(self.device)
        return self._forward(batch).float().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_spatial(self, images):
        """Feature maps (N, 2048, h, w) of the last ResNet stage, for the app's difference heatmaps."""
        if self.resnet is None:
            raise RuntimeError("No ResNet backbone found in this model, so spatial feature maps aren't available.")
        mode = "L" if self.gray else "RGB"
        batch = torch.stack([self.spatial_transform(im.convert(mode)) for im in images]).to(self.device)
        return self._feature_map(batch).float().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    e = Embedder(dev)
    print(f"Weights: {e.weights_path}")
    print(f"Loaded:  {e.kind}")
    print(f"Device:  {e.device}")
    print(f"ResNet backbone found: {e.resnet is not None}   grayscale input: {e.gray}")
    v = e.embed([Image.new("RGB", (300, 200), (120, 90, 60)), Image.new("RGB", (64, 64), (10, 200, 30))])
    print(f"Embedding size: {v.shape[1]}  (FEATURES = '{FEATURES}')")
    if e.resnet is not None:
        sp = e.embed_spatial([Image.new("RGB", (300, 200), (120, 90, 60))])
        print(f"Heatmap grid:   {sp.shape[2]}×{sp.shape[3]}  (SPATIAL_SIZE = {SPATIAL_SIZE})")
    print("Ready to use in the Mask Embedding Explorer.")
