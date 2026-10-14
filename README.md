# 🧩 Mask Embedding Explorer

An interactive dashboard to see **how our model sees segmentation masks**.

Give it images and their COCO annotation files. It crops every mask, runs it through the model, and places
each mask as a point on a 2-D map, so masks the model considers similar end up close together.
Click any point to see the mask, where it sits in its image, and its most similar masks.

Typical uses:

- Check whether the model separates our classes, and which classes it confuses
- Find **probably mislabelled annotations** (masks surrounded by masks of another class)
- Find outliers, unusual examples and near-duplicate annotations
- Search for masks similar to an example image

---

## Contents

1. [Requirements](#requirements)
2. [Installation](#installation-one-time)
3. [Adding the model](#adding-the-model)
4. [Adding your data](#adding-your-data)
5. [Running the app](#running-the-app)
6. [Using the dashboard](#using-the-dashboard)
7. [Using a different model](#using-a-different-model)
8. [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python 3.10 or newer** (check with `python3 --version`)
- macOS, Linux or Windows
- A GPU is optional. The app uses an NVIDIA GPU (CUDA) or an Apple Silicon GPU (MPS) automatically when available, and the CPU otherwise.

## Project structure

```
mask_explorer/
├── app.py               # the dashboard (single file)
├── c2t_embedder.py      # connects our c2t ResNet-50 model to the app
├── requirements.txt     # Python packages
├── model/               # put c2t_resnet50.pt here (not stored in git)
└── data/
    ├── images/          # put your images here (not stored in git)
    └── annotations/     # put your COCO .json files here (not stored in git)
```

The model and data are **not** in the repository: the weights are too large for GitHub, and the data may be confidential.
Each person adds them locally, as described below.

---

## Installation (one time)

**1. Get the code**

```bash
git clone https://github.com/<your-org>/mask_explorer.git
cd mask_explorer
```

**2. Create a virtual environment** (recommended, so the packages don't clash with other projects)

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your terminal prompt now starts with `(.venv)`. Repeat the `activate` line every time you open a new terminal.

**3. Install the packages**

```bash
python3 -m pip install -r requirements.txt
```

This takes a few minutes (PyTorch is large). On Windows, use `python` instead of `python3` everywhere.

> **NVIDIA GPU users:** to use the GPU, install the CUDA build of PyTorch from
> [pytorch.org](https://pytorch.org/get-started/locally/) *before* running the command above.

---

## Adding the model

The model is a ResNet-50 classifier, `c2t_resnet50.pt`, delivered inside a `model.tar.gz` archive.

**Where to get it:** `<fill in: e.g. s3://bucket/path/model.tar.gz, or ask NAME>`

Put the archive in the `model/` folder and unpack it there:

```bash
cd model
tar -xzf model.tar.gz
rm model.tar.gz          # optional: the archive isn't needed after unpacking
cd ..
```

`model/` should now contain `c2t_resnet50.pt` (and `results.json`).

**Check that the model loads:**

```bash
python3 c2t_embedder.py
```

Expected output:

```
Weights: .../mask_explorer/model/c2t_resnet50.pt
Loaded:  ResNet-50 state dict, head: 2 outputs
Device:  mps            (or cuda / cpu)
ResNet backbone found: True   grayscale input: False
Embedding size: 2048  (FEATURES = 'pooled')
Ready to use in the Mask Embedding Explorer.
```

---

## Adding your data

Copy your files into the `data/` folder:

- **Images** → `data/images/` (subfolders are fine; jpg, png, bmp, tif and webp are supported)
- **COCO annotation files** → `data/annotations/` (one or several `.json` files)

Notes:

- One COCO file can describe many images, and you can use several COCO files at once.
- Images are matched to annotations **by file name**, ignoring folders and extensions.
  For example, `train/IMG_001.jpg` in the COCO file matches `data/images/IMG_001.png`.
- Classes are matched **by name**, so COCO files that use different category ids still line up.
- Polygon and RLE segmentations (compressed and uncompressed) are supported.

You can also skip `data/` entirely and upload files in the app, or point the app to any folder on your computer.

---

## Running the app

From the `mask_explorer` folder (with the virtual environment activated):

```bash
python3 -m streamlit run app.py
```

- The first time, Streamlit asks for an email address. This is optional: **just press Enter**.
- The dashboard opens in your browser at **http://localhost:8501**. If it doesn't open, copy that address into the browser.
- Keep the terminal open while you use the app. Press `Ctrl + C` in the terminal to stop it.

To try the app without any model or data, choose **Demo** under *Images & COCO files* and
**Quick test** under *Embedding model*.

---

## Using the dashboard

### Sidebar (left), top to bottom

| Section | What to do |
|---|---|
| **① Images & COCO files** | Pre-filled with `data/images` and `data/annotations` when those folders exist. You can filter classes, ignore tiny masks, or cap the number of masks per class. |
| **② Embedding model** | Pre-selected to *My own model* with `c2t_embedder.py`. Leave as is to use our model. |
| **③ Crop style** | *Bounding box*: the model sees the object with its background. *Mask only*: the background is filled with a flat colour. Pick the one that matches how the model was trained. A preview shows exactly what the model will see. |
| **🚀 Extract embeddings** | Runs the model on every mask. |
| **④ Map & analysis** | Projection method (UMAP, t-SNE, PCA) and analysis settings. Changes apply instantly, with no re-extraction. |

### Tabs

| Tab | What it shows |
|---|---|
| 🗺️ **Explorer** | The map. **Click a point** to see the crop, the mask on the full image, and its 8 nearest neighbours (🟢 same label, 🔴 different label). Use **box or lasso select** (toolbar, top-right of the chart) to inspect a group. Colour points by class, image, COCO file, label agreement, outlier score or area. |
| 🩺 **Label quality** | Masks whose neighbours mostly have another label, with the label the neighbours suggest. Also lists masks far from their class centre, and near-duplicate annotations. |
| 📊 **Classes** | Per-class separation scores, class similarity heatmap, neighbour confusion matrix, and the most typical vs most unusual masks of each class. |
| 🔎 **Search** | Upload an example image to find the most similar masks. |
| 💾 **Export** | CSV with coordinates and scores, CSV of suspicious labels, embeddings (`.npz`), and crops as a ZIP with one folder per class. |
| ℹ️ **Run info** | Settings used, plus any masks that couldn't be processed and why. |

### Reading the map

- Each point is one mask. Points close together are masks **the model** considers similar.
- The axes have no units; only distances and groupings matter.
- **kNN label accuracy** measures how often a mask's nearest neighbours share its label. Higher means the model separates the classes better.
  Compare it across crop styles to see which one suits the model.

---

## Using a different model

Choose a model under **② Embedding model**:

- **Built-in options:** DINOv2, CLIP (also enables text search) and ResNet-50 (ImageNet). Their weights download automatically from Hugging Face on first use.
- **Any Hugging Face model ID or local Hugging Face folder**
- **Your own PyTorch model:** click *Download template* in the app, edit it to load your model, then select the file.
  Any file named `*_embedder.py` placed next to `app.py` is picked up automatically.

### Settings in `c2t_embedder.py`

Settings at the top of the file:

| Setting | Default | Meaning |
|---|---|---|
| `WEIGHTS` | `""` | Empty = look for `model/c2t_resnet50.pt` automatically. Set a full path to use another location. |
| `IMG_SIZE` | `224` | Input size used in training |
| `MEAN`, `STD` | ImageNet values | Normalisation used in training |
| `FEATURES` | `"pooled"` | `"pooled"` = 2048-d features before the final layer (recommended). `"output"` = the model's final 2 outputs. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named streamlit` | The packages aren't installed for this Python. Activate the virtual environment, then run `python3 -m pip install -r requirements.txt`. |
| `command not found: streamlit` | Use `python3 -m streamlit run app.py` instead of `streamlit run app.py`. |
| `cd: no such file or directory` | You're not in the project folder. Run `pwd` and `ls` to see where you are, then `cd` to the folder that contains `app.py`. |
| `Weights not found` | `c2t_resnet50.pt` isn't in `model/`. Unpack the archive inside `model/` (see [Adding the model](#adding-the-model)). |
| "None of the annotations match the images" | Check that the image file names in the COCO file match the files in `data/images/`. |
| "COCO says W×H, file is W×H" in *Run info* | The image was resized after annotation, so its masks will be offset. Use the original images. |
| Port 8501 already in use | Another app is running. Stop it, or run `python3 -m streamlit run app.py --server.port 8502`. |
| Uploading large files fails | Use the `data/` folders or *Local folder* instead, or start with `--server.maxUploadSize 4000` (MB). |
| Extraction is slow | Lower *Max masks per class* in the sidebar, or run on a machine with a GPU. |
