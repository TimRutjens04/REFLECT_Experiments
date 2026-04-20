#!/usr/bin/env python3
"""
visualize.py
------------
Generates three figures from the linear probe experiment:

  1. results_bar.png       — zero-shot AP vs linear probe AP (SigLIP 2 vs CLIP)
  2. results_tsne.png      — t-SNE of frozen embeddings coloured by full/empty
  3. results_crops.png     — sample crop grid with probe predictions

Embeddings are cached to data/embeddings_cache.npz to avoid re-extracting.
"""

import os, sys, pickle
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT        = Path(__file__).parent.parent
DATA        = ROOT / "data"
RESULTS_PNG = ROOT / "results" / "png"
sys.path.insert(0, str(ROOT / "utils"))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from transformers import AutoProcessor, AutoModel
import open_clip

CROPS_PATH  = DATA / "sim_crops.pkl"
CACHE_PATH  = DATA / "embeddings_cache.npz"
BATCH_SIZE  = 32

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Load crops ───────────────────────────────────────────────────────────────
with open(CROPS_PATH, "rb") as f:
    samples = pickle.load(f)

images = [s["image"] for s in samples]
labels = np.array([s["label"] for s in samples])
obj_types = [s["object_type"] for s in samples]

# ── Extract or load cached embeddings ────────────────────────────────────────
def extract_siglip(images):
    proc  = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-384")
    model = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384").to(device).eval()
    embs  = []
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="SigLIP 2"):
        batch  = images[i : i + BATCH_SIZE]
        inputs = proc(images=batch, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        embs.append(pooled.cpu().float().numpy())
    del model
    if device == "mps": torch.mps.empty_cache()
    return np.concatenate(embs)

def extract_clip(images):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()
    embs  = []
    for i in tqdm(range(0, len(images), BATCH_SIZE), desc="CLIP"):
        batch   = images[i : i + BATCH_SIZE]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().float().numpy())
    del model
    if device == "mps": torch.mps.empty_cache()
    return np.concatenate(embs)

if os.path.exists(CACHE_PATH):
    print(f"Loading cached embeddings from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    siglip_emb = cache["siglip"]
    clip_emb   = cache["clip"]
else:
    print("Extracting embeddings (will cache for next run)...")
    siglip_emb = extract_siglip(images)
    clip_emb   = extract_clip(images)
    np.savez(CACHE_PATH, siglip=siglip_emb, clip=clip_emb)
    print(f"Saved → {CACHE_PATH}")

# ── Probe (for prediction labels) ────────────────────────────────────────────
def fit_probe(emb, labels):
    X_tr, X_te, y_tr, y_te = train_test_split(emb, labels, test_size=0.2,
                                               random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y_tr)
    preds  = clf.predict(X_te)
    scores = clf.predict_proba(X_te)[:, 1]
    ap     = average_precision_score(y_te, scores)
    return ap, preds, y_te, X_te

siglip_ap, siglip_preds, siglip_yte, siglip_Xte = fit_probe(siglip_emb, labels)
clip_ap,   clip_preds,   clip_yte,   clip_Xte   = fit_probe(clip_emb,   labels)

# ── Known results ─────────────────────────────────────────────────────────────
ZS_SIGLIP = 0.881   # liquid-fill zero-shot AP from zero_shot_eval.py
ZS_CLIP   = 0.844

# ── Figure 1: Bar chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x      = np.array([0, 1])
width  = 0.32
colors = {"SigLIP 2": "#4C72B0", "CLIP": "#DD8452"}

bars_zs_s  = ax.bar(x[0] - width/2, ZS_SIGLIP,  width, color=colors["SigLIP 2"], alpha=0.55, label="SigLIP 2 — zero-shot")
bars_zs_c  = ax.bar(x[0] + width/2, ZS_CLIP,    width, color=colors["CLIP"],     alpha=0.55, label="CLIP — zero-shot")
bars_lp_s  = ax.bar(x[1] - width/2, siglip_ap,  width, color=colors["SigLIP 2"], alpha=1.0,  label="SigLIP 2 — linear probe")
bars_lp_c  = ax.bar(x[1] + width/2, clip_ap,    width, color=colors["CLIP"],     alpha=1.0,  label="CLIP — linear probe")

for bar in [bars_zs_s, bars_zs_c, bars_lp_s, bars_lp_c]:
    for b in bar:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(["Zero-shot\n(ChangeIt-Frames liquid subset)", "Linear probe\n(AI2-THOR sim crops)"])
ax.set_ylabel("Average Precision (AP)")
ax.set_ylim(0.75, 1.01)
ax.set_title("Zero-shot vs Linear Probe — full/empty state classification")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_PNG / "results_bar.png", dpi=150)
plt.close(fig)
print(f"Saved → {RESULTS_PNG / 'results_bar.png'}")

# ── Figure 2: t-SNE ──────────────────────────────────────────────────────────
print("Computing t-SNE (this takes ~30 s)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

# Subsample if large (300 is fine)
siglip_2d = tsne.fit_transform(siglip_emb)
tsne2     = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
clip_2d   = tsne2.fit_transform(clip_emb)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
label_colors = {0: "#E66", 1: "#66B"}
label_names  = {0: "empty", 1: "full"}

for ax, emb_2d, title, ap in [
    (axes[0], siglip_2d, f"SigLIP 2  (probe AP={siglip_ap:.3f})", siglip_ap),
    (axes[1], clip_2d,   f"CLIP ViT-L/14  (probe AP={clip_ap:.3f})", clip_ap),
]:
    for lbl in [0, 1]:
        mask = labels == lbl
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=label_colors[lbl], label=label_names[lbl],
                   alpha=0.6, s=18, linewidths=0)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("t-SNE of frozen vision encoder features — full vs empty (AI2-THOR sim crops)",
             fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_PNG / "results_tsne.png", dpi=150)
plt.close(fig)
print(f"Saved → {RESULTS_PNG / 'results_tsne.png'}")

# ── Figure 3: Sample crop grid ───────────────────────────────────────────────
# Show 5 correct + 5 incorrect predictions for SigLIP 2 probe on test set
# (use test-set crops only)
_, Xte_indices, _, _ = train_test_split(
    np.arange(len(labels)), labels, test_size=0.2, random_state=42, stratify=labels
)
te_images    = [images[i] for i in Xte_indices]
te_labels    = labels[Xte_indices]
te_obj_types = [obj_types[i] for i in Xte_indices]

correct_idx   = np.where(siglip_preds == siglip_yte)[0]
incorrect_idx = np.where(siglip_preds != siglip_yte)[0]

n_show = min(5, len(correct_idx), len(incorrect_idx))
chosen = (list(correct_idx[:n_show]) + list(incorrect_idx[:n_show]))

fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.4, 5.5))
for col, idx in enumerate(chosen[:n_show]):
    row = 0
    ax  = axes[row, col]
    ax.imshow(te_images[idx])
    ax.set_title(f"GT:{label_names[te_labels[idx]]}\nPred:{label_names[siglip_preds[idx]]}\n{te_obj_types[idx]}",
                 fontsize=7, color="green")
    ax.axis("off")

for col, idx in enumerate(chosen[n_show:]):
    row = 1
    ax  = axes[row, col]
    ax.imshow(te_images[idx])
    ax.set_title(f"GT:{label_names[te_labels[idx]]}\nPred:{label_names[siglip_preds[idx]]}\n{te_obj_types[idx]}",
                 fontsize=7, color="red")
    ax.axis("off")

axes[0, 0].set_ylabel("Correct", fontsize=9, color="green", labelpad=6)
axes[1, 0].set_ylabel("Incorrect", fontsize=9, color="red", labelpad=6)
fig.suptitle("SigLIP 2 linear probe predictions on AI2-THOR test crops", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_PNG / "results_crops.png", dpi=150)
plt.close(fig)
print(f"Saved → {RESULTS_PNG / 'results_crops.png'}")

print("\nAll figures saved.")
