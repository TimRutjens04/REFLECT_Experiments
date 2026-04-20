#!/usr/bin/env python3
"""
cross_domain_probe.py
---------------------
Sim-to-real transfer test:
  - Train: AI2-THOR sim crops (full/empty, Bowl/Cup/Pot) — all 300 samples
  - Test : ChangeIt-Frames liquid categories (beer, juice, milk)

Measures the sim-to-real gap by comparing:
  (a) zero-shot AP on ChangeIt (from zero_shot_eval.py)
  (b) sim-trained linear probe AP on the same ChangeIt crops

Embeddings for sim crops loaded from cache (data/embeddings_cache.npz).
ChangeIt embeddings extracted fresh and cached to data/changeit_liquid_cache.npz.
"""

import os, sys, pickle
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT        = Path(__file__).parent.parent
DATA        = ROOT / "data"
RESULTS_CSV = ROOT / "results" / "csv"
sys.path.insert(0, str(ROOT / "utils"))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score
from transformers import AutoProcessor, AutoModel
import open_clip

# ── Paths ────────────────────────────────────────────────────────────────────
SIM_CROPS_PATH    = DATA / "sim_crops.pkl"
SIM_CACHE_PATH    = DATA / "embeddings_cache.npz"
CIT_CACHE_PATH    = DATA / "changeit_liquid_cache.npz"
ANNOT_DIR         = DATA / "annotations"
CROP_DIR          = DATA / "ChangeIT-Subset-Crop"
LIQUID_CATS       = {"beer", "juice", "milk"}   # tea excluded: no both-class data
BATCH_SIZE        = 32

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

# ── Load sim training data ────────────────────────────────────────────────────
with open(SIM_CROPS_PATH, "rb") as f:
    sim_samples = pickle.load(f)
sim_labels = np.array([s["label"] for s in sim_samples])

cache = np.load(SIM_CACHE_PATH)
sim_siglip = cache["siglip"]
sim_clip   = cache["clip"]
print(f"Sim crops: {len(sim_samples)}  (empty={sum(sim_labels==0)}, full={sum(sim_labels==1)})")

# ── Build ChangeIt liquid test set ────────────────────────────────────────────
def build_changeit_liquid():
    """Returns (images, labels, cats) for liquid categories, labels 0 and 3 only."""
    import pandas as pd_local
    crop_folders = set(os.listdir(CROP_DIR))
    images, labels, cats = [], [], []

    for cat in sorted(LIQUID_CATS):
        cat_path = os.path.join(ANNOT_DIR, cat)
        if not os.path.isdir(cat_path):
            continue
        for csv_file in sorted(os.listdir(cat_path)):
            if not csv_file.endswith(".csv"):
                continue
            video_id = csv_file.split(".")[0]
            if video_id not in crop_folders:
                continue
            df = pd_local.read_csv(os.path.join(cat_path, csv_file),
                                   header=None, index_col=0)
            label_map = df[1].to_dict()
            crop_folder = os.path.join(CROP_DIR, video_id)
            for fname in sorted(os.listdir(crop_folder)):
                if not fname.endswith(".jpg"):
                    continue
                parts = fname.split("_")
                if len(parts) < 3:
                    continue
                try:
                    frame_idx = int(parts[2].split(".")[0])
                except ValueError:
                    continue
                label = label_map.get(frame_idx)
                if label == 0:
                    images.append(Image.open(os.path.join(crop_folder, fname)).convert("RGB"))
                    labels.append(0)
                    cats.append(cat)
                elif label == 3:
                    images.append(Image.open(os.path.join(crop_folder, fname)).convert("RGB"))
                    labels.append(1)
                    cats.append(cat)

    return images, np.array(labels), cats

cit_images, cit_labels, cit_cats = build_changeit_liquid()
print(f"ChangeIt liquid: {len(cit_images)} frames  "
      f"(empty={sum(cit_labels==0)}, full={sum(cit_labels==1)})")
print(f"  Categories: { {c: cit_cats.count(c) for c in LIQUID_CATS} }")

# ── Extract or load ChangeIt embeddings ───────────────────────────────────────
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

if os.path.exists(CIT_CACHE_PATH):
    print(f"\nLoading cached ChangeIt embeddings from {CIT_CACHE_PATH}")
    cit_cache = np.load(CIT_CACHE_PATH)
    cit_siglip = cit_cache["siglip"]
    cit_clip   = cit_cache["clip"]
else:
    print("\nExtracting ChangeIt embeddings...")
    cit_siglip = extract_siglip(cit_images)
    cit_clip   = extract_clip(cit_images)
    np.savez(CIT_CACHE_PATH, siglip=cit_siglip, clip=cit_clip)
    print(f"Saved → {CIT_CACHE_PATH}")

# ── Sim-trained probe → ChangeIt evaluation ───────────────────────────────────
def cross_domain_ap(train_emb, train_labels, test_emb, test_labels, name):
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(train_emb, train_labels)
    scores = clf.predict_proba(test_emb)[:, 1]
    preds  = clf.predict(test_emb)
    ap  = average_precision_score(test_labels, scores)
    acc = accuracy_score(test_labels, preds)
    print(f"\n{name}")
    print(f"  Cross-domain AP  : {ap:.3f}")
    print(f"  Cross-domain Acc : {acc:.3f}")
    return ap, acc

siglip_xd_ap, siglip_xd_acc = cross_domain_ap(
    sim_siglip, sim_labels, cit_siglip, cit_labels, "SigLIP 2 — sim→real probe")
clip_xd_ap, clip_xd_acc = cross_domain_ap(
    sim_clip, sim_labels, cit_clip, cit_labels, "CLIP — sim→real probe")

# ── Full results table ────────────────────────────────────────────────────────
ZS_SIGLIP    = 0.881   # liquid zero-shot AP (ChangeIt), from zero_shot_eval.py
ZS_CLIP      = 0.844
PROBE_SIGLIP = 0.954   # sim linear probe AP, from linear_probe.py
PROBE_CLIP   = 0.868

print("\n" + "=" * 62)
print(f"{'':35} {'SigLIP 2':>10} {'CLIP':>10}")
print("-" * 62)
print(f"{'Zero-shot AP (ChangeIt liquid)':35} {ZS_SIGLIP:>10.3f} {ZS_CLIP:>10.3f}")
print(f"{'Linear probe AP (sim in-domain)':35} {PROBE_SIGLIP:>10.3f} {PROBE_CLIP:>10.3f}")
print(f"{'Sim→real probe AP (ChangeIt)':35} {siglip_xd_ap:>10.3f} {clip_xd_ap:>10.3f}")
print(f"{'Sim→real probe Acc (ChangeIt)':35} {siglip_xd_acc:>10.3f} {clip_xd_acc:>10.3f}")
print("=" * 62)
print(f"\nSim-to-real AP gap:")
print(f"  SigLIP 2 : {PROBE_SIGLIP:.3f} (sim) → {siglip_xd_ap:.3f} (real)  Δ={siglip_xd_ap - PROBE_SIGLIP:+.3f}")
print(f"  CLIP     : {PROBE_CLIP:.3f} (sim) → {clip_xd_ap:.3f} (real)  Δ={clip_xd_ap - PROBE_CLIP:+.3f}")

pd.DataFrame([
    {"model": "SigLIP2", "zs_ap_changeit": ZS_SIGLIP,
     "probe_ap_sim": PROBE_SIGLIP, "xd_ap_changeit": siglip_xd_ap, "xd_acc_changeit": siglip_xd_acc},
    {"model": "CLIP", "zs_ap_changeit": ZS_CLIP,
     "probe_ap_sim": PROBE_CLIP, "xd_ap_changeit": clip_xd_ap, "xd_acc_changeit": clip_xd_acc},
]).to_csv(RESULTS_CSV / "results_cross_domain.csv", index=False)
print(f"\nSaved → {RESULTS_CSV / 'results_cross_domain.csv'}")
