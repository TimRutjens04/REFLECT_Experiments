#!/usr/bin/env python3
"""
linear_probe.py
---------------
Extracts frozen embeddings from SigLIP 2 So400m and CLIP ViT-L/14,
trains a LogisticRegression probe on 80% of the sim crops,
and reports AP + accuracy on the 20% holdout.

Input  : data/sim_crops.pkl  (from generate_sim_crops.py)
Output : results_linear_probe.csv
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, AutoModel
import open_clip
import pandas as pd

CROPS_PATH = DATA / "sim_crops.pkl"
BATCH_SIZE = 32

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")


# ── Load crops ──────────────────────────────────────────────────────────────
with open(CROPS_PATH, "rb") as f:
    samples = pickle.load(f)

images = [s["image"] for s in samples]
labels = np.array([s["label"] for s in samples])
print(f"Loaded {len(samples)} crops  (empty={sum(labels==0)}, full={sum(labels==1)})")


# ── Embedding extraction ─────────────────────────────────────────────────────
def extract_siglip(images):
    print("\nLoading SigLIP 2 So400m...")
    processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-384")
    model     = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384").to(device).eval()

    embeddings = []
    print("Extracting SigLIP 2 embeddings...")
    for i in tqdm(range(0, len(images), BATCH_SIZE)):
        batch = images[i : i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            feats = model.vision_model(**{k: v for k, v in inputs.items()
                                         if k in ("pixel_values",)})
            # pooled output: [B, hidden_dim]
            pooled = feats.pooler_output
        embeddings.append(pooled.cpu().float().numpy())

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return np.concatenate(embeddings, axis=0)


def extract_clip(images):
    print("\nLoading CLIP ViT-L/14...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()

    embeddings = []
    print("Extracting CLIP embeddings...")
    for i in tqdm(range(0, len(images), BATCH_SIZE)):
        batch = images[i : i + BATCH_SIZE]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            feats = model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().float().numpy())

    del model
    if device == "mps":
        torch.mps.empty_cache()

    return np.concatenate(embeddings, axis=0)


# ── Probe ────────────────────────────────────────────────────────────────────
def run_probe(name, embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    scores = clf.predict_proba(X_test)[:, 1]
    ap  = average_precision_score(y_test, scores)
    acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"\n{name}")
    print(f"  AP       : {ap:.3f}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Train n  : {len(y_train)}  Test n: {len(y_test)}")
    return ap, acc


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    siglip_emb = extract_siglip(images)
    clip_emb   = extract_clip(images)

    print("\n" + "=" * 50)
    siglip_ap, siglip_acc = run_probe("SigLIP 2 linear probe", siglip_emb, labels)
    clip_ap,   clip_acc   = run_probe("CLIP linear probe",     clip_emb,   labels)

    # ── Summary vs zero-shot ──────────────────────────────────────────────
    # Zero-shot liquid-fill AP from zero_shot_eval.py results
    zs_siglip = 0.881
    zs_clip   = 0.844

    print("\n" + "=" * 55)
    print(f"{'':25} {'SigLIP 2':>12} {'CLIP':>10}")
    print("-" * 55)
    print(f"{'Zero-shot AP (ChangeIt)':25} {zs_siglip:>12.3f} {zs_clip:>10.3f}")
    print(f"{'Linear probe AP (sim)':25} {siglip_ap:>12.3f} {clip_ap:>10.3f}")
    print(f"{'Linear probe Acc (sim)':25} {siglip_acc:>12.3f} {clip_acc:>10.3f}")
    print("=" * 55)

    pd.DataFrame([
        {"model": "SigLIP2", "zero_shot_ap_changeit": zs_siglip,
         "probe_ap_sim": siglip_ap, "probe_acc_sim": siglip_acc},
        {"model": "CLIP",    "zero_shot_ap_changeit": zs_clip,
         "probe_ap_sim": clip_ap,   "probe_acc_sim": clip_acc},
    ]).to_csv(RESULTS_CSV / "results_linear_probe.csv", index=False)
    print(f"Saved → {RESULTS_CSV / 'results_linear_probe.csv'}")
