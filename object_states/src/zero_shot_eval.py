#!/usr/bin/env python3
"""
zero_shot_eval.py
-----------------
Zero-shot evaluation of SigLIP 2 So400m and CLIP ViT-L/14 on ChangeIt-Frames crops.

Label scheme (from annotation CSVs):
  0 = initial steady state  → binary label 0
  3 = terminal steady state → binary label 1
  1, 2 = transition frames  → skipped

Metric: Average Precision (AP) per category, then mean AP across categories.
"""

import os, sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT        = Path(__file__).parent.parent
DATA        = ROOT / "data"
RESULTS_CSV = ROOT / "results" / "csv"
sys.path.insert(0, str(ROOT / "utils"))

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModel
import open_clip

from utils.classification_states import a  # {cat: {"initial": [...], "terminal": [...]}}

# ── Device ─────────────────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")

ANNOT_DIR  = DATA / "annotations"
CROP_DIR   = DATA / "ChangeIT-Subset-Crop"
BATCH_SIZE = 16

# ── Build dataset ───────────────────────────────────────────────────────────
def build_dataset():
    """Returns {cat: [(img_path, binary_label), ...]} using only labels 0 and 3."""
    crop_folders = set(os.listdir(CROP_DIR))
    dataset = {}

    for cat in sorted(os.listdir(ANNOT_DIR)):
        cat_path = os.path.join(ANNOT_DIR, cat)
        if not os.path.isdir(cat_path) or cat not in a:
            continue

        entries = []
        for csv_file in sorted(os.listdir(cat_path)):
            if not csv_file.endswith(".csv"):
                continue
            video_id = csv_file.split(".")[0]
            if video_id not in crop_folders:
                continue

            df = pd.read_csv(
                os.path.join(cat_path, csv_file), header=None, index_col=0
            )
            label_map = df[1].to_dict()  # {frame_idx: label}

            crop_folder = os.path.join(CROP_DIR, video_id)
            for fname in sorted(os.listdir(crop_folder)):
                if not fname.endswith(".jpg"):
                    continue
                # filename: cropped_frame_NNNN.jpg
                parts = fname.split("_")
                if len(parts) < 3:
                    continue
                try:
                    frame_idx = int(parts[2].split(".")[0])
                except ValueError:
                    continue

                label = label_map.get(frame_idx)
                if label == 0:
                    entries.append((os.path.join(crop_folder, fname), 0))
                elif label == 3:
                    entries.append((os.path.join(crop_folder, fname), 1))
                # labels 1, 2 (transition frames) → skip

        if entries and len({y for _, y in entries}) == 2:
            dataset[cat] = entries

    return dataset


# ── Scoring helpers ─────────────────────────────────────────────────────────
def terminal_score_from_logits(logits, n_initial):
    """
    logits: [B, n_prompts]  (initial prompts first, terminal prompts last)
    Returns scalar terminal probability per image via softmax of max logits.
    """
    init_score = logits[:, :n_initial].max(dim=1).values
    term_score = logits[:, n_initial:].max(dim=1).values
    stacked    = torch.stack([init_score, term_score], dim=1)
    return torch.softmax(stacked, dim=1)[:, 1].cpu().numpy()


def eval_siglip(cat, entries, model, processor):
    init_prompts = a[cat]["initial"]
    term_prompts = a[cat]["terminal"]
    all_prompts  = init_prompts + term_prompts
    n_initial    = len(init_prompts)

    scores, labels = [], []
    for i in range(0, len(entries), BATCH_SIZE):
        batch  = entries[i : i + BATCH_SIZE]
        paths, ys = zip(*batch)
        images = [Image.open(p).convert("RGB") for p in paths]

        inputs = processor(
            text=all_prompts, images=images,
            return_tensors="pt", padding="max_length",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits_per_image  # [B, n_prompts]

        scores.extend(terminal_score_from_logits(logits, n_initial))
        labels.extend(ys)

    return average_precision_score(labels, scores)


def eval_clip(cat, entries, model, preprocess, tokenizer):
    init_prompts = a[cat]["initial"]
    term_prompts = a[cat]["terminal"]
    all_prompts  = init_prompts + term_prompts
    n_initial    = len(init_prompts)

    text_tokens = tokenizer(all_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores, labels = [], []
    for i in range(0, len(entries), BATCH_SIZE):
        batch  = entries[i : i + BATCH_SIZE]
        paths, ys = zip(*batch)
        imgs   = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)

        with torch.no_grad():
            img_features = model.encode_image(imgs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = (img_features @ text_features.T) * model.logit_scale.exp()

        scores.extend(terminal_score_from_logits(logits, n_initial))
        labels.extend(ys)

    return average_precision_score(labels, scores)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = build_dataset()
    total_frames = sum(len(v) for v in dataset.values())
    print(f"Categories with both classes: {len(dataset)}")
    print(f"Total frames (label 0 + label 3): {total_frames}\n")

    # ── SigLIP 2 ──────────────────────────────────────────────────────────
    print("Loading SigLIP 2 So400m (google/siglip2-so400m-patch16-384)...")
    siglip_processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch16-384")
    siglip_model     = AutoModel.from_pretrained("google/siglip2-so400m-patch16-384").to(device).eval()

    siglip_aps = {}
    print("Running SigLIP 2 zero-shot...")
    for cat, entries in tqdm(dataset.items()):
        siglip_aps[cat] = eval_siglip(cat, entries, siglip_model, siglip_processor)

    del siglip_model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # ── CLIP ViT-L/14 ─────────────────────────────────────────────────────
    print("\nLoading CLIP ViT-L/14 (openai)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model     = clip_model.to(device).eval()

    clip_aps = {}
    print("Running CLIP zero-shot...")
    for cat, entries in tqdm(dataset.items()):
        clip_aps[cat] = eval_clip(cat, entries, clip_model, clip_preprocess, clip_tokenizer)

    # ── Results table ──────────────────────────────────────────────────────
    all_cats = sorted(set(siglip_aps) | set(clip_aps))
    n_frames_per_cat = {cat: len(dataset[cat]) for cat in all_cats}

    print("\n" + "=" * 65)
    print(f"{'Category':<20} {'n_frames':>8} {'SigLIP2 AP':>12} {'CLIP AP':>10}")
    print("-" * 65)
    for cat in all_cats:
        s = siglip_aps.get(cat, float("nan"))
        c = clip_aps.get(cat, float("nan"))
        n = n_frames_per_cat.get(cat, 0)
        print(f"{cat:<20} {n:>8} {s:>12.3f} {c:>10.3f}")
    print("-" * 65)

    mean_s = np.mean(list(siglip_aps.values()))
    mean_c = np.mean(list(clip_aps.values()))
    print(f"{'Mean AP':<20} {'':>8} {mean_s:>12.3f} {mean_c:>10.3f}")
    print("=" * 65)

    # Liquid-fill subset (closest to full/empty in AI2-THOR)
    liquid_cats = [c for c in all_cats if c in {"beer", "juice", "milk", "tea"}]
    if liquid_cats:
        liq_s = np.mean([siglip_aps[c] for c in liquid_cats if c in siglip_aps])
        liq_c = np.mean([clip_aps[c]   for c in liquid_cats if c in clip_aps])
        print(f"\nLiquid-fill subset {liquid_cats}")
        print(f"  SigLIP2 mean AP: {liq_s:.3f}")
        print(f"  CLIP    mean AP: {liq_c:.3f}")

    # Save results to CSV
    rows = [
        {
            "category": cat,
            "n_frames": n_frames_per_cat.get(cat, 0),
            "siglip2_ap": siglip_aps.get(cat, float("nan")),
            "clip_ap": clip_aps.get(cat, float("nan")),
        }
        for cat in all_cats
    ]
    rows.append({
        "category": "MEAN",
        "n_frames": total_frames,
        "siglip2_ap": mean_s,
        "clip_ap": mean_c,
    })
    pd.DataFrame(rows).to_csv(RESULTS_CSV / "results_zero_shot.csv", index=False)
    print(f"\nSaved → {RESULTS_CSV / 'results_zero_shot.csv'}")
